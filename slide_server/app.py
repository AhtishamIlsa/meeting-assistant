"""
Slide Presentation Server
─────────────────────────
GET  /            — Setup UI (upload slides, enter meeting URL, start)
GET  /present     — Full-screen slide viewer (shared via joinly share_screen)
POST /api/upload  — Upload PPTX or PDF; returns {slides_count, filename}
GET  /api/info    — Current slide state {index, total, title, content, notes}
POST /api/navigate— Navigate slides {action: next|previous|goto|repeat, slide_number?}
GET  /slides/{n}  — Serve slide PNG image
WS   /ws          — Push slide-change events to the /present page
"""
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from slide_server.manager import SlideManager

logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
IMAGE_DIR = "slide_images"
Path(UPLOAD_DIR).mkdir(exist_ok=True)
Path(IMAGE_DIR).mkdir(exist_ok=True)

app = FastAPI(title="AI Meeting Presenter — Slide Server")

# Shared state
manager = SlideManager(image_dir=IMAGE_DIR)
_ws_clients: list[WebSocket] = []

# ------------------------------------------------------------------ #
# Real-time transcript queue
# Recall pushes transcript events here via /api/recall/events webhook.
# The presenter agent drains this queue in _poll_loop.
# ------------------------------------------------------------------ #

recall_transcript_q: asyncio.Queue = asyncio.Queue()


@app.post("/api/recall/events")
async def recall_webhook(request: Request):
    """Receive real-time transcript events from Recall.ai realtime_endpoints."""
    try:
        data = await request.json()
        event = data.get("event", "unknown")
        inner = (data.get("data") or {}).get("data") or {}
        participant = inner.get("participant") or {}
        speaker = participant.get("name") or inner.get("speaker") or "unknown"
        words = inner.get("words") or []
        preview = " ".join(
            w.get("text") or w.get("word") or ""
            for w in words[:12]
            if isinstance(w, dict)
        ).strip()
        logger.info(
            "Recall realtime event: %s speaker=%s preview=%s",
            event,
            speaker,
            preview[:120] or "-",
        )
        await recall_transcript_q.put(data)
    except Exception as exc:
        body = await request.body()
        logger.warning("Recall webhook parse error %s — body: %s", exc, body[:200])
    return {"ok": True}


# ------------------------------------------------------------------ #
# Audio sink registry
# Maps session_id → (asyncio.Queue, event_loop) for the presenter agent.
# Recall connects to /ws/audio/{session_id} and we forward raw PCM to
# the agent's queue using call_soon_threadsafe (cross-loop safe).
# ------------------------------------------------------------------ #

_audio_sinks: dict[str, tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = {}


def register_audio_sink(
    session_id: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Called by RecallPresenterAgent before the bot is created."""
    _audio_sinks[session_id] = (queue, loop)
    logger.info("Audio sink registered: %s", session_id)


def unregister_audio_sink(session_id: str) -> None:
    """Called by RecallPresenterAgent on shutdown."""
    _audio_sinks.pop(session_id, None)
    logger.info("Audio sink unregistered: %s", session_id)


@app.websocket("/ws/audio/{session_id}")
async def ws_audio(ws: WebSocket, session_id: str):
    """
    Recall pushes raw meeting audio (S16LE 16 kHz mono) here as binary frames.
    We forward each frame to the presenter agent's asyncio.Queue using
    call_soon_threadsafe so it is safe to call from a different event loop.
    """
    await ws.accept()
    logger.info("Recall audio WebSocket connected: %s", session_id)
    sink = _audio_sinks.get(session_id)
    if not sink:
        logger.warning("No audio sink registered for session %s — closing", session_id)
        await ws.close(code=1008)
        return
    queue, loop = sink
    try:
        while True:
            data = await ws.receive_bytes()
            loop.call_soon_threadsafe(queue.put_nowait, data)
    except WebSocketDisconnect:
        logger.info("Recall audio WebSocket disconnected: %s", session_id)
    except Exception as exc:
        logger.warning("Audio WebSocket error (%s): %s", session_id, exc)


# ------------------------------------------------------------------ #
# WebSocket broadcast (called by agent when slide changes)
# ------------------------------------------------------------------ #

async def broadcast_slide_change(index: int, total: int):
    payload = json.dumps({"type": "slide_change", "index": index, "total": total})
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    # Send current state immediately
    if manager.is_loaded:
        slide = manager.current()
        await ws.send_text(json.dumps({
            "type": "slide_change",
            "index": slide.index,
            "total": slide.total,
        }))
    try:
        while True:
            await ws.receive_text()  # keep-alive
    except WebSocketDisconnect:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ------------------------------------------------------------------ #
# Setup UI  (served at /)
# ------------------------------------------------------------------ #

_SETUP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI Meeting Presenter — Setup</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>body{background:#0a0e1a;color:#fff;font-family:system-ui,sans-serif}</style>
</head>
<body class="min-h-screen flex items-center justify-center p-6">
  <div class="w-full max-w-lg bg-white/5 border border-white/10 rounded-2xl p-8">
    <div class="flex items-center gap-3 mb-6">
      <span class="text-3xl">🤖</span>
      <div>
        <h1 class="text-xl font-bold">AI Meeting Presenter</h1>
        <p class="text-xs text-gray-400">Powered by joinly.ai + Claude</p>
      </div>
    </div>

    <!-- Upload -->
    <label class="block text-sm font-medium text-gray-300 mb-2">📊 Slides (PPTX or PDF)</label>
    <div id="drop" onclick="document.getElementById('fi').click()"
      class="border-2 border-dashed border-white/20 rounded-xl p-8 text-center cursor-pointer hover:border-blue-500 transition mb-2">
      <div class="text-4xl mb-2">📁</div>
      <p class="text-gray-400 text-sm">Drag & drop or click to browse</p>
    </div>
    <input id="fi" type="file" accept=".pptx,.ppt,.pdf" class="hidden">
    <p id="uploadStatus" class="text-xs text-gray-400 mb-5 min-h-4"></p>

    <!-- Meeting URL -->
    <label class="block text-sm font-medium text-gray-300 mb-2">🎥 Zoom / Meet URL</label>
    <input id="meetingUrl" type="url" placeholder="https://zoom.us/j/..."
      class="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm mb-6
             focus:outline-none focus:border-blue-500">

    <!-- Start -->
    <button id="btnStart" disabled
      class="w-full bg-gradient-to-r from-blue-600 to-purple-600 disabled:opacity-40
             rounded-xl py-3 font-semibold text-sm transition"
      onclick="startPresentation()">
      🚀 Start AI Presentation
    </button>
    <p id="startStatus" class="text-xs text-center text-gray-400 mt-3 min-h-4"></p>
  </div>

<script>
  let slidesReady = false;

  document.getElementById('drop').addEventListener('dragover', e => e.preventDefault());
  document.getElementById('drop').addEventListener('drop', e => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });
  document.getElementById('fi').addEventListener('change', e => {
    if (e.target.files[0]) uploadFile(e.target.files[0]);
  });

  async function uploadFile(file) {
    const st = document.getElementById('uploadStatus');
    st.textContent = `Uploading ${file.name}…`;
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    if (!res.ok) {
      const d = await res.json();
      st.textContent = '❌ ' + (d.detail || 'Upload failed');
      return;
    }
    const d = await res.json();
    st.textContent = `✅ ${d.slides_count} slides loaded from "${d.filename}"`;
    st.classList.add('text-green-400');
    slidesReady = true;
    document.getElementById('btnStart').disabled = false;
  }

  async function startPresentation() {
    const url = document.getElementById('meetingUrl').value.trim();
    if (!url) { alert('Enter a meeting URL'); return; }
    const st = document.getElementById('startStatus');
    st.textContent = 'Starting…';
    const res = await fetch('/api/start', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ meeting_url: url }),
    });
    if (!res.ok) {
      st.textContent = '❌ ' + (await res.json()).detail;
      return;
    }
    st.textContent = '✅ Presentation agent started! Check terminal for logs.';
    document.getElementById('btnStart').disabled = true;
  }
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def setup_ui():
    return _SETUP_HTML


# ------------------------------------------------------------------ #
# Presentation viewer  (shared in meeting via share_screen)
# ------------------------------------------------------------------ #

_PRESENT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Presentation</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background:#000; width:100vw; height:100vh; overflow:hidden;
           display:flex; align-items:center; justify-content:center; }
    #slide { max-width:100vw; max-height:100vh; object-fit:contain; }
    #counter { position:fixed; bottom:12px; right:16px; color:rgba(255,255,255,.35);
               font:14px/1 system-ui; letter-spacing:.04em; }
    #loading { color:rgba(255,255,255,.3); font:18px system-ui; }
  </style>
</head>
<body>
  <div id="loading">Loading slides…</div>
  <img id="slide" style="display:none" alt="Slide">
  <div id="counter"></div>

<script>
  const img = document.getElementById('slide');
  const counter = document.getElementById('counter');
  const loading = document.getElementById('loading');

  function showSlide(index, total) {
    img.src = '/slides/' + index + '?t=' + Date.now();
    img.style.display = 'block';
    loading.style.display = 'none';
    counter.textContent = (index + 1) + ' / ' + total;
  }

  // Fetch current state immediately
  fetch('/api/info').then(r => r.json()).then(d => {
    if (d.total > 0) showSlide(d.index, d.total);
  });

  // Live updates via WebSocket
  const ws = new WebSocket((location.protocol==='https:'?'wss':'ws') + '://' + location.host + '/ws');
  ws.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.type === 'slide_change') showSlide(d.index, d.total);
  };
  ws.onclose = () => setTimeout(() => location.reload(), 2000);
</script>
</body>
</html>"""


@app.get("/present", response_class=HTMLResponse)
async def present_view():
    return _PRESENT_HTML


# ------------------------------------------------------------------ #
# API
# ------------------------------------------------------------------ #

@app.post("/api/upload")
async def upload_slides(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pptx", ".ppt", ".pdf"):
        raise HTTPException(400, f"Unsupported format '{ext}'. Use PPTX or PDF.")

    dest = os.path.join(UPLOAD_DIR, f"slides{ext}")
    async with aiofiles.open(dest, "wb") as fh:
        await fh.write(await file.read())

    # Clear stale PNG cache AND derived PDF so new slides are rendered fresh
    import glob
    for old_img in glob.glob(os.path.join(IMAGE_DIR, "slide_*.png")):
        try:
            os.remove(old_img)
        except OSError:
            pass
    for old_pdf in glob.glob(os.path.join(UPLOAD_DIR, "slides*.pdf")):
        try:
            os.remove(old_pdf)
        except OSError:
            pass

    try:
        count = manager.load(dest)
    except Exception as exc:
        raise HTTPException(422, f"Failed to parse slides: {exc}") from exc

    # Pre-render all slide images in background
    import asyncio
    asyncio.create_task(_prerender_images())

    return {"slides_count": count, "filename": file.filename}


async def _prerender_images():
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, manager.export_all_images)
    logger.info("All slide images rendered")


@app.get("/api/info")
async def slide_info():
    if not manager.is_loaded:
        return {"index": 0, "total": 0, "title": "", "content": "", "notes": ""}
    s = manager.current()
    return {
        "index": s.index,
        "total": s.total,
        "title": s.title,
        "content": s.content,
        "notes": s.speaker_notes,
    }


class NavigateRequest(BaseModel):
    action: str           # next | previous | goto | repeat
    slide_number: Optional[int] = None   # 1-based, only for goto


@app.post("/api/navigate")
async def navigate(req: NavigateRequest):
    if not manager.is_loaded:
        raise HTTPException(400, "No slides loaded")

    if req.action == "next":
        slide = manager.next()
    elif req.action == "previous":
        slide = manager.previous()
    elif req.action == "goto":
        if req.slide_number is None:
            raise HTTPException(400, "slide_number required for goto")
        slide = manager.goto(req.slide_number - 1)
    elif req.action == "repeat":
        slide = manager.current()
    else:
        raise HTTPException(400, f"Unknown action '{req.action}'")

    await broadcast_slide_change(slide.index, slide.total)
    return {
        "index": slide.index,
        "total": slide.total,
        "title": slide.title,
        "content": slide.content,
        "notes": slide.speaker_notes,
        "summary": slide.summary(),
    }


# /api/start is called by the setup UI — triggers the presenter agent
_start_callback = None   # set by main.py


class StartRequest(BaseModel):
    meeting_url: str


@app.post("/api/start")
async def start_presentation(req: StartRequest):
    if not manager.is_loaded:
        raise HTTPException(400, "Upload slides first")
    if _start_callback is None:
        raise HTTPException(503, "Presenter agent not configured")
    import asyncio
    asyncio.create_task(_start_callback(req.meeting_url))
    return {"message": "Presentation agent starting"}


@app.get("/slides/{index}")
async def serve_slide_image(index: int):
    path = manager.export_image(index)
    if path and os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    raise HTTPException(404, "Slide image not found")
