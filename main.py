"""
AI Meeting Presenter — FastAPI backend
Exposes REST + WebSocket APIs consumed by the dashboard and external services.
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import config
from orchestrator.session import PresentationSession, SessionEvent

# ------------------------------------------------------------------ #
# Bootstrap
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

for d in (config.UPLOAD_DIR, config.AUDIO_CACHE_DIR, config.SLIDE_IMAGE_DIR, "static"):
    Path(d).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AI Meeting Presenter",
    description="Autonomous AI sales presenter — joins meetings, presents slides, talks & listens.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory session store (swap for Redis in production)
sessions: Dict[str, PresentationSession] = {}
ws_clients: Dict[str, List[WebSocket]] = {}


# ------------------------------------------------------------------ #
# WebSocket broadcast helper
# ------------------------------------------------------------------ #

def _make_broadcaster(session_id: str):
    async def broadcast(event: SessionEvent):
        dead = []
        for ws in ws_clients.get(session_id, []):
            try:
                await ws.send_json(event.to_dict())
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients[session_id].remove(ws)
    return broadcast


# ------------------------------------------------------------------ #
# Dashboard
# ------------------------------------------------------------------ #

@app.get("/", include_in_schema=False)
async def dashboard():
    return FileResponse("static/index.html")


# ------------------------------------------------------------------ #
# Session management
# ------------------------------------------------------------------ #

@app.post("/api/sessions", summary="Create a new presentation session")
async def create_session():
    session = PresentationSession()
    sessions[session.session_id] = session
    ws_clients[session.session_id] = []
    session.add_listener(_make_broadcaster(session.session_id))
    logger.info("Session created: %s", session.session_id)
    return {"session_id": session.session_id}


@app.get("/api/sessions/{session_id}", summary="Get session status")
async def get_session(session_id: str):
    session = _get_or_404(session_id)
    return session.status()


@app.get("/api/sessions", summary="List all active sessions")
async def list_sessions():
    return [s.status() for s in sessions.values()]


# ------------------------------------------------------------------ #
# Slide upload
# ------------------------------------------------------------------ #

@app.post("/api/sessions/{session_id}/slides", summary="Upload presentation slides")
async def upload_slides(session_id: str, file: UploadFile = File(...)):
    session = _get_or_404(session_id)

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pptx", ".ppt", ".pdf"):
        raise HTTPException(400, f"Unsupported format '{ext}'. Use PPTX or PDF.")

    dest = os.path.join(config.UPLOAD_DIR, f"{session_id}{ext}")
    async with aiofiles.open(dest, "wb") as fh:
        await fh.write(await file.read())

    try:
        count = session.load_slides(dest)
    except Exception as exc:
        raise HTTPException(422, f"Failed to parse slides: {exc}") from exc

    return {"slides_count": count, "filename": file.filename}


# ------------------------------------------------------------------ #
# Start / Control / End
# ------------------------------------------------------------------ #

class StartRequest(BaseModel):
    meeting_url: Optional[str] = None


@app.post("/api/sessions/{session_id}/start", summary="Start the presentation")
async def start_session(session_id: str, body: StartRequest = StartRequest()):
    session = _get_or_404(session_id)
    if not session.slides.total_slides:
        raise HTTPException(400, "Upload slides before starting")
    asyncio.create_task(session.start(meeting_url=body.meeting_url))
    return {"message": "Presentation starting", "session_id": session_id}


class ControlRequest(BaseModel):
    action: str                     # next | previous | goto | pause | resume | repeat | end
    slide_number: Optional[int] = None


@app.post("/api/sessions/{session_id}/control", summary="Manual presenter control")
async def control_session(session_id: str, body: ControlRequest):
    session = _get_or_404(session_id)
    await session.manual_control(body.action, body.slide_number)
    return {"message": f"Action '{body.action}' executed"}


class TranscriptRequest(BaseModel):
    text: str


@app.post("/api/sessions/{session_id}/transcript", summary="Submit a voice transcript")
async def submit_transcript(session_id: str, body: TranscriptRequest):
    """
    Accepts transcripts from:
    - The browser (Web Speech API via dashboard)
    - Recall.ai webhook proxy
    - Deepgram direct push
    """
    session = _get_or_404(session_id)
    asyncio.create_task(session.process_transcript(body.text))
    return {"message": "Processing"}


@app.post("/api/sessions/{session_id}/end", summary="End the session")
async def end_session(session_id: str):
    session = _get_or_404(session_id)
    await session.end()
    return {"message": "Session ended"}


# ------------------------------------------------------------------ #
# Recall.ai webhook
# ------------------------------------------------------------------ #

@app.post("/webhook/recall/{session_id}", include_in_schema=False)
async def recall_webhook(session_id: str, payload: dict):
    """Receives real-time transcripts from Recall.ai."""
    from meeting.recall_bot import RecallBot

    text = RecallBot.parse_transcript_webhook(payload)
    if text and session_id in sessions:
        asyncio.create_task(sessions[session_id].process_transcript(text))
    return {"ok": True}


# ------------------------------------------------------------------ #
# Audio file serving
# ------------------------------------------------------------------ #

@app.get("/audio/{session_id}/{filename}", include_in_schema=False)
async def serve_audio(session_id: str, filename: str):
    path = os.path.join(config.AUDIO_CACHE_DIR, f"session_{session_id}_{filename}")
    if not os.path.exists(path):
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type="audio/mpeg")


# ------------------------------------------------------------------ #
# Slide image serving
# ------------------------------------------------------------------ #

@app.get("/slides/{session_id}/{filename}", include_in_schema=False)
async def serve_slide_image(session_id: str, filename: str):
    path = os.path.join("slide_images", session_id, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Slide image not found")
    return FileResponse(path, media_type="image/png")


# ------------------------------------------------------------------ #
# WebSocket — real-time event stream
# ------------------------------------------------------------------ #

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    if session_id not in sessions:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    ws_clients[session_id].append(websocket)
    logger.info("WebSocket connected: %s", session_id)

    # Push current state immediately on connect
    await websocket.send_json({
        "type": "status",
        "data": sessions[session_id].status(),
        "timestamp": "",
    })

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                import json
                msg = json.loads(raw)
                # Client can push transcripts via WebSocket too
                if msg.get("type") == "transcript" and "text" in msg:
                    asyncio.create_task(
                        sessions[session_id].process_transcript(msg["text"])
                    )
                elif msg.get("type") == "control":
                    asyncio.create_task(
                        sessions[session_id].manual_control(
                            msg.get("action", ""),
                            msg.get("slide_number"),
                        )
                    )
            except Exception as exc:
                logger.debug("WS message parse error: %s", exc)

    except WebSocketDisconnect:
        if websocket in ws_clients.get(session_id, []):
            ws_clients[session_id].remove(websocket)
        logger.info("WebSocket disconnected: %s", session_id)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _get_or_404(session_id: str) -> PresentationSession:
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return session


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info",
    )
