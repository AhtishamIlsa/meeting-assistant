"""
AI Meeting Presenter — Recall.ai Agent
──────────────────────────────────────
Uses Recall.ai REST API for meeting join, screen share, and TTS.
GPT-4o Realtime handles conversation intelligence.

Flow:
  1. Create Recall bot → bot joins the meeting
  2. Push current slide as JPEG to output_screenshare (no public URL needed)
  3. Poll /transcript/ every 1.5 s for new participant speech
  4. Speech → GPT-4o Realtime → text response + tool calls
  5. Text → ElevenLabs → MP3 → base64 → Recall output_audio
  6. navigate_slide → slide server API + push new JPEG frame

Screen sharing works without any ngrok/tunnel because we push rendered slide
images directly as JPEG frames to Recall's output_screenshare endpoint.
"""
import asyncio
import base64
import io
import json
import logging
import os
import time
from typing import Optional

import httpx
from openai import AsyncOpenAI
from PIL import Image

from config import config
from presenter.prompt import get_presenter_prompt
from slide_server.manager import SlideManager

logger = logging.getLogger(__name__)

SLIDE_BASE = f"http://localhost:{config.SLIDE_SERVER_PORT}"
_SLIDE_IMAGE_DIR = "slide_images"

try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE = Image.LANCZOS  # type: ignore[attr-defined]

_TOOLS = [
    {
        "type": "function",
        "name": "navigate_slide",
        "description": (
            "Navigate the presentation slides. "
            "Call when the client says next / previous / go to slide N / repeat."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["next", "previous", "goto", "repeat"]},
                "slide_number": {
                    "type": "integer",
                    "description": "1-based slide number — only required for 'goto'",
                },
            },
            "required": ["action"],
        },
    },
    {
        "type": "function",
        "name": "get_current_slide",
        "description": "Return the current slide title, content, and speaker notes.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "leave_meeting",
        "description": "Leave the meeting. Call when the client says goodbye / done / end.",
        "parameters": {"type": "object", "properties": {}},
    },
]


class RecallPresenterAgent:
    def __init__(self, slide_manager: SlideManager):
        self.manager = slide_manager
        self._last_word_end: float = 0.0
        self._transcript_id: Optional[str] = None

    @property
    def _base(self) -> str:
        return f"https://{config.RECALL_REGION}.recall.ai/api/v1"

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Token {config.RECALL_API_KEY}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    async def run(self, meeting_url: str) -> None:
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as http:
            bot_id = await self._create_bot(http, meeting_url)
            logger.info("Created Recall bot %s", bot_id)

            await self._wait_for_joined(http, bot_id)
            logger.info("Bot joined meeting")

            # Push first slide immediately
            first = self.manager.current()
            if first:
                asyncio.create_task(self._push_slide(http, bot_id, first.index))

            openai = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            async with openai.beta.realtime.connect(
                model="gpt-4o-realtime-preview-2024-12-17"
            ) as rt:
                logger.info("Connected to OpenAI Realtime API")

                await rt.session.update(session={
                    "modalities": ["text"],
                    "instructions": get_presenter_prompt(config.PRESENTER_NAME),
                    "tools": _TOOLS,
                    "tool_choice": "auto",
                    "temperature": 0.6,
                })

                turn_queue: asyncio.Queue[str] = asyncio.Queue()

                if first:
                    summary = first.summary()
                    logger.info("First slide content:\n%s", summary)
                    await turn_queue.put(
                        f"The presentation has started. Present slide 1 now.\n\n"
                        f"SLIDE CONTENT — narrate exactly this, do not change or invent:\n{summary}"
                    )

                async def _worker() -> None:
                    while True:
                        text = await turn_queue.get()
                        try:
                            await self._turn(rt, http, bot_id, text)
                        except Exception:
                            logger.exception("Turn error")
                        finally:
                            turn_queue.task_done()

                worker = asyncio.create_task(_worker())
                logger.info("Entering main loop — polling transcript every 1.5 s")
                try:
                    while True:
                        await asyncio.sleep(1.5)
                        chunk = await self._poll_transcript(http, bot_id)
                        if chunk:
                            logger.info("Client said: %s", chunk)
                            await turn_queue.put(chunk)
                finally:
                    worker.cancel()

    # ------------------------------------------------------------------ #
    # Bot lifecycle
    # ------------------------------------------------------------------ #

    async def _create_bot(self, http: httpx.AsyncClient, meeting_url: str) -> str:
        payload: dict = {
            "meeting_url": meeting_url,
            "bot_name": config.PRESENTER_NAME,
            "recording_config": {
                "transcript": {
                    "provider": {"deepgram_streaming": {"language": "en"}},
                },
            },
        }
        resp = await http.post(f"{self._base}/bot/", json=payload)
        if not resp.is_success:
            logger.error("Recall create_bot failed %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        data = resp.json()
        logger.info("Bot created: %s", json.dumps(data))
        return data["id"]

    async def _wait_for_joined(
        self, http: httpx.AsyncClient, bot_id: str, timeout: int = 90
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = await http.get(f"{self._base}/bot/{bot_id}/")
            resp.raise_for_status()
            changes = resp.json().get("status_changes") or []
            code = changes[-1]["code"] if changes else "created"
            logger.info("Bot status: %s", code)
            if code in ("in_call_not_recording", "in_call_recording"):
                return
            if code in ("call_ended", "done", "fatal"):
                raise RuntimeError(f"Bot failed to join: {code}")
            await asyncio.sleep(3)
        raise TimeoutError("Bot did not join within 90 s")

    # ------------------------------------------------------------------ #
    # Transcript polling
    # ------------------------------------------------------------------ #

    async def _poll_transcript(
        self, http: httpx.AsyncClient, bot_id: str
    ) -> Optional[str]:
        try:
            # Resolve transcript artifact ID once from recording entry
            if not self._transcript_id:
                bot_resp = await http.get(f"{self._base}/bot/{bot_id}/")
                bot_resp.raise_for_status()
                recs = bot_resp.json().get("recordings") or []
                if not recs:
                    return None
                shortcuts = (recs[0].get("media_shortcuts") or {})
                t = shortcuts.get("transcript") or {}
                self._transcript_id = t.get("id")
                if not self._transcript_id:
                    logger.warning("No transcript artifact yet in recording")
                    return None
                logger.info("Transcript artifact ID: %s", self._transcript_id)

            resp = await http.get(f"{self._base}/transcript/{self._transcript_id}/")
            if not resp.is_success:
                logger.warning("Transcript poll %s: %s", resp.status_code, resp.text[:200])
                return None
            data = resp.json()
            # Log the response format once so we can see the structure
            if not hasattr(self, "_transcript_format_logged"):
                self._transcript_format_logged = True
                logger.info("Transcript response sample: %s", str(data)[:800])
            return self._parse_words(data)
        except Exception as exc:
            logger.warning("Transcript poll: %s", exc)
            return None

    def _parse_words(self, data) -> Optional[str]:
        """Convert Recall transcript response → speaker-labelled text, or None if nothing new."""
        new_words: list[tuple[float, str, str]] = []

        # Normalise to a flat list of word dicts with speaker, text, end_time
        if isinstance(data, list):
            # Old format: [{speaker, words:[{text, end_time}]}, ...]
            for seg in data:
                if not isinstance(seg, dict):
                    continue
                speaker = seg.get("speaker") or "Participant"
                for w in seg.get("words", []):
                    end = w.get("end_time") or w.get("end_timestamp", 0)
                    if end and end > self._last_word_end:
                        new_words.append((end, speaker, w.get("text", "")))
        elif isinstance(data, dict):
            # New artifact format: top-level "words" array
            raw_words = data.get("words") or []
            for w in raw_words:
                if not isinstance(w, dict):
                    continue
                speaker = w.get("speaker") or w.get("speaker_id") or "Participant"
                end = w.get("end_time") or w.get("end_timestamp") or 0
                if end and end > self._last_word_end:
                    new_words.append((end, speaker, w.get("text", "")))

        if not new_words:
            return None

        new_words.sort(key=lambda x: x[0])
        self._last_word_end = new_words[-1][0]

        lines: list[str] = []
        cur_speaker: Optional[str] = None
        cur_buf: list[str] = []
        for _, speaker, word in new_words:
            if speaker != cur_speaker:
                if cur_buf:
                    lines.append(f"{cur_speaker}: {' '.join(cur_buf)}")
                cur_speaker, cur_buf = speaker, [word]
            else:
                cur_buf.append(word)
        if cur_buf:
            lines.append(f"{cur_speaker}: {' '.join(cur_buf)}")
        return "\n".join(lines) or None

    # ------------------------------------------------------------------ #
    # Conversation turn
    # ------------------------------------------------------------------ #

    async def _turn(self, rt, http: httpx.AsyncClient, bot_id: str, text: str) -> None:
        await rt.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        })
        await rt.response.create()
        await self._drain(rt, http, bot_id)

    async def _drain(self, rt, http: httpx.AsyncClient, bot_id: str) -> None:
        response_text = ""
        tool_calls: dict[str, dict] = {}

        async for event in rt:
            t = event.type
            if t == "response.text.delta":
                response_text += event.delta
            elif t == "response.output_item.done":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    tool_calls[item.call_id] = {
                        "name": item.name,
                        "arguments": item.arguments or "{}",
                    }
            elif t == "response.done":
                break
            elif t == "error":
                logger.error("Realtime error: %s", event)
                break

        if response_text.strip():
            logger.info("Speaking: %.120s", response_text)
            asyncio.create_task(self._speak(http, bot_id, response_text))

        if tool_calls:
            for call_id, tc in tool_calls.items():
                result = await self._run_tool(http, bot_id, tc["name"], json.loads(tc["arguments"]))
                logger.info("Tool %s → %.80s", tc["name"], result)
                await rt.conversation.item.create(item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                })
            await rt.response.create()
            await self._drain(rt, http, bot_id)

    # ------------------------------------------------------------------ #
    # TTS — OpenAI tts-1 (primary) → MP3 → base64 → Recall output_audio
    # ------------------------------------------------------------------ #

    async def _speak(self, http: httpx.AsyncClient, bot_id: str, text: str) -> None:
        try:
            audio = await self._openai_tts(text)
        except Exception as exc:
            logger.error("TTS failed: %s", exc)
            return
        b64 = base64.b64encode(audio).decode()
        try:
            resp = await http.post(
                f"{self._base}/bot/{bot_id}/output_audio/",
                json={"kind": "mp3", "b64_data": b64},
            )
            if not resp.is_success:
                logger.error("output_audio failed %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("output_audio failed: %s", exc)

    async def _openai_tts(self, text: str) -> bytes:
        """Generate MP3 via OpenAI tts-1 — no extra credits needed beyond the API key."""
        openai = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        response = await openai.audio.speech.create(
            model="tts-1",
            voice="nova",       # natural female voice; options: alloy echo fable onyx nova shimmer
            input=text,
            response_format="mp3",
        )
        return response.content

    # ------------------------------------------------------------------ #
    # Screen share — slide PNG → JPEG 1280×720 → base64 → output_screenshare
    # ------------------------------------------------------------------ #

    async def _push_slide(self, http: httpx.AsyncClient, bot_id: str, slide_index: int) -> None:
        img_path = os.path.join(_SLIDE_IMAGE_DIR, f"slide_{slide_index:03d}.png")
        # Wait up to 15 s for the image to be rendered
        for _ in range(15):
            if os.path.exists(img_path):
                break
            await asyncio.sleep(1)
        if not os.path.exists(img_path):
            logger.warning("Slide image not found after waiting: %s", img_path)
            return
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((1280, 720), _RESAMPLE)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode()
            resp = await http.post(
                f"{self._base}/bot/{bot_id}/output_screenshare/",
                json={"kind": "jpeg", "b64_data": b64},
            )
            resp.raise_for_status()
            logger.info("Screen share updated → slide %d", slide_index)
        except Exception as exc:
            logger.warning("Screen share push failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #

    async def _run_tool(
        self, http: httpx.AsyncClient, bot_id: str, name: str, args: dict
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=10) as slide_http:
                if name == "navigate_slide":
                    payload: dict = {"action": args.get("action", "next")}
                    if args.get("action") == "goto" and args.get("slide_number"):
                        payload["slide_number"] = args["slide_number"]
                    resp = await slide_http.post(f"{SLIDE_BASE}/api/navigate", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    asyncio.create_task(self._push_slide(http, bot_id, data["index"]))
                    result = data["summary"]
                    logger.info("navigate_slide → %s", result)
                    return result

                if name == "get_current_slide":
                    resp = await slide_http.get(f"{SLIDE_BASE}/api/info")
                    resp.raise_for_status()
                    d = resp.json()
                    if d["total"] == 0:
                        return "No slides loaded."
                    parts = [f"Slide {d['index']+1}/{d['total']}: {d['title']}"]
                    if d["content"]:
                        parts.append(d["content"])
                    if d["notes"]:
                        parts.append(f"[Notes: {d['notes']}]")
                    result = "\n".join(parts)
                    logger.info("get_current_slide → %s", result)
                    return result

            if name == "leave_meeting":
                resp = await http.post(f"{self._base}/bot/{bot_id}/leave_call/")
                resp.raise_for_status()
                return "Left the meeting."

            return f"Unknown tool: {name}"
        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc)
            return f"Error: {exc}"
