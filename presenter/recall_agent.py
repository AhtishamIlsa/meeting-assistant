"""
AI Meeting Presenter — Recall.ai Agent
────────────────────────────────────────
Pipeline:

  Recall bot joins meeting
    │
    ├─► audio_mixed_raw.data over Recall websocket
    │       raw PCM → local Deepgram STT → transcript text
    │           │
    │     OpenAI Realtime API  (modalities: ["text"])
    │     • text-in / text-out — no audio processing in the model
    │     • tool calls: navigate_slide, get_current_slide, leave_meeting
    │     • full conversation history maintained across turns
    │           │
    │     response text  (_tts_worker)
    │           │
    │     ElevenLabs eleven_turbo_v2_5  (fallback: OpenAI TTS-HD)
    │     • natural, human-quality voice
    │           │
    └─── output_audio ◄── MP3 POST to Recall

    ├─► output_screenshare ◄── slide JPEG on each navigation

Engagement flow (per slide):
  narration done → 8 s grace → check-in question → 10 s → advance slide

End of deck:
  bot summarises questions raised + next steps → stays for Q&A

Concurrent coroutines (asyncio.gather):
  _audio_input_worker — mixed audio websocket data → Deepgram STT
  _poll_loop   — final STT transcripts → inject turns into Realtime
  _event_handler — RT events → text buffer → _text_out_q + tool calls
  _tts_worker  — ElevenLabs/OpenAI TTS → Recall output_audio
  _auto_advance — engagement timer → check-in → slide advance → summary
  _refresh_participants — fetch names → update session instructions
"""
import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
import wave
from urllib.parse import urlencode, urlparse, urlunparse
from typing import Optional

import httpx
from openai import AsyncOpenAI
from PIL import Image

from config import config
from presenter.prompt import get_presenter_prompt
from slide_server.app import register_audio_sink, unregister_audio_sink
from slide_server.manager import SlideManager
from voice.stt_engine import create_stt_engine

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
        self._text_out_q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._turn_q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._audio_in_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._recall_event_q: asyncio.Queue[Optional[dict]] = asyncio.Queue()
        self._stt_text_q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._last_emitted_chunk: str = ""
        self._last_emitted_at: float = 0.0
        self._audio_event_seen: bool = False
        self._audio_warning_emitted: bool = False
        self._stt = None
        self._stt_backend: str = "deepgram"
        self._participant_speaking: bool = False
        self._active_speaker: Optional[str] = None
        self._speech_buffer = bytearray()
        self._speech_tasks: set[asyncio.Task] = set()
        self._stt_openai: Optional[AsyncOpenAI] = None
        self._realtime_session_id: Optional[str] = None
        # Coordination events
        self._user_spoke: asyncio.Event = asyncio.Event()
        self._audio_finished: asyncio.Event = asyncio.Event()
        self._cancel_audio: asyncio.Event = asyncio.Event()
        self._rt_idle: asyncio.Event = asyncio.Event()
        self._rt_write_lock: Optional[asyncio.Lock] = None
        self._interrupted: bool = False
        # Realtime connection handle (set while run() is active)
        self._rt = None
        self._response_active: bool = False
        # Meeting memory
        self._questions: list[str] = []
        self._summary_given: bool = False

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
        self._text_out_q = asyncio.Queue()
        self._turn_q = asyncio.Queue()
        self._audio_in_q = asyncio.Queue()
        self._recall_event_q = asyncio.Queue()
        self._stt_text_q = asyncio.Queue()
        self._last_emitted_chunk = ""
        self._last_emitted_at = 0.0
        self._audio_event_seen = False
        self._audio_warning_emitted = False
        self._stt_backend = "deepgram"
        self._participant_speaking = False
        self._active_speaker = None
        self._speech_buffer = bytearray()
        self._speech_tasks = set()
        self._stt_openai = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self._realtime_session_id = uuid.uuid4().hex
        self._user_spoke.clear()
        self._audio_finished.clear()
        self._cancel_audio.clear()
        self._rt_idle = asyncio.Event()
        self._rt_idle.set()
        self._rt_write_lock = asyncio.Lock()
        self._interrupted = False
        self._rt = None
        self._response_active = False
        self._questions = []
        self._summary_given = False
        loop = asyncio.get_running_loop()
        register_audio_sink(
            self._realtime_session_id,
            self._audio_in_q,
            loop,
            event_queue=self._recall_event_q,
        )
        self._start_stt(loop)

        try:
            async with httpx.AsyncClient(headers=self._headers, timeout=30) as http:
                bot_id = await self._create_bot(http, meeting_url)
                logger.info("Bot created: %s", bot_id)

                await self._wait_for_joined(http, bot_id)
                logger.info("Bot joined meeting")

                first = self.manager.current()
                if first:
                    asyncio.create_task(self._push_slide(http, bot_id, first.index))

                openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
                async with openai_client.beta.realtime.connect(
                    model="gpt-4o-realtime-preview"
                ) as rt:
                    self._rt = rt
                    logger.info("Connected to OpenAI Realtime")

                    await rt.session.update(session={
                        "modalities": ["text"],
                        "instructions": get_presenter_prompt(config.PRESENTER_NAME),
                        "tools": _TOOLS,
                        "tool_choice": "auto",
                        "temperature": 0.6,
                    })

                    tasks = [
                        asyncio.create_task(
                            self._turn_worker(rt), name="turn_worker"
                        ),
                        asyncio.create_task(
                            self._audio_input_worker(), name="audio_input_worker"
                        ),
                        asyncio.create_task(
                            self._recall_event_worker(), name="recall_event_worker"
                        ),
                        asyncio.create_task(
                            self._poll_loop(), name="poll_loop"
                        ),
                        asyncio.create_task(
                            self._event_handler(rt, http, bot_id), name="event_handler"
                        ),
                        asyncio.create_task(
                            self._tts_worker(http, bot_id), name="tts_worker"
                        ),
                        asyncio.create_task(
                            self._auto_advance(rt, http, bot_id), name="auto_advance"
                        ),
                        asyncio.create_task(
                            self._refresh_participants(rt, http, bot_id),
                            name="refresh_participants",
                        ),
                    ]

                    if first:
                        await self._queue_turn(
                            "The presentation has started. Present slide 1 now.\n\n"
                            f"SLIDE CONTENT — narrate exactly this:\n{first.summary()}"
                        )

                    try:
                        await asyncio.gather(*tasks)
                    except Exception:
                        logger.exception("Pipeline error — shutting down")
                        for t in tasks:
                            t.cancel()
                        raise
                    finally:
                        self._rt = None
                        self._response_active = False
        finally:
            for task in list(self._speech_tasks):
                task.cancel()
            if self._speech_tasks:
                await asyncio.gather(*self._speech_tasks, return_exceptions=True)
                self._speech_tasks.clear()
            if self._stt:
                try:
                    self._stt.stop()
                except Exception:
                    logger.exception("Could not stop STT engine cleanly")
                self._stt = None
            if self._stt_openai:
                await self._stt_openai.close()
                self._stt_openai = None
            if self._realtime_session_id:
                unregister_audio_sink(self._realtime_session_id)

    # ------------------------------------------------------------------ #
    # Bot lifecycle
    # ------------------------------------------------------------------ #

    async def _create_bot(self, http: httpx.AsyncClient, meeting_url: str) -> str:
        realtime_url = self._build_realtime_ws_url()
        logger.info("Recall realtime websocket: %s", realtime_url)

        payload: dict = {
            "meeting_url": meeting_url,
            "bot_name": config.PRESENTER_NAME,
            "recording_config": {
                "audio_mixed_raw": {},
                "realtime_endpoints": [{
                    "type": "websocket",
                    "url": realtime_url,
                    "events": [
                        "audio_mixed_raw.data",
                        "participant_events.speech_on",
                        "participant_events.speech_off",
                    ],
                }],
            },
        }
        resp = await http.post(f"{self._base}/bot/", json=payload)
        if not resp.is_success:
            logger.error("create_bot failed %s: %s", resp.status_code, resp.text)
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

    def _build_realtime_ws_url(self) -> str:
        source = (
            config.RECALL_REALTIME_WS_BASE_URL.strip()
            or config.RECALL_WEBHOOK_URL.strip()
        )
        if not source:
            raise RuntimeError(
                "Set RECALL_REALTIME_WS_BASE_URL to your public ngrok/cloudflared URL "
                f"that forwards to localhost:{config.SLIDE_SERVER_PORT}"
            )
        if not self._realtime_session_id:
            raise RuntimeError("Realtime session id is not initialised")

        parsed = urlparse(source)
        if not parsed.scheme or not parsed.netloc:
            raise RuntimeError(
                "RECALL_REALTIME_WS_BASE_URL must be a full public URL like "
                "https://your-subdomain.ngrok-free.app"
            )

        scheme = {
            "https": "wss",
            "http": "ws",
            "wss": "wss",
            "ws": "ws",
        }.get(parsed.scheme, parsed.scheme)

        base_path = ""
        if config.RECALL_REALTIME_WS_BASE_URL.strip():
            base_path = parsed.path.rstrip("/")

        path = f"{base_path}/ws/recall/{self._realtime_session_id}"
        query = ""
        token = config.RECALL_REALTIME_TOKEN.strip()
        if token:
            query = urlencode({"token": token})

        return urlunparse((scheme, parsed.netloc, path, "", query, ""))

    def _start_stt(self, loop: asyncio.AbstractEventLoop) -> None:
        def _on_transcript(text: str) -> None:
            loop.call_soon_threadsafe(self._stt_text_q.put_nowait, text)

        def _on_error(exc: Exception) -> None:
            loop.call_soon_threadsafe(self._enable_openai_stt_fallback, exc)

        try:
            self._stt = create_stt_engine(_on_transcript, on_error=_on_error)
            self._stt.start()
            self._stt_backend = "deepgram"
            logger.info("Local streaming STT enabled (Deepgram)")
        except Exception as exc:
            self._stt = None
            self._enable_openai_stt_fallback(exc)

    def _enable_openai_stt_fallback(
        self,
        exc: Optional[Exception] = None,
    ) -> None:
        if self._stt_backend == "openai_segmented":
            return
        self._stt_backend = "openai_segmented"
        self._stt = None
        self._participant_speaking = False
        self._active_speaker = None
        self._speech_buffer = bytearray()
        if exc:
            logger.warning(
                "Deepgram STT unavailable (%s). Falling back to OpenAI segmented transcription.",
                exc,
            )
        else:
            logger.info("Using OpenAI segmented transcription fallback")

    # ------------------------------------------------------------------ #
    # Realtime request scheduling
    # ------------------------------------------------------------------ #

    async def _queue_turn(self, text: str) -> None:
        text = text.strip()
        if text:
            await self._turn_q.put(text)

    async def _turn_worker(self, rt) -> None:
        while True:
            text = await self._turn_q.get()
            try:
                if text is None:
                    return
                await self._rt_idle.wait()
                self._rt_idle.clear()
                await self._send_user_turn(rt, text)
            except Exception:
                logger.exception("Failed to submit realtime turn")
                self._rt_idle.set()
            finally:
                self._turn_q.task_done()

    async def _send_user_turn(self, rt, text: str) -> None:
        if self._rt_write_lock is None:
            raise RuntimeError("Realtime write lock is not initialised")
        async with self._rt_write_lock:
            await rt.conversation.item.create(item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            })
            await rt.response.create()

    async def _send_tool_outputs(
        self,
        rt,
        outputs: list[tuple[str, str]],
    ) -> None:
        if self._rt_write_lock is None:
            raise RuntimeError("Realtime write lock is not initialised")
        async with self._rt_write_lock:
            for call_id, output in outputs:
                await rt.conversation.item.create(item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                })
            await rt.response.create()

    async def _update_instructions(self, rt, instructions: str) -> None:
        if self._rt_write_lock is None:
            raise RuntimeError("Realtime write lock is not initialised")
        async with self._rt_write_lock:
            await rt.session.update(session={"instructions": instructions})

    async def _cancel_response(self) -> None:
        """
        Send response.cancel to the Realtime API when a participant barges in.
        The server will emit response.cancelled (handled in _event_handler),
        which sets _rt_idle so the turn_worker can immediately send the new turn.
        """
        if not self._response_active or self._rt is None or self._rt_write_lock is None:
            return
        async with self._rt_write_lock:
            if not self._response_active:
                return  # already cancelled under lock
            try:
                await self._rt.response.cancel()
                logger.info("Barge-in: sent response.cancel to Realtime API")
            except Exception as exc:
                logger.warning("response.cancel failed: %s", exc)
                # Ensure idle is set so the pipeline doesn't stall
                self._response_active = False
                self._rt_idle.set()

    # ------------------------------------------------------------------ #
    # Participant awareness
    # ------------------------------------------------------------------ #

    async def _fetch_participants(
        self, http: httpx.AsyncClient, bot_id: str
    ) -> list[str]:
        try:
            resp = await http.get(f"{self._base}/bot/{bot_id}/")
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("meeting_participants") or data.get("participants") or []
            return [
                p["name"]
                for p in raw
                if isinstance(p, dict)
                and p.get("name")
                and p["name"].lower() != config.PRESENTER_NAME.lower()
            ]
        except Exception as exc:
            logger.warning("Could not fetch participants: %s", exc)
            return []

    async def _refresh_participants(
        self, rt, http: httpx.AsyncClient, bot_id: str
    ) -> None:
        """Wait 15 s for latecomers then inject participant names into session."""
        await asyncio.sleep(15)
        names = await self._fetch_participants(http, bot_id)
        if names:
            logger.info("Participants: %s", names)
            await self._update_instructions(
                rt,
                get_presenter_prompt(config.PRESENTER_NAME, participants=names),
            )
        else:
            logger.info("No participant names retrieved")

    # ------------------------------------------------------------------ #
    # Audio input + STT
    # ------------------------------------------------------------------ #

    async def _recall_event_worker(self) -> None:
        while True:
            payload = await self._recall_event_q.get()
            try:
                if payload is None:
                    return

                event = str(payload.get("event") or "")
                speaker = str(payload.get("speaker") or "unknown").strip() or "unknown"
                if speaker.lower() == config.PRESENTER_NAME.lower():
                    continue

                if event == "participant_events.speech_on":
                    logger.info("Participant speech started: %s", speaker)
                    self._user_spoke.set()
                    self._cancel_audio.set()
                    self._participant_speaking = True
                    self._active_speaker = speaker
                    self._speech_buffer = bytearray()
                    # Cancel any in-progress Realtime response so the pipeline
                    # doesn't keep generating text that the TTS worker would
                    # queue and play after the barge-in.
                    asyncio.create_task(
                        self._cancel_response(), name="barge_in_cancel"
                    )
                    continue

                if event != "participant_events.speech_off":
                    continue

                logger.info("Participant speech ended: %s", speaker)
                active_speaker = self._active_speaker or speaker
                audio = bytes(self._speech_buffer)
                self._participant_speaking = False
                self._active_speaker = None
                self._speech_buffer = bytearray()

                if (
                    self._stt_backend == "openai_segmented"
                    and len(audio) >= 6400
                ):
                    task = asyncio.create_task(
                        self._transcribe_openai_segment(audio, active_speaker),
                        name="openai_stt_segment",
                    )
                    self._speech_tasks.add(task)
                    task.add_done_callback(self._speech_tasks.discard)
            finally:
                self._recall_event_q.task_done()

    async def _audio_input_worker(self) -> None:
        """
        Receive raw PCM frames from the Recall websocket endpoint and push them
        into the local Deepgram streaming connection.
        """
        start_wait = time.monotonic()

        while True:
            try:
                audio = await asyncio.wait_for(self._audio_in_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if (
                    not self._audio_event_seen
                    and not self._audio_warning_emitted
                    and time.monotonic() - start_wait > 20
                ):
                    self._audio_warning_emitted = True
                    logger.warning(
                        "No Recall audio websocket data received after 20s. "
                        "Start ngrok on port %s and set RECALL_REALTIME_WS_BASE_URL "
                        "to the public https URL for this app.",
                        config.SLIDE_SERVER_PORT,
                    )
                continue

            if audio is None:
                return

            if not self._audio_event_seen:
                self._audio_event_seen = True
                logger.info("Receiving Recall mixed audio stream")

            if self._stt_backend == "openai_segmented" and self._participant_speaking:
                self._speech_buffer.extend(audio)

            if self._stt and self._stt_backend == "deepgram":
                self._stt.send_audio(audio)

    @staticmethod
    def _pcm_to_wav(audio: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio)
        return buf.getvalue()

    async def _transcribe_openai_segment(
        self,
        audio: bytes,
        speaker: Optional[str],
    ) -> None:
        if not audio or not self._stt_openai:
            return

        try:
            wav_bytes = self._pcm_to_wav(audio)
            result = await self._stt_openai.audio.transcriptions.create(
                file=("participant.wav", wav_bytes, "audio/wav"),
                model="gpt-4o-mini-transcribe",
                language="en",
            )
            text = (getattr(result, "text", "") or "").strip()
            if not text:
                return
            if speaker and speaker.lower() != "unknown":
                text = f"{speaker}: {text}"
            await self._stt_text_q.put(text)
        except Exception as exc:
            logger.warning("OpenAI fallback transcription failed: %s", exc)

    async def _poll_loop(self) -> None:
        """
        Final transcript listener:
          Deepgram final transcripts → Realtime user turns
        """
        while True:
            text = await self._stt_text_q.get()
            if text is None:
                return

            chunk = f"Participant: {text.strip()}"
            if not text.strip():
                continue

            if not self._should_emit_chunk(chunk):
                continue

            logger.info("Participant: %s", chunk)
            self._user_spoke.set()
            self._cancel_audio.set()
            if "?" in text or any(
                text.lower().startswith(w)
                for w in ("what", "how", "why", "when", "where",
                          "can", "could", "would", "tell", "explain")
            ):
                self._questions.append(text.strip())
            # The old response is cancelled and TTS is stopped — clear the
            # barge-in signal so the bot's reply to this input can play.
            self._cancel_audio.clear()
            await self._queue_turn(chunk)

    def _should_emit_chunk(self, chunk: str) -> bool:
        normalized = " ".join(chunk.lower().split())
        now = time.monotonic()
        if (
            normalized
            and normalized == self._last_emitted_chunk
            and now - self._last_emitted_at < 4.0
        ):
            logger.debug("Skipping duplicate transcript chunk: %s", chunk)
            return False
        self._last_emitted_chunk = normalized
        self._last_emitted_at = now
        return True

    # ------------------------------------------------------------------ #
    # Realtime event loop
    # ------------------------------------------------------------------ #

    async def _event_handler(self, rt, http: httpx.AsyncClient, bot_id: str) -> None:
        """
        Process OpenAI Realtime events (text-only output):
          • response.text.delta       — accumulate response text
          • response.done             — push text to _text_out_q for TTS
          • response.output_item.done — execute tool calls
          • error                     — log
        """
        response_text = ""
        tool_calls: list[tuple[str, str, dict]] = []

        async for event in rt:
            t = event.type

            if t == "response.created":
                response_text = ""
                tool_calls = []
                self._response_active = True

            elif t == "response.text.delta":
                # Drop deltas that arrive after a barge-in cancel so we don't
                # queue stale text for TTS.
                if not self._cancel_audio.is_set():
                    response_text += event.delta

            elif t == "response.output_item.done":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    args = json.loads(item.arguments or "{}")
                    tool_calls.append((item.call_id, item.name, args))

            elif t == "response.cancelled":
                # Barge-in cancel acknowledged — unblock the turn worker so
                # the participant's transcript can be sent immediately.
                logger.info("Realtime response cancelled (barge-in)")
                self._response_active = False
                response_text = ""
                tool_calls = []
                self._rt_idle.set()

            elif t == "response.done":
                self._response_active = False
                if self._cancel_audio.is_set():
                    # Barge-in happened; discard any partial text that arrived
                    # before the cancel was acknowledged.
                    logger.info("Discarding response text — barge-in in progress")
                    self._rt_idle.set()
                elif response_text.strip():
                    logger.info("Queuing TTS: %.120s", response_text)
                    self._text_out_q.put_nowait(response_text)
                elif not tool_calls:
                    # Nothing to say and no tool call — signal done immediately
                    self._audio_finished.set()

                if not self._cancel_audio.is_set() and tool_calls:
                    outputs: list[tuple[str, str]] = []
                    for call_id, name, args in tool_calls:
                        result = await self._run_tool(http, bot_id, name, args)
                        logger.info("Tool %s → %.80s", name, result)
                        outputs.append((call_id, result))
                    await self._send_tool_outputs(rt, outputs)
                elif not tool_calls:
                    self._rt_idle.set()

                response_text = ""
                tool_calls = []

            elif t == "error":
                logger.error("Realtime error: %s", event)
                self._response_active = False
                self._rt_idle.set()

    # ------------------------------------------------------------------ #
    # TTS — ElevenLabs (or OpenAI fallback) → Recall output_audio
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences so TTS can be interrupted between them."""
        import re
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    async def _synthesise(self, text: str, use_el: bool, el, oai) -> Optional[bytes]:
        """Generate MP3 for one sentence. Returns None on failure."""
        if use_el and el:
            try:
                chunks: list[bytes] = []
                async for chunk in el.text_to_speech.convert(
                    voice_id=config.ELEVENLABS_VOICE_ID,
                    text=text,
                    model_id="eleven_turbo_v2_5",
                    output_format="mp3_44100_128",
                ):
                    if chunk:
                        chunks.append(chunk)
                if chunks:
                    return b"".join(chunks)
            except Exception as exc:
                logger.warning("ElevenLabs failed: %s", exc)
        try:
            r = await oai.audio.speech.create(
                model="tts-1-hd", voice="shimmer",
                input=text, response_format="mp3",
            )
            return r.content
        except Exception as exc:
            logger.error("OpenAI TTS failed: %s", exc)
            return None

    async def _tts_worker(self, http: httpx.AsyncClient, bot_id: str) -> None:
        """
        Sentence-by-sentence TTS → Recall output_audio.
        Sends one sentence (~2-3 s) at a time so barge-in stops audio
        within 3 s instead of waiting for the whole paragraph to finish.
        """
        use_el = bool(config.ELEVENLABS_API_KEY and config.ELEVENLABS_VOICE_ID)
        el = None
        if use_el:
            try:
                from elevenlabs import AsyncElevenLabs
                el = AsyncElevenLabs(api_key=config.ELEVENLABS_API_KEY)
            except ImportError:
                logger.warning("elevenlabs package not installed — falling back to OpenAI TTS")
                use_el = False
        oai = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        logger.info(
            "TTS: %s",
            f"ElevenLabs {config.ELEVENLABS_VOICE_ID} (fallback OpenAI)" if use_el else "OpenAI TTS-HD",
        )

        while True:
            text = await self._text_out_q.get()
            if text is None:
                break
            text = text.strip()
            if not text:
                self._audio_finished.set()
                continue

            # If barge-in was signalled before we even start this utterance,
            # discard it entirely rather than clearing the flag and speaking.
            if self._cancel_audio.is_set():
                logger.info("Dropping queued utterance — barge-in already active")
                self._audio_finished.set()
                continue

            # Safe to clear now: no pending barge-in.
            self._cancel_audio.clear()

            sentences = self._split_sentences(text)
            for sentence in sentences:
                # Stop mid-paragraph if user interrupted
                if self._cancel_audio.is_set():
                    logger.info("Barge-in — stopping after current sentence")
                    break

                mp3 = await self._synthesise(sentence, use_el, el, oai)
                if not mp3:
                    continue

                try:
                    b64 = base64.b64encode(mp3).decode()
                    resp = await http.post(
                        f"{self._base}/bot/{bot_id}/output_audio/",
                        json={"kind": "mp3", "b64_data": b64},
                    )
                    if not resp.is_success:
                        logger.error("output_audio %s: %s", resp.status_code, resp.text[:200])
                        continue

                    # Wait for this sentence to finish playing, or until barge-in
                    duration_s = len(mp3) * 8 / 128_000
                    try:
                        await asyncio.wait_for(
                            self._cancel_audio.wait(), timeout=duration_s + 0.5
                        )
                        logger.info("Barge-in detected mid-sentence")
                        break
                    except asyncio.TimeoutError:
                        pass  # sentence finished naturally
                except Exception as exc:
                    logger.error("output_audio send failed: %s", exc)

            self._audio_finished.set()

    # ------------------------------------------------------------------ #
    # Auto-advance with proactive check-in + end-of-meeting summary
    # ------------------------------------------------------------------ #

    async def _auto_advance(self, rt, http: httpx.AsyncClient, bot_id: str) -> None:
        """
        Phase 1 — 8 s grace after narration:  wait for participant speech.
        Phase 2 — check-in question:           bot asks one contextual question.
        Phase 3 — 10 s more silence:           advance to next slide.
        End of deck: deliver closing summary, then stay live for Q&A.
        """
        while True:
            await self._audio_finished.wait()
            self._audio_finished.clear()

            if self.manager.at_end:
                if not self._summary_given:
                    self._summary_given = True
                    await self._give_summary(rt)
                continue

            # Phase 1
            if not await self._wait_for_silence(8.0):
                continue

            # Phase 2 — check-in
            logger.info("8 s silence — asking check-in")
            try:
                await self._queue_turn(
                    "No one has asked a question. Ask the participants one brief, "
                    "natural check-in question about this slide's content. "
                    "Use a participant's name if you know them. One sentence only."
                )
            except Exception as exc:
                logger.warning("Check-in failed: %s", exc)
                continue

            await self._audio_finished.wait()
            self._audio_finished.clear()

            # Phase 3
            if not await self._wait_for_silence(10.0):
                continue

            if self.manager.at_end:
                continue

            # Advance slide
            logger.info("Advancing slide")
            try:
                async with httpx.AsyncClient(timeout=10) as nav:
                    resp = await nav.post(
                        f"{SLIDE_BASE}/api/navigate", json={"action": "next"}
                    )
                    resp.raise_for_status()
                    d = resp.json()
                asyncio.create_task(self._push_slide(http, bot_id, d["index"]))
                await self._queue_turn(
                    f"Present slide {d['index']+1} of {d['total']} now.\n\n"
                    f"SLIDE CONTENT — narrate exactly this:\n{d['summary']}"
                )
            except Exception as exc:
                logger.warning("Auto-advance failed: %s", exc)

    async def _wait_for_silence(self, duration: float) -> bool:
        """
        Poll every 0.5 s. Returns False if participant speaks.
        Resets the timer if bot audio finishes (it answered a question).
        Returns True on genuine silence.
        """
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            if self._user_spoke.is_set():
                self._user_spoke.clear()
                return False
            if self._audio_finished.is_set():
                self._audio_finished.clear()
                deadline = time.monotonic() + duration
        return True

    async def _give_summary(self, rt) -> None:
        q_text = (
            "\n".join(f"- {q}" for q in self._questions)
            if self._questions
            else "No specific questions were recorded."
        )
        logger.info("Delivering end-of-meeting summary")
        try:
            await self._queue_turn(
                "The presentation is complete. Give a warm closing that:\n"
                "1. Thanks participants by name (if known)\n"
                "2. Recaps 2–3 key topics covered\n"
                "3. Addresses questions raised during the meeting\n"
                "4. States a clear next step\n"
                "Keep it under 90 words.\n\n"
                f"Questions raised:\n{q_text}"
            )
        except Exception as exc:
            logger.warning("Summary failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Screen share
    # ------------------------------------------------------------------ #

    async def _push_slide(self, http: httpx.AsyncClient, bot_id: str, slide_index: int) -> None:
        img_path = os.path.join(_SLIDE_IMAGE_DIR, f"slide_{slide_index:03d}.png")
        for _ in range(15):
            if os.path.exists(img_path):
                break
            await asyncio.sleep(1)
        if not os.path.exists(img_path):
            logger.warning("Slide image not found: %s", img_path)
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
            logger.info("Screen share → slide %d", slide_index)
        except Exception as exc:
            logger.warning("Screen share failed: %s", exc)

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
                    return data["summary"]

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
                    return "\n".join(parts)

            if name == "leave_meeting":
                resp = await http.post(f"{self._base}/bot/{bot_id}/leave_call/")
                resp.raise_for_status()
                return "Left the meeting."

            return f"Unknown tool: {name}"
        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc)
            return f"Error: {exc}"
