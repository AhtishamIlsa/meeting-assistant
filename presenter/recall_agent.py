"""
AI Meeting Presenter — Recall.ai Agent
────────────────────────────────────────
Pipeline:

  Recall bot joins meeting
    │
    ├─► transcript polling every 1 s  (_poll_loop)
    │       Deepgram → word-level timestamps → deduplicated text
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
  _poll_loop   — poll Recall transcript → inject turns into Realtime
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
from datetime import datetime
from typing import Optional

import httpx
from openai import AsyncOpenAI
from PIL import Image

from config import config
from presenter.prompt import get_presenter_prompt
from slide_server.app import recall_transcript_q
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
        self._text_out_q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._turn_q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._last_word_end: float = 0.0
        self._last_emitted_chunk: str = ""
        self._last_emitted_at: float = 0.0
        self._realtime_event_seen: bool = False
        self._realtime_warning_emitted: bool = False
        # Coordination events
        self._user_spoke: asyncio.Event = asyncio.Event()
        self._audio_finished: asyncio.Event = asyncio.Event()
        self._cancel_audio: asyncio.Event = asyncio.Event()
        self._rt_idle: asyncio.Event = asyncio.Event()
        self._rt_write_lock: Optional[asyncio.Lock] = None
        self._interrupted: bool = False
        # Meeting memory
        self._questions: list[str] = []
        self._summary_given: bool = False
        self._recording_id: Optional[str] = None

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
        self._last_word_end = 0.0
        self._last_emitted_chunk = ""
        self._last_emitted_at = 0.0
        self._realtime_event_seen = False
        self._realtime_warning_emitted = False
        self._user_spoke.clear()
        self._audio_finished.clear()
        self._cancel_audio.clear()
        self._rt_idle = asyncio.Event()
        self._rt_idle.set()
        self._rt_write_lock = asyncio.Lock()
        self._interrupted = False
        self._questions = []
        self._summary_given = False
        self._recording_id = None

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
                        self._poll_loop(rt, http, bot_id), name="poll_loop"
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

    # ------------------------------------------------------------------ #
    # Bot lifecycle
    # ------------------------------------------------------------------ #

    async def _create_bot(self, http: httpx.AsyncClient, meeting_url: str) -> str:
        realtime_endpoints = []
        if config.RECALL_WEBHOOK_URL:
            webhook = f"{config.RECALL_WEBHOOK_URL.rstrip('/')}/api/recall/events"
            realtime_endpoints = [{
                "type": "webhook",
                "url": webhook,
                "events": [
                    "transcript.data",
                    "participant_events.speech_on",
                    "participant_events.speech_off",
                ],
            }]
            logger.info("Real-time transcript webhook: %s", webhook)
        else:
            logger.warning(
                "RECALL_WEBHOOK_URL not set — live participant listening will not work. "
                "Run: cloudflared tunnel --url http://localhost:%s  then set RECALL_WEBHOOK_URL",
                config.SLIDE_SERVER_PORT,
            )

        provider = self._build_transcript_provider()
        logger.info("Using Recall transcript provider: %s", next(iter(provider.keys())))

        payload: dict = {
            "meeting_url": meeting_url,
            "bot_name": config.PRESENTER_NAME,
            "recording_config": {
                "transcript": {
                    "provider": provider,
                    "diarization": {"use_separate_streams_when_available": True},
                },
                "realtime_endpoints": realtime_endpoints,
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

    def _build_transcript_provider(self) -> dict:
        provider = (config.RECALL_TRANSCRIPTION_PROVIDER or "").strip().lower()
        language = (config.RECALL_TRANSCRIPTION_LANGUAGE or "en").strip()
        mode = (config.RECALL_TRANSCRIPTION_MODE or "prioritize_low_latency").strip()

        if provider in ("", "recallai", "recallai_streaming"):
            return {
                "recallai_streaming": {
                    "mode": mode,
                    "language_code": language,
                }
            }

        if provider == "deepgram_streaming":
            cfg: dict[str, object] = {"language": language}
            if language == "multi":
                cfg["model"] = "nova-3"
            return {"deepgram_streaming": cfg}

        logger.warning(
            "Unknown RECALL_TRANSCRIPTION_PROVIDER=%s — falling back to recallai_streaming",
            provider,
        )
        return {
            "recallai_streaming": {
                "mode": "prioritize_low_latency",
                "language_code": "en",
            }
        }

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
    # Transcript polling
    # ------------------------------------------------------------------ #

    async def _poll_loop(self, rt, http: httpx.AsyncClient, bot_id: str) -> None:
        """
        Live transcript listener:
          - Webhook queue (< 1 s): drains Recall realtime_endpoints push events

        Recall's v1.11 real-time transcription flow is webhook/websocket driven.
        The transcript API is not a live "transcript-so-far" polling API for
        active calls, so live participant reactions depend on realtime events.
        """
        start_wait = time.monotonic()

        while True:
            chunk: Optional[str] = None

            if config.RECALL_WEBHOOK_URL:
                segments: list[dict] = []
                try:
                    while True:
                        event = recall_transcript_q.get_nowait()
                        self._realtime_event_seen = True
                        segments.extend(self._extract_transcript_segments(event))
                except asyncio.QueueEmpty:
                    pass
                if segments:
                    chunk = self._parse_words(segments)

            await asyncio.sleep(0.5)

            if (
                config.RECALL_WEBHOOK_URL
                and not self._realtime_event_seen
                and not self._realtime_warning_emitted
                and time.monotonic() - start_wait > 20
            ):
                self._realtime_warning_emitted = True
                logger.warning(
                    "No Recall realtime events received after 20s. "
                    "Participant listening depends on realtime endpoint delivery. "
                    "Check that %s is still publicly reachable and that Recall transcription is enabled.",
                    config.RECALL_WEBHOOK_URL,
                )

            if chunk:
                if not self._should_emit_chunk(chunk):
                    continue
                logger.info("Participant: %s", chunk)
                self._user_spoke.set()
                self._cancel_audio.set()
                for line in chunk.splitlines():
                    text = line.split(":", 1)[-1].strip()
                    if text and ("?" in text or any(
                        text.lower().startswith(w)
                        for w in ("what", "how", "why", "when", "where",
                                  "can", "could", "would", "tell", "explain")
                    )):
                        self._questions.append(text)
                await self._queue_turn(chunk)

    def _parse_words(self, data) -> Optional[str]:
        """
        Parse Recall transcript segments.
        Format: list of {participant: {name}, words: [{text, end_timestamp: {relative, absolute}}]}
        """
        new_words: list[tuple[float, str, str]] = []
        bot_name = config.PRESENTER_NAME.lower()
        synthetic_end = self._last_word_end

        segments = self._extract_transcript_segments(data)
        for seg in segments:
            speaker = self._speaker_name(seg)
            if (
                not speaker
                or speaker.lower() == bot_name
                or seg.get("speaker_role") == "agent"
            ):
                continue
            for w in seg.get("words") or []:
                text = self._word_text(w)
                if not text:
                    continue
                end = self._word_end(w)
                if end is None:
                    synthetic_end += 0.001
                    end = synthetic_end
                if end > self._last_word_end:
                    new_words.append((end, speaker, text))

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

    def _extract_transcript_segments(self, payload) -> list[dict]:
        segments: list[dict] = []
        stack = [payload]
        seen: set[int] = set()

        while stack:
            current = stack.pop()
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)

            if isinstance(current, dict):
                words = current.get("words")
                if isinstance(words, list) and words:
                    segments.append(current)
                for value in current.values():
                    if isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(current, list):
                stack.extend(current)

        return segments

    @staticmethod
    def _speaker_name(segment: dict) -> str:
        participant = segment.get("participant")
        if isinstance(participant, dict) and participant.get("name"):
            return str(participant["name"]).strip()
        for key in ("speaker", "speaker_name", "participant_name", "name"):
            value = segment.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "Participant"

    @staticmethod
    def _word_text(word: dict) -> str:
        for key in ("text", "word", "punctuated_word"):
            value = word.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _as_float(value) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _word_end(self, word: dict) -> Optional[float]:
        end_timestamp = word.get("end_timestamp")
        if isinstance(end_timestamp, dict):
            for key in ("relative", "seconds", "time"):
                value = self._as_float(end_timestamp.get(key))
                if value is not None:
                    return value
            absolute = end_timestamp.get("absolute")
            if isinstance(absolute, str):
                try:
                    return datetime.fromisoformat(
                        absolute.replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    pass
        else:
            value = self._as_float(end_timestamp)
            if value is not None:
                return value

        for key in ("end", "end_time"):
            value = self._as_float(word.get(key))
            if value is not None:
                return value

        end_ms = self._as_float(word.get("end_ms"))
        if end_ms is not None:
            return end_ms / 1000.0

        return None

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

            elif t == "response.text.delta":
                response_text += event.delta

            elif t == "response.output_item.done":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    args = json.loads(item.arguments or "{}")
                    tool_calls.append((item.call_id, item.name, args))

            elif t == "response.done":
                if response_text.strip():
                    logger.info("Queuing TTS: %.120s", response_text)
                    self._text_out_q.put_nowait(response_text)
                elif not tool_calls:
                    # Nothing to say and no tool call — signal done immediately
                    self._audio_finished.set()

                if tool_calls:
                    outputs: list[tuple[str, str]] = []
                    for call_id, name, args in tool_calls:
                        result = await self._run_tool(http, bot_id, name, args)
                        logger.info("Tool %s → %.80s", name, result)
                        outputs.append((call_id, result))
                    await self._send_tool_outputs(rt, outputs)
                else:
                    self._rt_idle.set()

                response_text = ""
                tool_calls = []

            elif t == "error":
                logger.error("Realtime error: %s", event)
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

            # Clear cancel flag at start of new utterance
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
