"""
Orchestrator — PresentationSession ties every layer together.

State machine:
  IDLE → JOINING → PRESENTING ↔ PAUSED
                  PRESENTING → SPEAKING → LISTENING → (loop)
                  PRESENTING / LISTENING → ENDED
"""
import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ai.brain import AIBrain, ActionType
from meeting.recall_bot import RecallBot
from slides.manager import SlideData, SlideManager
from voice.tts_engine import TTSEngine
from config import config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# State & Events
# ------------------------------------------------------------------ #

class SessionState(str, Enum):
    IDLE = "idle"
    JOINING = "joining"
    PRESENTING = "presenting"
    SPEAKING = "speaking"
    LISTENING = "listening"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"


@dataclass
class SessionEvent:
    type: str
    data: Dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data, "timestamp": self.timestamp}


EventCallback = Callable[[SessionEvent], None]


# ------------------------------------------------------------------ #
# Session
# ------------------------------------------------------------------ #

class PresentationSession:
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, session_id: Optional[str] = None):
        self.session_id: str = session_id or str(uuid.uuid4())
        self.state: SessionState = SessionState.IDLE
        self.created_at: str = datetime.now(timezone.utc).isoformat()

        # Components
        self.slides = SlideManager()
        self.tts = TTSEngine()
        self.brain = AIBrain()
        self.bot = RecallBot()

        # Runtime state
        self.meeting_url: Optional[str] = None
        self.slides_file: Optional[str] = None
        self.events_log: List[SessionEvent] = []
        self._listeners: List[EventCallback] = []
        self._slide_image_dir = os.path.join("slide_images", self.session_id)

    # ------------------------------------------------------------------ #
    # Event system
    # ------------------------------------------------------------------ #

    def add_listener(self, callback: EventCallback):
        self._listeners.append(callback)

    def _emit(self, event_type: str, data: dict):
        event = SessionEvent(type=event_type, data=data)
        self.events_log.append(event)
        for cb in self._listeners:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(event))
                else:
                    cb(event)
            except Exception as exc:
                logger.error("Listener error: %s", exc)

    def _set_state(self, state: SessionState, message: str = ""):
        self.state = state
        self._emit("state_change", {"state": state.value, "message": message})

    # ------------------------------------------------------------------ #
    # Slide loading
    # ------------------------------------------------------------------ #

    def load_slides(self, file_path: str) -> int:
        self.slides_file = file_path
        count = self.slides.load(file_path)
        self._emit("slides_loaded", {"count": count, "file": os.path.basename(file_path)})
        return count

    # ------------------------------------------------------------------ #
    # Presentation lifecycle
    # ------------------------------------------------------------------ #

    async def start(self, meeting_url: Optional[str] = None):
        self.meeting_url = meeting_url

        if meeting_url:
            self._set_state(SessionState.JOINING, "Joining meeting...")
            try:
                bot_data = self.bot.join_meeting(
                    meeting_url=meeting_url,
                    bot_name="AI Presenter",
                )
                self._emit("bot_joined", {"bot_id": bot_data.get("id"), "demo": bot_data.get("demo", False)})
            except Exception as exc:
                logger.warning("Bot join failed: %s — continuing without meeting bot", exc)
                self._emit("warning", {"message": f"Meeting bot unavailable: {exc}"})

        self._set_state(SessionState.PRESENTING, "Presentation started")
        await self._present_current_slide()

    async def end(self):
        self._set_state(SessionState.ENDED, "Session ended")
        try:
            self.bot.leave_meeting()
        except Exception:
            pass
        self._emit("session_ended", {"session_id": self.session_id})

    # ------------------------------------------------------------------ #
    # Slide presentation
    # ------------------------------------------------------------------ #

    async def _present_current_slide(self):
        slide = self.slides.get_current()
        if not slide:
            self._emit("error", {"message": "No slides loaded"})
            return

        # Export preview image (best-effort)
        image_path = self.slides.export_slide_image(slide.index, self._slide_image_dir)
        slide.image_path = image_path

        self._emit("slide_change", slide.to_dict())

        # Generate AI narration and synthesize voice
        self._set_state(SessionState.SPEAKING)
        narration = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.brain.generate_narration(
                slide_number=slide.number,
                total_slides=self.slides.total_slides,
                slide_content=slide.content,
                speaker_notes=slide.speaker_notes,
            ),
        )
        self._emit("narration", {"text": narration, "slide": slide.number})

        audio_path = os.path.join(
            config.AUDIO_CACHE_DIR,
            f"session_{self.session_id}_slide_{slide.index}.mp3",
        )
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.tts.synthesize_to_file(narration, audio_path),
            )
            self._emit("audio_ready", {
                "path": audio_path,
                "url": f"/audio/{self.session_id}/slide_{slide.index}.mp3",
                "text": narration,
            })
        except Exception as exc:
            logger.error("TTS failed: %s", exc)
            self._emit("warning", {"message": f"Voice synthesis unavailable: {exc}"})

        self._set_state(SessionState.LISTENING, "Waiting for client input")

    # ------------------------------------------------------------------ #
    # Voice input processing
    # ------------------------------------------------------------------ #

    async def process_transcript(self, transcript: str):
        """Called when a new STT transcript arrives from any source."""
        if self.state not in (SessionState.LISTENING, SessionState.PRESENTING):
            return

        self._emit("transcript", {"text": transcript})

        slide = self.slides.get_current()
        if not slide:
            return

        action = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.brain.process_transcript(
                transcript=transcript,
                current_slide=slide.number,
                total_slides=self.slides.total_slides,
                slide_content=slide.content,
            ),
        )

        self._emit("ai_action", {
            "transcript": transcript,
            "action": action.action,
            "confidence": action.confidence,
            "response": action.response_text,
            "slide_number": action.slide_number,
        })

        if action.confidence < self.CONFIDENCE_THRESHOLD:
            logger.debug("Low confidence (%.2f) — ignoring: %s", action.confidence, transcript)
            return

        await self._dispatch(action)

    # ------------------------------------------------------------------ #
    # Action dispatcher
    # ------------------------------------------------------------------ #

    async def _dispatch(self, action):
        if action.action == ActionType.NEXT_SLIDE:
            if self.slides.is_at_end:
                await self._speak("That was the final slide. Thank you for joining today!")
                await self.end()
            else:
                self.slides.next()
                await self._present_current_slide()

        elif action.action == ActionType.PREV_SLIDE:
            self.slides.previous()
            await self._present_current_slide()

        elif action.action == ActionType.GOTO_SLIDE and action.slide_number:
            self.slides.goto(action.slide_number - 1)
            await self._present_current_slide()

        elif action.action == ActionType.REPEAT_SLIDE:
            await self._present_current_slide()

        elif action.action == ActionType.ANSWER_QUESTION and action.response_text:
            await self._speak(action.response_text)

        elif action.action == ActionType.PAUSE:
            self._set_state(SessionState.PAUSED, "Paused by client request")

        elif action.action == ActionType.RESUME:
            self._set_state(SessionState.LISTENING, "Resumed")

        elif action.action == ActionType.END_PRESENTATION:
            await self._speak("Thank you for your time. The presentation is now complete.")
            await self.end()

    async def _speak(self, text: str):
        self._set_state(SessionState.SPEAKING)
        self._emit("narration", {"text": text})
        audio_path = os.path.join(
            config.AUDIO_CACHE_DIR,
            f"session_{self.session_id}_resp_{uuid.uuid4().hex[:8]}.mp3",
        )
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.tts.synthesize_to_file(text, audio_path),
            )
            self._emit("audio_ready", {"path": audio_path, "text": text})
        except Exception as exc:
            logger.error("TTS speak error: %s", exc)

        self._set_state(SessionState.LISTENING)

    # ------------------------------------------------------------------ #
    # Manual controls (from dashboard)
    # ------------------------------------------------------------------ #

    async def manual_control(self, action: str, slide_number: Optional[int] = None):
        """Handle manual commands from the dashboard."""
        if action == "next":
            if not self.slides.is_at_end:
                self.slides.next()
                await self._present_current_slide()
        elif action == "previous":
            if not self.slides.is_at_start:
                self.slides.previous()
                await self._present_current_slide()
        elif action == "goto" and slide_number is not None:
            self.slides.goto(slide_number - 1)
            await self._present_current_slide()
        elif action == "pause":
            self._set_state(SessionState.PAUSED)
        elif action == "resume":
            self._set_state(SessionState.LISTENING)
        elif action == "repeat":
            await self._present_current_slide()
        elif action == "end":
            await self.end()

    # ------------------------------------------------------------------ #
    # Status snapshot
    # ------------------------------------------------------------------ #

    def status(self) -> dict:
        slide = self.slides.get_current()
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "meeting_url": self.meeting_url,
            "slides_loaded": self.slides.total_slides > 0,
            "current_slide": slide.to_dict() if slide else None,
            "total_slides": self.slides.total_slides,
            "events_count": len(self.events_log),
        }
