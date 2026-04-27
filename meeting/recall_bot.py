"""
Meeting Bot — Recall.ai integration.

Recall.ai provides a cloud bot that:
  - Joins Zoom / Google Meet / Teams meetings automatically
  - Captures meeting audio (used for STT transcription)
  - Can receive screen-share commands
  - Sends real-time transcripts via webhook

API docs: https://docs.recall.ai/
"""
import logging
from typing import Optional

import httpx

from config import config

logger = logging.getLogger(__name__)


class RecallBot:
    """
    Wraps the Recall.ai v2 Bot API.
    If RECALL_AI_API_KEY is empty the bot operates in mock/demo mode —
    all calls succeed silently so the rest of the system still works.
    """

    BASE_URL_TEMPLATE = "https://{region}.recall.ai/api/v2"

    def __init__(self):
        self.bot_id: Optional[str] = None
        self._base = self.BASE_URL_TEMPLATE.format(region=config.RECALL_AI_REGION)
        self._demo_mode = not bool(config.RECALL_AI_API_KEY)
        if self._demo_mode:
            logger.warning("RECALL_AI_API_KEY not set — running in demo mode (no real bot)")

    # ------------------------------------------------------------------ #
    # Headers
    # ------------------------------------------------------------------ #

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Token {config.RECALL_AI_API_KEY}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------ #
    # Bot lifecycle
    # ------------------------------------------------------------------ #

    def join_meeting(
        self,
        meeting_url: str,
        bot_name: str = "AI Presenter",
        webhook_url: Optional[str] = None,
    ) -> dict:
        """
        Spin up a Recall.ai bot and join the meeting.
        Returns the bot object dict (includes bot_id).
        """
        if self._demo_mode:
            self.bot_id = "demo-bot-id"
            return {"id": "demo-bot-id", "status": "joining", "demo": True}

        payload: dict = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "transcription_options": {
                "provider": "deepgram",
                "deepgram": {
                    "model": "nova-2",
                    "language": "en",
                    "smart_format": True,
                },
            },
        }

        if webhook_url:
            payload["webhook_url"] = webhook_url

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self._base}/bot",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            self.bot_id = data["id"]
            logger.info("Recall bot joined: %s", self.bot_id)
            return data

    def get_status(self) -> dict:
        """Return current bot status from Recall.ai."""
        if self._demo_mode or not self.bot_id:
            return {"id": self.bot_id, "status": "in_call" if self.bot_id else "not_joined"}

        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{self._base}/bot/{self.bot_id}",
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

    def leave_meeting(self) -> bool:
        """Remove the bot from the meeting."""
        if self._demo_mode:
            self.bot_id = None
            return True

        if not self.bot_id:
            return False

        try:
            with httpx.Client(timeout=15) as client:
                resp = client.delete(
                    f"{self._base}/bot/{self.bot_id}",
                    headers=self._headers,
                )
            self.bot_id = None
            return resp.status_code in (200, 204)
        except Exception as exc:
            logger.warning("Failed to remove Recall bot: %s", exc)
            return False

    def send_chat_message(self, message: str):
        """Post a text message to the meeting chat."""
        if self._demo_mode or not self.bot_id:
            logger.debug("Demo: chat message → %s", message)
            return

        with httpx.Client(timeout=10) as client:
            client.post(
                f"{self._base}/bot/{self.bot_id}/send_chat_message",
                headers=self._headers,
                json={"message": message},
            )

    # ------------------------------------------------------------------ #
    # Webhook helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_transcript_webhook(payload: dict) -> Optional[str]:
        """
        Extract the final transcript text from a Recall.ai webhook payload.
        Returns None if not a transcript event or no text found.
        """
        event = payload.get("event", "")
        if event not in ("bot.transcription", "transcript.data"):
            return None

        data = payload.get("data", {})
        words = data.get("words", [])
        if words:
            return " ".join(w.get("text", "") for w in words).strip() or None

        # Alternative schema
        transcript = data.get("transcript", "")
        return transcript.strip() or None
