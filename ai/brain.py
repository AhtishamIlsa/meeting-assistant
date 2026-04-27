"""
AI Brain — uses Claude to understand voice commands and drive the presentation.
All transcripts from the client/audience are processed here to produce structured actions.
"""
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import anthropic

from config import config

logger = logging.getLogger(__name__)

COMMAND_SYSTEM_PROMPT = """You are an AI presentation assistant controlling a live sales/demo meeting.
You listen to the client and decide what action to take in real time.

You will receive:
- The current slide number and total slide count
- The current slide's text content
- What the client just said (transcript)

Classify the intent and respond with ONLY valid JSON — no markdown, no explanation:

{
  "action": "<action_type>",
  "slide_number": <integer or null>,
  "response_text": "<spoken response text or null>",
  "confidence": <float 0.0–1.0>
}

Available actions:
- "next_slide"        — client wants to advance (e.g. "next", "move on", "continue", "go ahead")
- "prev_slide"        — client wants to go back (e.g. "go back", "previous", "back one")
- "goto_slide"        — jump to a specific slide; set slide_number to the 1-based target
- "repeat_slide"      — re-explain the current slide (e.g. "explain again", "say that again", "repeat")
- "answer_question"   — client asked a question; include a concise, helpful answer in response_text
- "pause"             — client wants to pause/hold (e.g. "hold on", "wait", "stop for a moment")
- "resume"            — client wants to continue after a pause
- "end_presentation"  — client wants to end (e.g. "that's all", "we're done", "end the presentation")
- "ignore"            — background noise, filler words, or clearly unrelated speech

Confidence rules:
- 0.9–1.0: Explicit, unambiguous command
- 0.6–0.9: Probable intent, slight ambiguity
- 0.3–0.6: Uncertain; do not act unless > 0.5 threshold
- 0.0–0.3: Noise/filler — always "ignore"

For "answer_question" the response_text must be a complete spoken answer (1–3 sentences, conversational).
For navigation actions, response_text can be a short acknowledgement or null."""

NARRATION_SYSTEM_PROMPT = """You are a professional sales presenter and demo specialist.
Generate natural, engaging spoken narration for presentation slides.

Rules:
- Sound like a confident human presenter, not a robot
- Keep it concise: 2–4 sentences (about 30–60 words)
- Use conversational language; avoid bullet-point reading
- Highlight the key insight or benefit of the slide
- Do NOT mention slide numbers or say "as you can see on this slide"
- Return ONLY the narration text — no quotes, no labels"""


@dataclass
class AIAction:
    action: str
    slide_number: Optional[int] = None
    response_text: Optional[str] = None
    confidence: float = 1.0


class ActionType:
    NEXT_SLIDE = "next_slide"
    PREV_SLIDE = "prev_slide"
    GOTO_SLIDE = "goto_slide"
    REPEAT_SLIDE = "repeat_slide"
    ANSWER_QUESTION = "answer_question"
    PAUSE = "pause"
    RESUME = "resume"
    END_PRESENTATION = "end_presentation"
    IGNORE = "ignore"


class AIBrain:
    def __init__(self):
        self._client: Optional[anthropic.Anthropic] = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not config.ANTHROPIC_API_KEY:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        return self._client

    def process_transcript(
        self,
        transcript: str,
        current_slide: int,
        total_slides: int,
        slide_content: str,
    ) -> AIAction:
        """Classify a voice transcript into a structured presenter action."""
        user_message = (
            f"Current slide: {current_slide} of {total_slides}\n"
            f"Slide content:\n{slide_content[:600]}\n\n"
            f"Client said: \"{transcript}\"\n\n"
            "What action should the AI presenter take?"
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=[
                    {
                        "type": "text",
                        "text": COMMAND_SYSTEM_PROMPT,
                        # Cache the large static system prompt across rapid-fire transcript calls
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )

            raw = response.content[0].text.strip()
            data = json.loads(raw)
            return AIAction(
                action=data.get("action", ActionType.IGNORE),
                slide_number=data.get("slide_number"),
                response_text=data.get("response_text"),
                confidence=float(data.get("confidence", 1.0)),
            )

        except json.JSONDecodeError as exc:
            logger.warning("AI Brain returned non-JSON: %s", exc)
            return AIAction(action=ActionType.IGNORE, confidence=0.0)
        except Exception as exc:
            logger.error("AI Brain error: %s", exc)
            return AIAction(action=ActionType.IGNORE, confidence=0.0)

    def generate_narration(
        self,
        slide_number: int,
        total_slides: int,
        slide_content: str,
        speaker_notes: str = "",
    ) -> str:
        """Generate natural spoken narration for a slide."""
        notes_section = f"\nSpeaker notes: {speaker_notes}" if speaker_notes else ""
        user_message = (
            f"Slide {slide_number} of {total_slides}.\n"
            f"Content:\n{slide_content}{notes_section}\n\n"
            "Generate the spoken narration."
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                system=[
                    {
                        "type": "text",
                        "text": NARRATION_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text.strip()

        except Exception as exc:
            logger.error("Narration generation error: %s", exc)
            # Graceful fallback — read the raw slide title
            first_line = slide_content.split("\n")[0] if slide_content else f"Slide {slide_number}"
            return f"Let's look at {first_line}."
