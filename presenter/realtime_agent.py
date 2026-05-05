"""
AI Meeting Presenter — OpenAI Realtime Agent
─────────────────────────────────────────────
GPT-4o Realtime for low-latency conversation.

Audio flow:
  joinly STT (Deepgram) → transcript text
    → GPT-4o Realtime (text mode, tool calls for slide nav)
    → response text → joinly speak_text (ElevenLabs) — fire-and-forget

Key design: speak_text is non-blocking so the main loop stays responsive
while TTS is playing. A single-worker queue serialises RT turns so we
never have two concurrent writes on the WebSocket.
"""
import asyncio
import json
import logging
from typing import Any

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp import ResourceUpdatedNotification, ServerNotification
from openai import AsyncOpenAI
from pydantic import AnyUrl

from config import config
from presenter.prompt import get_presenter_prompt
from slide_server.manager import SlideManager

logger = logging.getLogger(__name__)

TRANSCRIPT_URI = AnyUrl("transcript://live")
SLIDE_BASE = f"http://localhost:{config.SLIDE_SERVER_PORT}"

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
                "action": {
                    "type": "string",
                    "enum": ["next", "previous", "goto", "repeat"],
                },
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
        "description": "Leave the meeting. Call only when the client says goodbye / done / end the presentation.",
        "parameters": {"type": "object", "properties": {}},
    },
]


class RealtimePresenterAgent:
    MODEL = "gpt-4o-realtime-preview-2024-12-17"

    def __init__(self, slide_manager: SlideManager):
        self.manager = slide_manager
        self._last_segment_time: float = -1.0

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    async def run(self, meeting_url: str) -> None:
        transcript_event = asyncio.Event()
        # Queue serialises RT turns; main loop enqueues, worker dequeues
        turn_queue: asyncio.Queue[str] = asyncio.Queue()

        async def _on_notification(msg: Any) -> None:
            if (
                isinstance(msg, ServerNotification)
                and isinstance(msg.root, ResourceUpdatedNotification)
                and str(msg.root.params.uri) == str(TRANSCRIPT_URI)
            ):
                transcript_event.set()

        transport = StreamableHttpTransport(
            url=config.JOINLY_MCP_URL,
            headers={"joinly-settings": self._joinly_settings()},
        )

        async with Client(transport, message_handler=_on_notification) as joinly:
            logger.info("Connected to joinly MCP server at %s", config.JOINLY_MCP_URL)
            await joinly.session.subscribe_resource(TRANSCRIPT_URI)

            # Leave any lingering session before joining a new one
            try:
                await joinly.call_tool("leave_meeting", {})
                logger.info("Left previous meeting session")
                await asyncio.sleep(2)  # let the browser fully reset
            except Exception:
                pass  # no active meeting — that's fine

            logger.info("Joining meeting: %s", meeting_url)
            for attempt in range(1, 4):
                try:
                    await joinly.call_tool("join_meeting", {"meeting_url": meeting_url})
                    break
                except Exception as exc:
                    if attempt == 3:
                        raise
                    logger.warning("join_meeting attempt %d failed: %s — retrying in 5s", attempt, exc)
                    await asyncio.sleep(5)

            openai = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            async with openai.beta.realtime.connect(model=self.MODEL) as rt:
                logger.info("Connected to OpenAI Realtime API (model=%s)", self.MODEL)

                await rt.session.update(session={
                    "modalities": ["text"],
                    "instructions": get_presenter_prompt(config.PRESENTER_NAME),
                    "tools": _TOOLS,
                    "tool_choice": "auto",
                    "temperature": 0.8,
                })

                # Queue first-slide narration
                first = self.manager.current()
                if first:
                    await turn_queue.put(
                        f"[SYSTEM] The presentation has started. "
                        f"Narrate the first slide now.\n{first.summary()}"
                    )

                # Worker: drains the queue sequentially, doesn't block main loop
                async def _worker():
                    while True:
                        text = await turn_queue.get()
                        try:
                            await self._turn(rt, joinly, text)
                        except Exception:
                            logger.exception("Turn failed")
                        finally:
                            turn_queue.task_done()

                worker_task = asyncio.create_task(_worker())

                # Main loop — enqueues client speech immediately
                logger.info("Entering main loop — listening for client input")
                try:
                    while True:
                        await transcript_event.wait()
                        transcript_event.clear()

                        try:
                            resources = await joinly.read_resource(TRANSCRIPT_URI)
                            transcript = json.loads(resources[0].text)
                        except Exception as exc:
                            logger.warning("Failed to read transcript: %s", exc)
                            continue

                        new_segs = [
                            s for s in transcript.get("segments", [])
                            if s.get("start", 0) > self._last_segment_time
                            and s.get("speaker_role") != "agent"
                        ]
                        if not new_segs:
                            continue

                        self._last_segment_time = new_segs[-1].get("start", 0)
                        user_text = "\n".join(
                            f"{s.get('speaker') or 'Client'}: {s['text']}"
                            for s in new_segs
                        )
                        logger.info("Client said: %s", user_text)
                        await turn_queue.put(user_text)
                finally:
                    worker_task.cancel()

    # ------------------------------------------------------------------ #
    # Conversation turn
    # ------------------------------------------------------------------ #

    async def _turn(self, rt: Any, joinly: Any, text: str) -> None:
        await rt.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        })
        await rt.response.create()
        await self._drain(rt, joinly)

    async def _drain(self, rt: Any, joinly: Any) -> None:
        """Consume RT events until response.done. speak_text is fire-and-forget."""
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
                logger.error("Realtime API error: %s", event)
                break

        # Fire speak_text without blocking — TTS plays in background
        if response_text.strip():
            logger.info("Speaking: %.120s", response_text)
            asyncio.create_task(joinly.call_tool("speak_text", {"text": response_text}))

        # Execute tool calls then get the follow-up response
        if tool_calls:
            for call_id, tc in tool_calls.items():
                name = tc["name"]
                if name == "leave_meeting":
                    logger.info("Leaving meeting")
                    try:
                        await joinly.call_tool("leave_meeting", {})
                        result = "Left the meeting."
                    except Exception as exc:
                        result = f"Could not leave: {exc}"
                else:
                    result = self._run_tool(name, json.loads(tc["arguments"]))
                logger.info("Tool %s → %.80s", name, result)
                await rt.conversation.item.create(item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                })
            await rt.response.create()
            await self._drain(rt, joinly)

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #

    def _run_tool(self, name: str, args: dict) -> str:
        try:
            if name == "navigate_slide":
                payload: dict = {"action": args.get("action", "next")}
                if args.get("action") == "goto" and args.get("slide_number"):
                    payload["slide_number"] = args["slide_number"]
                resp = httpx.post(f"{SLIDE_BASE}/api/navigate", json=payload, timeout=10)
                resp.raise_for_status()
                return resp.json()["summary"]

            if name == "get_current_slide":
                resp = httpx.get(f"{SLIDE_BASE}/api/info", timeout=5)
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

            return f"Unknown tool: {name}"
        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc)
            return f"Error: {exc}"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _joinly_settings(self) -> str:
        # Only fields accepted by Joinly's Settings model.
        # API keys (ElevenLabs, Deepgram) must be passed to the Docker
        # container via --tts-arg / --stt-arg at startup, not here.
        return json.dumps({
            "tts": config.JOINLY_TTS,
            "stt": config.JOINLY_STT,
            "name": config.PRESENTER_NAME,
        })
