"""
AI Meeting Presenter Agent
──────────────────────────
Connects to a running joinly MCP server, adds slide-navigation tools,
then drives the meeting: joins, starts screen-sharing, narrates slides,
and reacts to the client's voice in real time.

Flow:
  1. Connect to joinly MCP server (StreamableHttpTransport)
  2. Subscribe to transcript://live for real-time utterance events
  3. Build LangGraph ReAct agent with joinly tools + custom slide tools
  4. Join meeting, narrate first slide via speak_text
  5. Loop: each transcript event → run agent → speak_text / navigate_slide
"""
import asyncio
import json
import logging
from typing import Any

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent
from mcp import ResourceUpdatedNotification, ServerNotification
from pydantic import AnyUrl

from config import config
from presenter.prompt import get_presenter_prompt
from slide_server.manager import SlideManager

logger = logging.getLogger(__name__)

TRANSCRIPT_URL = AnyUrl("transcript://live")
SLIDE_SERVER_BASE = f"http://localhost:{config.SLIDE_SERVER_PORT}"


class PresenterAgent:
    """
    LangGraph-based AI presenter that drives a joinly-powered meeting.
    """

    def __init__(self, slide_manager: SlideManager):
        self.manager = slide_manager
        self._last_segment_time: float = -1.0

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    async def run(self, meeting_url: str) -> None:
        """Join the meeting, present slides, handle client input until done."""
        transcript_event = asyncio.Event()

        async def _on_notification(msg: Any) -> None:
            if (
                isinstance(msg, ServerNotification)
                and isinstance(msg.root, ResourceUpdatedNotification)
                and msg.root.params.uri == TRANSCRIPT_URL
            ):
                transcript_event.set()

        joinly_settings_dict: dict = {
            "tts": config.JOINLY_TTS,
            "stt": config.JOINLY_STT,
            "name": config.PRESENTER_NAME,
            "share_url": config.SLIDE_SHARE_URL,
        }
        if config.ELEVENLABS_API_KEY:
            joinly_settings_dict["elevenlabs_api_key"] = config.ELEVENLABS_API_KEY
        if config.DEEPGRAM_API_KEY:
            joinly_settings_dict["deepgram_api_key"] = config.DEEPGRAM_API_KEY
        joinly_settings = json.dumps(joinly_settings_dict)
        transport = StreamableHttpTransport(
            url=config.JOINLY_MCP_URL,
            headers={"joinly-settings": joinly_settings},
        )

        async with Client(transport, message_handler=_on_notification) as client:
            logger.info("Connected to joinly MCP server at %s", config.JOINLY_MCP_URL)

            await client.session.subscribe_resource(TRANSCRIPT_URL)

            # ── Build agent ──────────────────────────────────────────
            joinly_tools = await load_mcp_tools(client.session)
            slide_tools = self._make_slide_tools()

            @tool(return_direct=True)
            def end_turn() -> str:
                """End the current response turn. Always call this last."""
                return "Turn ended."

            all_tools = joinly_tools + slide_tools + [end_turn]

            llm = init_chat_model("claude-sonnet-4-6", model_provider="anthropic")
            llm_bound = llm.bind_tools(all_tools)
            tool_node = ToolNode(all_tools, handle_tool_errors=lambda e: str(e))

            agent = create_react_agent(
                llm_bound,
                tool_node,
                prompt=get_presenter_prompt(config.PRESENTER_NAME),
                checkpointer=MemorySaver(),
            )

            # ── Initialise: join meeting ─────────────────────────────
            logger.info("Joining meeting: %s", meeting_url)
            await client.call_tool("join_meeting", {"meeting_url": meeting_url})

            # ── Trigger first-slide narration ────────────────────────
            first = self.manager.current()
            if first:
                await self._run_agent(
                    agent,
                    f"[SYSTEM] The presentation has started. "
                    f"Please narrate the first slide now.\n{first.summary()}",
                )

            # ── Main loop: react to client speech ────────────────────
            logger.info("Entering main loop — listening for client input")
            while True:
                await transcript_event.wait()
                transcript_event.clear()

                try:
                    resources = await client.read_resource(TRANSCRIPT_URL)
                    raw = resources[0].text  # type: ignore[attr-defined]
                    transcript = json.loads(raw)
                except Exception as exc:
                    logger.warning("Failed to read transcript: %s", exc)
                    continue

                new_segments = [
                    s for s in transcript.get("segments", [])
                    if s.get("start", 0) > self._last_segment_time
                    and s.get("speaker_role") != "agent"  # ignore own speech
                ]
                if not new_segments:
                    continue

                self._last_segment_time = new_segments[-1].get("start", 0)
                user_text = "\n".join(
                    f"{s.get('speaker') or 'Client'}: {s['text']}"
                    for s in new_segments
                )
                logger.info("Client said: %s", user_text)

                await self._run_agent(agent, user_text)

    # ------------------------------------------------------------------ #
    # Agent runner
    # ------------------------------------------------------------------ #

    async def _run_agent(self, agent: Any, user_text: str) -> None:
        try:
            async for chunk in agent.astream(
                {"messages": [HumanMessage(content=user_text)]},
                config={"configurable": {"thread_id": "presenter"}},
                stream_mode="updates",
            ):
                if "agent" in chunk:
                    for msg in chunk["agent"]["messages"]:
                        for tc in getattr(msg, "tool_calls", []) or []:
                            args_str = ", ".join(
                                f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                                for k, v in tc.get("args", {}).items()
                            )
                            logger.info("→ %s(%s)", tc["name"], args_str)
        except asyncio.CancelledError:
            logger.debug("Agent run cancelled (client interrupted)")
        except Exception:
            logger.exception("Agent error")

    # ------------------------------------------------------------------ #
    # Slide tools  (LangChain @tool functions calling the slide server)
    # ------------------------------------------------------------------ #

    def _make_slide_tools(self) -> list:
        base = SLIDE_SERVER_BASE

        @tool
        def navigate_slide(action: str, slide_number: int = 0) -> str:
            """Navigate the presentation slides.

            Args:
                action: 'next' | 'previous' | 'goto' | 'repeat'
                slide_number: 1-based target slide (only required for 'goto')

            Returns the new slide's content so you can narrate it with speak_text.
            """
            payload: dict = {"action": action}
            if action == "goto" and slide_number:
                payload["slide_number"] = slide_number

            try:
                resp = httpx.post(f"{base}/api/navigate", json=payload, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                return data["summary"]
            except Exception as exc:
                logger.error("navigate_slide error: %s", exc)
                return f"Navigation failed: {exc}"

        @tool
        def get_current_slide() -> str:
            """Get the title, content, and speaker notes of the current slide."""
            try:
                resp = httpx.get(f"{base}/api/info", timeout=5)
                resp.raise_for_status()
                d = resp.json()
                if d["total"] == 0:
                    return "No slides are loaded yet."
                parts = [f"Slide {d['index']+1} of {d['total']}: {d['title']}"]
                if d["content"]:
                    parts.append(d["content"])
                if d["notes"]:
                    parts.append(f"[Notes: {d['notes']}]")
                return "\n".join(parts)
            except Exception as exc:
                logger.error("get_current_slide error: %s", exc)
                return f"Could not retrieve slide info: {exc}"

        return [navigate_slide, get_current_slide]
