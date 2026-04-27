"""
AI Meeting Presenter — launcher
────────────────────────────────
Starts the slide server (FastAPI on :8080) and optionally immediately
kicks off a presentation session if --meeting-url is provided.

Usage:
    # Open setup UI in your browser at http://localhost:8080, upload slides,
    # enter a Zoom link and click Start:
    python main.py

    # Or start a session directly from the CLI:
    python main.py --meeting-url "https://zoom.us/j/123456" --slides deck.pptx
"""
import argparse
import asyncio
import logging
import os
import signal
import threading

import uvicorn

from config import config
from slide_server.app import app as slide_app
from slide_server.app import manager as slide_manager
from slide_server import app as slide_app_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Slide server thread
# ------------------------------------------------------------------ #

def _run_slide_server():
    uvicorn.run(
        slide_app,
        host=config.SLIDE_SERVER_HOST,
        port=config.SLIDE_SERVER_PORT,
        log_level="warning",
    )


# ------------------------------------------------------------------ #
# Presenter agent
# ------------------------------------------------------------------ #

async def _run_presenter(meeting_url: str):
    from presenter.agent import PresenterAgent
    agent = PresenterAgent(slide_manager)
    await agent.run(meeting_url)


def _start_presenter(meeting_url: str):
    """Called from slide server's /api/start endpoint or directly from CLI."""
    asyncio.run(_run_presenter(meeting_url))


# ------------------------------------------------------------------ #
# Wire /api/start callback into the slide server
# ------------------------------------------------------------------ #

def _setup_start_callback():
    def _cb(meeting_url: str):
        t = threading.Thread(target=_start_presenter, args=(meeting_url,), daemon=True)
        t.start()
        return asyncio.sleep(0)   # returns a coroutine for create_task

    slide_app_module.app._start_callback = _cb  # type: ignore[attr-defined]
    # Patch the module-level variable used in app.py
    import slide_server.app as _sa
    _sa._start_callback = _cb


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="AI Meeting Presenter")
    parser.add_argument("--meeting-url", default=None,
                        help="Zoom/Meet URL to join immediately")
    parser.add_argument("--slides", default=None,
                        help="Path to PPTX or PDF to load immediately")
    args = parser.parse_args()

    # Ensure storage directories exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("slide_images", exist_ok=True)

    # Pre-load slides if provided via CLI
    if args.slides:
        count = slide_manager.load(args.slides)
        logger.info("Loaded %d slides from %s", count, args.slides)

    _setup_start_callback()

    # Start slide server in background thread
    server_thread = threading.Thread(target=_run_slide_server, daemon=True)
    server_thread.start()
    logger.info(
        "Slide server running at http://%s:%d",
        "localhost", config.SLIDE_SERVER_PORT,
    )

    # Graceful shutdown
    def _shutdown(sig, _frame):
        logger.info("Shutting down…")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.meeting_url:
        # CLI mode: start presentation immediately
        if not slide_manager.is_loaded:
            logger.error("No slides loaded. Pass --slides path/to/deck.pptx")
            raise SystemExit(1)
        logger.info("Starting presentation for %s", args.meeting_url)
        asyncio.run(_run_presenter(args.meeting_url))
    else:
        # UI mode: wait for user to start via browser
        logger.info(
            "\n"
            "  ┌─────────────────────────────────────────┐\n"
            "  │  Open http://localhost:%d in your browser │\n"
            "  │  Upload slides and enter your Zoom URL   │\n"
            "  └─────────────────────────────────────────┘",
            config.SLIDE_SERVER_PORT,
        )
        server_thread.join()


if __name__ == "__main__":
    main()
