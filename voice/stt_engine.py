"""
Speech-to-Text Engine — real-time streaming transcription.
Primary:  Deepgram Nova-2  (low-latency WebSocket streaming)
Fallback: OpenAI Whisper   (batch, triggered when Deepgram unavailable)

The engine is callback-based: register an on_transcript handler that receives
final transcript strings as they arrive.
"""
import asyncio
import logging
import threading
from typing import Callable, Optional

from config import config

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str], None]
ErrorCallback = Callable[[Exception], None]


class STTEngine:
    """
    Wraps Deepgram's async live-transcription API.
    Runs the async event loop in a dedicated daemon thread so callers
    do not need to manage asyncio themselves.
    """

    def __init__(
        self,
        on_transcript: TranscriptCallback,
        on_error: Optional[ErrorCallback] = None,
    ):
        self.on_transcript = on_transcript
        self.on_error = on_error
        self._connection = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._startup_ready = threading.Event()
        self._startup_error: Optional[Exception] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self):
        """Start background STT thread."""
        if not config.DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY not configured")

        self._running = True
        self._connected = False
        self._startup_ready.clear()
        self._startup_error = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="stt-thread"
        )
        self._thread.start()
        if self._startup_ready.wait(timeout=8) and self._startup_error:
            self._running = False
            raise RuntimeError("Deepgram STT startup failed") from self._startup_error
        logger.info("STT engine started (Deepgram)")

    def stop(self):
        """Signal the STT loop to shut down."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("STT engine stopped")

    def send_audio(self, audio_data: bytes):
        """Thread-safe: forward a raw PCM/WAV chunk to Deepgram."""
        if self._connection and self._loop and self._running and self._connected:
            asyncio.run_coroutine_threadsafe(
                self._async_send(audio_data), self._loop
            )

    # ------------------------------------------------------------------ #
    # Internal asyncio helpers
    # ------------------------------------------------------------------ #

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._stream())
        except Exception as exc:
            self._startup_error = exc
            logger.exception("Deepgram STT stream failed: %s", exc)
            if self.on_error:
                try:
                    self.on_error(exc)
                except Exception:
                    logger.exception("Deepgram STT error callback failed")
        finally:
            self._startup_ready.set()
            self._connected = False
            self._loop.close()

    async def _stream(self):
        from deepgram import AsyncDeepgramClient
        from deepgram.core.events import EventType
        from deepgram.listen.v1 import ListenV1Results

        dg = AsyncDeepgramClient(api_key=config.DEEPGRAM_API_KEY)

        async def _on_message(result):
            try:
                if not isinstance(result, ListenV1Results):
                    return
                alt = result.channel.alternatives[0]
                text = alt.transcript.strip()
                if text and result.is_final:
                    logger.debug("STT transcript: %s", text)
                    self.on_transcript(text)
            except Exception as exc:
                logger.warning("STT callback error: %s", exc)

        async with dg.listen.v1.connect(
            model="nova-3",
            language="en-US",
            smart_format="true",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results="true",
            vad_events="true",
            endpointing=300,
        ) as connection:
            self._connection = connection
            self._connected = True
            self._startup_ready.set()
            self._connection.on(EventType.MESSAGE, _on_message)

            listen_task = asyncio.create_task(self._connection.start_listening())
            keepalive_task = asyncio.create_task(self._keepalive_loop())

            try:
                while self._running:
                    await asyncio.sleep(0.1)
            finally:
                keepalive_task.cancel()
                try:
                    await self._connection.send_close_stream()
                except Exception:
                    pass
                await listen_task
                self._connection = None
                self._connected = False

    async def _async_send(self, data: bytes):
        if self._connection:
            await self._connection.send_media(data)

    async def _keepalive_loop(self):
        while self._running:
            await asyncio.sleep(3)
            if self._connection:
                try:
                    await self._connection.send_keep_alive()
                except Exception as exc:
                    logger.debug("STT keepalive failed: %s", exc)


class WhisperSTTEngine:
    """
    Batch Whisper fallback — transcribes audio files when Deepgram is unavailable.
    The caller writes audio to a temp file and calls transcribe_file().
    """

    def __init__(self, on_transcript: TranscriptCallback):
        self.on_transcript = on_transcript
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        return self._client

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe a WAV/MP3 file synchronously and invoke the callback."""
        client = self._get_client()
        with open(audio_path, "rb") as fh:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=fh,
                language="en",
            )
        text = result.text.strip()
        if text:
            self.on_transcript(text)
        return text


def create_stt_engine(
    on_transcript: TranscriptCallback,
    on_error: Optional[ErrorCallback] = None,
) -> STTEngine:
    """Factory: returns Deepgram engine if key is present, else raises."""
    if config.DEEPGRAM_API_KEY:
        return STTEngine(on_transcript, on_error=on_error)
    raise RuntimeError(
        "No STT engine configured. Set DEEPGRAM_API_KEY in .env"
    )
