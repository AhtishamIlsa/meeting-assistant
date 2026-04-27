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


class STTEngine:
    """
    Wraps Deepgram's async live-transcription API.
    Runs the async event loop in a dedicated daemon thread so callers
    do not need to manage asyncio themselves.
    """

    def __init__(self, on_transcript: TranscriptCallback):
        self.on_transcript = on_transcript
        self._connection = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self):
        """Start background STT thread."""
        if not config.DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY not configured")

        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="stt-thread"
        )
        self._thread.start()
        logger.info("STT engine started (Deepgram)")

    def stop(self):
        """Signal the STT loop to shut down."""
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("STT engine stopped")

    def send_audio(self, audio_data: bytes):
        """Thread-safe: forward a raw PCM/WAV chunk to Deepgram."""
        if self._connection and self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._async_send(audio_data), self._loop
            )

    # ------------------------------------------------------------------ #
    # Internal asyncio helpers
    # ------------------------------------------------------------------ #

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._stream())

    async def _stream(self):
        from deepgram import (
            DeepgramClient,
            LiveTranscriptionEvents,
            LiveOptions,
        )

        dg = DeepgramClient(config.DEEPGRAM_API_KEY)
        self._connection = dg.listen.asynclive.v("1")

        async def _on_transcript(_, result, **__):
            try:
                alt = result.channel.alternatives[0]
                text = alt.transcript.strip()
                if text and result.is_final:
                    logger.debug("STT transcript: %s", text)
                    self.on_transcript(text)
            except Exception as exc:
                logger.warning("STT callback error: %s", exc)

        self._connection.on(LiveTranscriptionEvents.Transcript, _on_transcript)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            interim_results=False,
            endpointing=500,
            encoding="linear16",
            sample_rate=16000,
        )

        await self._connection.start(options)

        while self._running:
            await asyncio.sleep(0.1)

        await self._connection.finish()

    async def _async_send(self, data: bytes):
        if self._connection:
            await self._connection.send(data)


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


def create_stt_engine(on_transcript: TranscriptCallback) -> STTEngine:
    """Factory: returns Deepgram engine if key is present, else raises."""
    if config.DEEPGRAM_API_KEY:
        return STTEngine(on_transcript)
    raise RuntimeError(
        "No STT engine configured. Set DEEPGRAM_API_KEY in .env"
    )
