"""
Text-to-Speech Engine
Primary:  ElevenLabs  (high-quality, natural voice)
Fallback: OpenAI TTS  (reliable, lower latency)

Audio is cached by content hash to avoid re-generating the same text.
"""
import hashlib
import logging
from pathlib import Path
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self):
        self._cache_dir = Path(config.AUDIO_CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._el_client = None
        self._oai_client = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def synthesize(self, text: str) -> bytes:
        """Return MP3 audio bytes for the given text (cached)."""
        if config.ELEVENLABS_API_KEY:
            return self._elevenlabs(text)
        if config.OPENAI_API_KEY:
            return self._openai_tts(text)
        raise RuntimeError(
            "No TTS engine configured. Set ELEVENLABS_API_KEY or OPENAI_API_KEY in .env"
        )

    def synthesize_to_file(self, text: str, output_path: str) -> str:
        """Synthesize and write to a file. Returns output_path."""
        audio = self.synthesize(text)
        with open(output_path, "wb") as fh:
            fh.write(audio)
        return output_path

    # ------------------------------------------------------------------ #
    # ElevenLabs
    # ------------------------------------------------------------------ #

    def _elevenlabs(self, text: str) -> bytes:
        cache_path = self._cache_path(text, "el")
        if cache_path.exists():
            return cache_path.read_bytes()

        if self._el_client is None:
            from elevenlabs import ElevenLabs
            self._el_client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)

        try:
            audio_iter = self._el_client.text_to_speech.convert(
                voice_id=config.ELEVENLABS_VOICE_ID,
                text=text,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
            audio_bytes = b"".join(audio_iter)
            cache_path.write_bytes(audio_bytes)
            logger.debug("ElevenLabs TTS: %d chars → %d bytes", len(text), len(audio_bytes))
            return audio_bytes
        except Exception as exc:
            logger.warning("ElevenLabs failed (%s), falling back to OpenAI TTS", exc)
            if config.OPENAI_API_KEY:
                return self._openai_tts(text)
            raise

    # ------------------------------------------------------------------ #
    # OpenAI TTS
    # ------------------------------------------------------------------ #

    def _openai_tts(self, text: str) -> bytes:
        cache_path = self._cache_path(text, "oai")
        if cache_path.exists():
            return cache_path.read_bytes()

        if self._oai_client is None:
            from openai import OpenAI
            self._oai_client = OpenAI(api_key=config.OPENAI_API_KEY)

        response = self._oai_client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=text,
            response_format="mp3",
        )
        audio_bytes = response.content
        cache_path.write_bytes(audio_bytes)
        logger.debug("OpenAI TTS: %d chars → %d bytes", len(text), len(audio_bytes))
        return audio_bytes

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #

    def _cache_path(self, text: str, prefix: str) -> Path:
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        return self._cache_dir / f"{prefix}_{digest}.mp3"
