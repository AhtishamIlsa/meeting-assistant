import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # AI Brain
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Voice Output
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Voice Input
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")

    # Meeting Bot
    RECALL_AI_API_KEY: str = os.getenv("RECALL_AI_API_KEY", "")
    RECALL_AI_REGION: str = os.getenv("RECALL_AI_REGION", "us-east-1")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Storage
    UPLOAD_DIR: str = "uploads"
    AUDIO_CACHE_DIR: str = "audio_cache"
    SLIDE_IMAGE_DIR: str = "slide_images"

    def __post_init__(self):
        Path(self.UPLOAD_DIR).mkdir(exist_ok=True)
        Path(self.AUDIO_CACHE_DIR).mkdir(exist_ok=True)
        Path(self.SLIDE_IMAGE_DIR).mkdir(exist_ok=True)


config = Config()
