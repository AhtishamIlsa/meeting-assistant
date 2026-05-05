import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # AI
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Recall.ai
    RECALL_API_KEY: str = os.getenv("RECALL_API_KEY", "")
    RECALL_REGION: str = os.getenv("RECALL_REGION", "us-east-1")
    RECALL_TRANSCRIPTION_PROVIDER: str = os.getenv(
        "RECALL_TRANSCRIPTION_PROVIDER", "recallai_streaming"
    )
    RECALL_TRANSCRIPTION_MODE: str = os.getenv(
        "RECALL_TRANSCRIPTION_MODE", "prioritize_low_latency"
    )
    RECALL_TRANSCRIPTION_LANGUAGE: str = os.getenv(
        "RECALL_TRANSCRIPTION_LANGUAGE", "en"
    )

    # Joinly
    JOINLY_MCP_URL: str = os.getenv("JOINLY_MCP_URL", "http://localhost:8000/mcp/")
    JOINLY_TTS: str = os.getenv("JOINLY_TTS", "kokoro")
    JOINLY_STT: str = os.getenv("JOINLY_STT", "whisper")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")

    # Slide server
    SLIDE_SERVER_HOST: str = os.getenv("SLIDE_SERVER_HOST", "0.0.0.0")
    SLIDE_SERVER_PORT: int = int(os.getenv("SLIDE_SERVER_PORT", "8080"))
    SLIDE_SHARE_URL: str = os.getenv("SLIDE_SHARE_URL", "http://localhost:8080/present")
    # Public URL for Recall to push real-time transcript events (ngrok/cloudflared)
    RECALL_WEBHOOK_URL: str = os.getenv("RECALL_WEBHOOK_URL", "")

    # Presenter
    PRESENTER_NAME: str = os.getenv("PRESENTER_NAME", "Alex")


config = Config()
