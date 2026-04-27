import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # AI
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Joinly
    JOINLY_MCP_URL: str = os.getenv("JOINLY_MCP_URL", "http://localhost:8000/mcp/")
    JOINLY_TTS: str = os.getenv("JOINLY_TTS", "kokoro")
    JOINLY_STT: str = os.getenv("JOINLY_STT", "whisper")

    # Slide server
    SLIDE_SERVER_HOST: str = os.getenv("SLIDE_SERVER_HOST", "0.0.0.0")
    SLIDE_SERVER_PORT: int = int(os.getenv("SLIDE_SERVER_PORT", "8080"))
    SLIDE_SHARE_URL: str = os.getenv("SLIDE_SHARE_URL", "http://localhost:8080/present")

    # Presenter
    PRESENTER_NAME: str = os.getenv("PRESENTER_NAME", "Alex")


config = Config()
