import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env is loaded from the project root
ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# API Key configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Model configuration
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Document configuration
PDF_FILE_PATH = ROOT_DIR / "rag_file.pdf"
CACHE_DIR = ROOT_DIR / ".rag_cache"


def is_api_key_valid() -> bool:
    """Check if a valid Gemini API key is configured."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return False
    key = key.strip()
    if key in ("", "your_gemini_api_key_here", "YOUR_API_KEY"):
        return False
    return True


def get_gemini_client():
    """
    Initialize and return Google GenAI Client.
    Raises ValueError if API key is not configured.
    """
    # Reload in case .env was edited during runtime
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not key or key.strip() in ("", "your_gemini_api_key_here", "YOUR_API_KEY"):
        raise ValueError(
            "\n" + "=" * 65 + "\n"
            "⚠️  GEMINI API KEY MISSING!\n\n"
            "Please open the '.env' file in this project and add your API key:\n"
            "    GEMINI_API_KEY=AIzaSy...\n\n"
            "You can get an API key for free at: https://aistudio.google.com/app/apikey\n"
            + "=" * 65
        )

    from google import genai
    return genai.Client(api_key=key.strip())
