import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LARAVEL_BASE_URL = os.getenv("LARAVEL_PUBLIC_API_BASE_URL", "http://localhost:8000/api/public")
INTERNAL_SECRET = os.getenv("CHATBOT_INTERNAL_SECRET", "")
REQUEST_TIMEOUT = int(os.getenv("CHATBOT_REQUEST_TIMEOUT_SECONDS", "30"))
RATE_LIMIT = os.getenv("CHATBOT_RATE_LIMIT_PER_IP", "10/minute")


def require_google_api_key() -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY belum di-set di .env")

    return GOOGLE_API_KEY
