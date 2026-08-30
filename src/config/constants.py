import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = os.getenv("MODEL", "claude-haiku-4-5-20251001")

# Multimodal model used to transcribe + assess spoken answers (litellm route).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash-lite")

# overall_score = CONTENT_WEIGHT * content_score + (1 - CONTENT_WEIGHT) * communication_score
CONTENT_WEIGHT = float(os.getenv("CONTENT_WEIGHT", "0.8"))
