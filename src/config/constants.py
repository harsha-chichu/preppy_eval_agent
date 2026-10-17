import os
from dotenv import load_dotenv

load_dotenv()

# Optional — only needed if the MODEL is switched back to an Anthropic model.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Main text model used for followup, plan, execute, and validate nodes.
# Uses litellm routing: prefix "gemini/" for Google models, "anthropic/" for Anthropic.
MODEL = os.getenv("MODEL", "gemini/gemini-3.5-flash-lite")

# Multimodal model used to transcribe + assess spoken answers (litellm route).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash-lite")

# overall_score = CONTENT_WEIGHT * content_score + (1 - CONTENT_WEIGHT) * communication_score
CONTENT_WEIGHT = float(os.getenv("CONTENT_WEIGHT", "0.8"))
