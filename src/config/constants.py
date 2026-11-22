import os
from dotenv import load_dotenv

load_dotenv(override=True)  # .env wins over stale shell vars

# Used when MODEL starts with "anthropic/".
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Main text model used for followup, plan, execute, and validate nodes.
# Uses litellm routing: prefix "gemini/" for Google models, "anthropic/" for Anthropic.
MODEL = os.getenv("MODEL", "gemini/gemini-3.5-flash-lite")

# Multimodal model used to transcribe + assess spoken answers (litellm route).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-3.5-flash-lite")

# Retries for temporary provider errors (503 "high demand", 429, 5xx, timeouts,
# dropped connections), with exponential backoff starting at LLM_RETRY_BASE_S.
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "3"))
LLM_RETRY_BASE_S = float(os.getenv("LLM_RETRY_BASE_S", "1.0"))

# Optional litellm model to try once the retries are used up on a temporary error,
# e.g. a different Gemini model with separate capacity. Unset: no fallback.
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL") or None

# overall_score = CONTENT_WEIGHT * content_score + (1 - CONTENT_WEIGHT) * communication_score
CONTENT_WEIGHT = float(os.getenv("CONTENT_WEIGHT", "0.8"))
