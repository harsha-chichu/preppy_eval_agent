"""Prompt templates for the evaluation agent nodes.

Each template lives in its own module; they are re-exported here so callers can
do ``from src.prompts import PLAN_PROMPT``. All templates are plain
``str.format`` strings - the rubric text is rendered by the calling node from
the static registry (``fetch_rubric``); the model never invents a rubric.
"""

from src.prompts.audio import AUDIO_PROMPT
from src.prompts.execute import EXECUTE_PROMPT
from src.prompts.followup import FOLLOWUP_PROMPT
from src.prompts.plan import PLAN_PROMPT
from src.prompts.validate import VALIDATE_PROMPT

__all__ = ["PLAN_PROMPT", "FOLLOWUP_PROMPT", "EXECUTE_PROMPT", "VALIDATE_PROMPT", "AUDIO_PROMPT"]
