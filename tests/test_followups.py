"""Follow-ups: every follow-up reaches scoring, and each gets its own suggested answer.

Run:  uv run pytest tests  (no API keys or network needed)
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("EVAL_AGENT_TOKEN", "test-token")

import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from src.agent.nodes import followup as followup_mod  # noqa: E402
from src.agent.state import (  # noqa: E402
    AgentState,
    EvaluationResult,
    FollowupTurn,
    UserInput,
    render_followups,
)
from src.prompts import EXECUTE_PROMPT  # noqa: E402


def _state(**kw) -> AgentState:
    user = UserInput(actual_question="what is a cache?", user_response="it stores stuff", difficulty="medium", domain="backend")
    return AgentState(input=user, **kw)


def test_render_followups_lists_every_round_and_marks_skips():
    text = render_followups(
        [FollowupTurn(question="how do you invalidate it?", answer="ttl"), FollowupTurn(question="what about stampedes?")]
    )
    assert "1. Q: how do you invalidate it?" in text
    assert "A: ttl" in text
    assert "2. Q: what about stampedes?" in text
    assert "skipped" in text
    assert render_followups([]) == ""


def test_ask_followup_appends_instead_of_overwriting(monkeypatch):
    monkeypatch.setattr(followup_mod, "interrupt", lambda _payload: "second answer")
    first = FollowupTurn(question="first?", answer="first answer")
    state = _state(followups=[first], followup_question="second?", awaiting_followup_answer=True, iteration=1)

    update = followup_mod.ask_followup_node.__wrapped__(state)

    assert [f.question for f in update["followups"]] == ["first?", "second?"]
    assert [f.answer for f in update["followups"]] == ["first answer", "second answer"]
    assert update["iteration"] == 2


def test_evaluation_result_requires_followup_suggestions():
    base = dict(score=7, is_correct=True, feedback="ok", suggested_answer="a", suggested_points=["p"])
    with pytest.raises(ValidationError):
        EvaluationResult(**base)
    result = EvaluationResult(**base, followup_suggestions=[{"question": "q?", "suggested_answer": "a"}])
    assert result.followup_suggestions[0].suggested_answer == "a"


def test_execute_prompt_asks_for_followup_suggestions():
    assert "followup_suggestions" in EXECUTE_PROMPT
    assert "{followup_block}" in EXECUTE_PROMPT
