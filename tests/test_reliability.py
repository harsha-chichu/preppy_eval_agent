"""Reliability: retries on temporary provider errors, and a server that keeps
answering while an evaluation waits on its follow-up decision.

Run:  uv run pytest tests  (no API keys or network needed)
"""

import asyncio
import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("EVAL_AGENT_TOKEN", "test-token")

import httpx  # noqa: E402
import litellm  # noqa: E402
import pytest  # noqa: E402

from src.config import client  # noqa: E402


def _ok(text: str = '{"ok": true}'):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=None, model="m")


def _overloaded():
    return litellm.ServiceUnavailableError(
        message="This model is currently experiencing high demand.", llm_provider="gemini", model="gemini/x"
    )


@pytest.fixture
def calls(monkeypatch):
    """Records the model of each completion call; no real sleeping between retries."""
    seen: list[str] = []
    monkeypatch.setattr(client.time, "sleep", lambda s: None)
    return seen


def _fake_completion(seen, outcomes):
    outcomes = iter(outcomes)

    def completion(model, **kwargs):
        seen.append(model)
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return completion


def test_retries_gemini_503_then_succeeds(monkeypatch, calls):
    monkeypatch.setattr(client, "completion", _fake_completion(calls, [_overloaded(), _overloaded(), _ok()]))
    response = client._complete("decide-followup", "gemini/x", [], 0.2, 100)
    assert response.choices[0].message.content == '{"ok": true}'
    assert calls == ["gemini/x"] * 3


def test_gives_up_after_the_retries(monkeypatch, calls):
    monkeypatch.setattr(client, "LLM_RETRIES", 2)
    monkeypatch.setattr(client, "FALLBACK_MODEL", None)
    monkeypatch.setattr(client, "completion", _fake_completion(calls, [_overloaded()] * 3))
    with pytest.raises(litellm.ServiceUnavailableError):
        client._complete("execute", "gemini/x", [], 0.2, 100)
    assert len(calls) == 3


def test_falls_back_to_another_model_once_retries_are_used_up(monkeypatch, calls):
    monkeypatch.setattr(client, "LLM_RETRIES", 1)
    monkeypatch.setattr(client, "FALLBACK_MODEL", "gemini/backup")
    monkeypatch.setattr(client, "completion", _fake_completion(calls, [_overloaded(), _overloaded(), _ok()]))
    client._complete("execute", "gemini/x", [], 0.2, 100)
    assert calls == ["gemini/x", "gemini/x", "gemini/backup"]


def test_does_not_retry_a_bad_request(monkeypatch, calls):
    bad = litellm.BadRequestError(message="invalid schema", llm_provider="gemini", model="gemini/x")
    monkeypatch.setattr(client, "completion", _fake_completion(calls, [bad]))
    with pytest.raises(litellm.BadRequestError):
        client._complete("execute", "gemini/x", [], 0.2, 100)
    assert len(calls) == 1


def test_server_answers_other_requests_while_a_decision_is_pending(monkeypatch):
    import server

    def slow_decision(thread_id, run, graph_input, user_input, segment, run_input):
        time.sleep(1.0)  # stand-in for the follow-up gate's Gemini call
        run.status = "scoring"
        run.decided.set()

    monkeypatch.setattr(server, "_drive", slow_decision)

    async def scenario():
        transport = httpx.ASGITransport(app=server.app)
        headers = {"x-agent-token": "test-token"}
        form = {"question": "q?", "difficulty": "medium", "domain": "backend", "answer": "a"}
        async with httpx.AsyncClient(transport=transport, base_url="http://agent", headers=headers) as c:

            async def submit():
                started = time.perf_counter()
                res = await c.post("/evaluations", data=form)
                assert res.status_code == 200 and res.json()["status"] == "scoring"
                return time.perf_counter() - started

            async def health():
                await asyncio.sleep(0.2)
                started = time.perf_counter()
                assert (await c.get("/health")).status_code == 200
                return time.perf_counter() - started

            started = time.perf_counter()
            *submits, health_wait = await asyncio.gather(submit(), submit(), submit(), health())
            return time.perf_counter() - started, submits, health_wait

    total, submits, health_wait = asyncio.run(scenario())
    # Blocking waits would serialize these: ~3s in total, health stuck behind them.
    assert total < 2.0, (total, submits)
    assert health_wait < 0.5
    assert threading.active_count() < 50
