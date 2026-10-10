"""Langfuse tracing for the evaluation agent.

Trace layout (one trace per graph segment, grouped into a session per evaluation):

    evaluate-answer (agent)                session_id = thread_id
    ├── transcribe-answer (chain)
    │   └── transcribe-and-assess-audio (generation, Gemini)
    ├── decide-followup (chain)
    │   └── decide-followup (generation)
    ├── plan-evaluation (chain)
    │   └── write-grading-plan (generation)
    ├── score-answer (chain)
    │   └── score-answer (generation)
    └── review-evaluation (evaluator)
        └── review-evaluation (generation)

A follow-up pauses the graph, so the resumed run is a second trace in the same
session. Tracing is a no-op when the LANGFUSE_* keys are not set.
"""

import functools
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from src.config import constants as _constants  # noqa: F401  (loads .env before the client reads it)
from langfuse import get_client, propagate_attributes

langfuse = get_client()

TRACE_NAME = "evaluate-answer"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def traced_node(name: str, as_type: str = "chain", when: Callable | None = None) -> Callable:
    """Wrap a LangGraph node in a Langfuse observation.

    Only the node's state update is recorded as output; the full AgentState is
    not captured as input (the child generation already shows the prompt).
    ``when(state)`` returning False skips the observation (e.g. a no-op node).
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state):
            if when is not None and not when(state):
                return fn(state)
            with langfuse.start_as_current_observation(as_type=as_type, name=name) as obs:
                update = fn(state)
                obs.update(output=_jsonable(update) or None)
                return update

        return wrapper

    return decorator


@contextmanager
def trace_generation(name: str, model: str, messages: list, temperature: float, max_tokens: int, attempt: int = 1):
    """Record one LiteLLM completion as a Langfuse generation.

    Yields a callback that takes the LiteLLM response and records output,
    resolved model and token usage (Langfuse derives cost from model + usage).
    """
    with langfuse.start_as_current_observation(
        as_type="generation",
        name=name,
        model=model,
        input=messages,
        model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        metadata={"attempt": attempt} if attempt > 1 else None,
    ) as gen:

        def record(response) -> None:
            usage = getattr(response, "usage", None)
            usage_details = None
            if usage is not None:
                usage_details = {
                    "input": usage.prompt_tokens or 0,
                    "output": usage.completion_tokens or 0,
                }
            gen.update(
                output=response.choices[0].message.content,
                model=getattr(response, "model", None) or model,
                usage_details=usage_details,
            )

        yield record


@contextmanager
def trace_run(thread_id: str, user_input, segment: str, run_input: dict) -> Iterator[Any]:
    """Root observation for one graph segment (initial run or a resume).

    ``thread_id`` is the session id, so the first pass and any follow-up turns
    of the same evaluation show up together in the Sessions view.
    """
    tags = [
        f"domain:{user_input.domain}",
        f"difficulty:{user_input.difficulty.value}",
        "answer:voice" if user_input.audio_path else "answer:text",
    ]
    with langfuse.start_as_current_observation(
        as_type="agent",
        name=TRACE_NAME,
        input=run_input,
        metadata={"segment": segment, "thread_id": thread_id},
    ) as root:
        with propagate_attributes(
            session_id=thread_id,
            trace_name=TRACE_NAME,
            tags=tags,
            metadata={"segment": segment},
        ):
            yield root


def start_input(user_input) -> dict:
    """Trace input for a first pass: what a reviewer needs to judge the run."""
    data = {
        "question": user_input.actual_question,
        "domain": user_input.domain,
        "difficulty": user_input.difficulty.value,
    }
    if user_input.audio_path:
        data["answer"] = "(voice answer - see transcribe-and-assess-audio)"
    else:
        data["answer"] = user_input.user_response
    return data


def resume_input(followup_question: str | None, answer: str | None, audio_path: str | None) -> dict:
    """Trace input for a resumed run (answer to a follow-up, or a skip)."""
    if audio_path:
        answer = "(voice answer - see transcribe-followup-audio)"
    return {"followup_question": followup_question, "followup_answer": answer}


def _field(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def record_outcome(root, evaluation_result=None, review_result=None, followup_question=None) -> None:
    """Set the trace output and attach the validator verdict as trace scores.

    Accepts pydantic models or their dumped dicts (server.py keeps dicts).
    """
    if followup_question:
        root.update(output={"status": "pending_followup", "followup_question": followup_question})
        return
    if evaluation_result is None:
        root.update(output={"status": "failed"}, level="ERROR")
        return

    output = dict(_jsonable(evaluation_result))
    output.pop("evaluation_plan_used", None)
    root.update(output={"status": "completed", **output})

    if review_result is not None:
        for name, key in (("grounded", "is_grounded"), ("rubric-consistent", "is_rubric_consistent")):
            root.score_trace(name=name, value=float(bool(_field(review_result, key))), data_type="BOOLEAN")
