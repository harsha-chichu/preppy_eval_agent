"""LangGraph wiring for the evaluation agent.

Flow:  [transcribe] -> followup_gate -> (ask_followup -> followup_gate)*
       -> (plan || assess_delivery) -> execute -> validate -> (execute -> validate)*  -> END

The caller waits only until ``followup_gate`` has decided, so everything before
it is latency the user feels. Keep that path to at most one LLM call:

* ``transcribe`` runs first ONLY for a recording sent without text - the gate
  needs the words. When the user's reviewed text came with the recording, the
  slow Gemini audio call is only needed for delivery scores, so it runs as
  ``assess_delivery`` after the gate, in parallel with ``plan``.
* ``followup_gate`` skips its LLM call when no follow-up is allowed (easy mode,
  or the follow-up budget is spent). It decides using the question, the answer
  and the fixed rubric (a free static lookup) - not the plan, whose LLM call is
  deferred until after the follow-up loop clears.

``ask_followup`` calls ``interrupt()``, so the graph pauses there and hands the
follow-up question back to the caller. The caller collects an answer and calls
``resume()`` to continue into scoring.

``validate`` audits the produced ``EvaluationResult`` against the plan; if it is
not grounded / rubric-consistent it routes back to ``execute``, at most
``max_validate_iterations`` (default 2) times.

Public surface is stateless: ``start()`` / ``resume()`` keyed by ``thread_id``.
"""

from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from logger import logger
from src.agent import state as _state
from src.agent.nodes import (
    ask_followup_node,
    assess_delivery_node,
    execute_node,
    followup_gate_node,
    plan_node,
    should_regenerate,
    transcribe_node,
    validate_node,
)
from src.agent.state import AgentState, EvalStatus, UserInput
from src.config.tracing import record_outcome, resume_input, start_input, trace_run

# Our custom state types must be on the checkpoint deserialization allowlist -
# unregistered types are only warned about today but will be blocked by a
# future langgraph. List every model/enum that can land in AgentState.
_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        _state.AgentState,
        _state.UserInput,
        _state.Difficulty,
        _state.EvalStatus,
        _state.Rubric,
        _state.EvaluationPlan,
        _state.FollowupDecision,
        _state.CommunicationEvaluation,
        _state.AudioAnalysis,
        _state.EvaluationResult,
        _state.ReviewResult,
    ]
)


def route_start(state: AgentState) -> str:
    # Only a recording without text needs transcribing before the gate.
    return "transcribe" if state.input.audio_path and not state.input.user_response else "followup_gate"


def route_after_gate(state: AgentState) -> str | list[str]:
    # plan and assess_delivery are independent; run them side by side.
    return "ask_followup" if state.awaiting_followup_answer else ["plan", "assess_delivery"]


def route_after_validate(state: AgentState) -> str:
    return "execute" if should_regenerate(state) else END


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("transcribe", transcribe_node)
    builder.add_node("plan", plan_node)
    builder.add_node("followup_gate", followup_gate_node)
    builder.add_node("ask_followup", ask_followup_node)
    builder.add_node("execute", execute_node)
    builder.add_node("validate", validate_node)
    builder.add_node("assess_delivery", assess_delivery_node)

    builder.add_conditional_edges(START, route_start, ["transcribe", "followup_gate"])
    builder.add_edge("transcribe", "followup_gate")
    builder.add_conditional_edges(
        "followup_gate",
        route_after_gate,
        ["ask_followup", "plan", "assess_delivery"],
    )
    builder.add_edge("ask_followup", "followup_gate")
    # execute waits for both branches.
    builder.add_edge(["plan", "assess_delivery"], "execute")
    builder.add_edge("execute", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {"execute": "execute", END: END},
    )

    return builder.compile(checkpointer=MemorySaver(serde=_SERDE))


_GRAPH = build_graph()


def _get(out, key):
    return out.get(key) if isinstance(out, dict) else getattr(out, key, None)


def _package(thread_id: str, out) -> dict:
    interrupts = out.get("__interrupt__") if isinstance(out, dict) else None
    if interrupts:
        return {
            "thread_id": thread_id,
            "status": EvalStatus.pending_followup,
            "followup_question": interrupts[0].value["followup_question"],
        }
    return {
        "thread_id": thread_id,
        "status": EvalStatus.completed,
        "evaluation_result": _get(out, "evaluation_result"),
        "review_result": _get(out, "review_result"),
        "transcript": _get(out, "transcript"),
    }


def start(user_input: UserInput, thread_id: str | None = None, user_id: str | None = None) -> dict:
    """Run plan + follow-up gate. Returns either a completed evaluation or a
    ``pending_followup`` payload carrying the question and ``thread_id``.

    ``user_id``, when provided, is stored in the graph state and attached to the
    Langfuse trace (and to any resumed follow-up segment) for per-user analytics.
    """
    thread_id = thread_id or uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("graph.start thread_id=%s", thread_id)
    with trace_run(thread_id, user_input, "start", start_input(user_input), user_id=user_id) as root:
        out = _GRAPH.invoke(AgentState(input=user_input, user_id=user_id), config)
        packaged = _package(thread_id, out)
        _record(root, packaged)
    return packaged


def resume(
    thread_id: str,
    answer: str | None = None,
    audio_path: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Supply the follow-up answer for ``thread_id`` and finish scoring (or ask
    another follow-up, up to ``max_iterations``).

    The answer may be typed (``answer``) or spoken (``audio_path``, transcribed
    inside ``ask_followup_node``). ``user_id`` defaults to the one stored by
    ``start()`` so every segment of an evaluation traces under the same user."""
    if not answer and not audio_path:
        raise ValueError("resume() needs answer or audio_path")
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("graph.resume thread_id=%s audio=%s", thread_id, bool(audio_path))
    payload = {"audio_path": audio_path} if audio_path else answer
    values = _GRAPH.get_state(config).values
    user_id = user_id or _get(values, "user_id")
    run_input = resume_input(_get(values, "followup_question"), answer, audio_path)
    with trace_run(thread_id, _get(values, "input"), "followup", run_input, user_id=user_id) as root:
        out = _GRAPH.invoke(Command(resume=payload), config)
        packaged = _package(thread_id, out)
        _record(root, packaged)
    return packaged


def _record(root, packaged: dict) -> None:
    record_outcome(
        root,
        evaluation_result=packaged.get("evaluation_result"),
        review_result=packaged.get("review_result"),
        followup_question=packaged.get("followup_question"),
    )
