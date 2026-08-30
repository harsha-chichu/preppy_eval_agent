"""LangGraph wiring for the evaluation agent.

Flow:  plan -> followup_gate -> (ask_followup -> followup_gate)* -> execute
       -> validate -> (execute -> validate)*  -> END

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
    execute_node,
    followup_gate_node,
    plan_node,
    should_regenerate,
    validate_node,
)
from src.agent.state import AgentState, EvalStatus, UserInput

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
        _state.EvaluationResult,
        _state.ReviewResult,
    ]
)


def route_after_gate(state: AgentState) -> str:
    return "ask_followup" if state.awaiting_followup_answer else "execute"


def route_after_validate(state: AgentState) -> str:
    return "execute" if should_regenerate(state) else END


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("followup_gate", followup_gate_node)
    builder.add_node("ask_followup", ask_followup_node)
    builder.add_node("execute", execute_node)
    builder.add_node("validate", validate_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "followup_gate")
    builder.add_conditional_edges(
        "followup_gate",
        route_after_gate,
        {"ask_followup": "ask_followup", "execute": "execute"},
    )
    builder.add_edge("ask_followup", "followup_gate")
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
    }


def start(user_input: UserInput, thread_id: str | None = None) -> dict:
    """Run plan + follow-up gate. Returns either a completed evaluation or a
    ``pending_followup`` payload carrying the question and ``thread_id``."""
    thread_id = thread_id or uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("graph.start thread_id=%s", thread_id)
    out = _GRAPH.invoke(AgentState(input=user_input), config)
    return _package(thread_id, out)


def resume(thread_id: str, answer: str) -> dict:
    """Supply the follow-up answer for ``thread_id`` and finish scoring (or ask
    another follow-up, up to ``max_iterations``)."""
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("graph.resume thread_id=%s", thread_id)
    out = _GRAPH.invoke(Command(resume=answer), config)
    return _package(thread_id, out)
