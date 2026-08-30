from logger import logger
from langgraph.types import interrupt

from src.agent.state import AgentState, AudioAnalysis, FollowupDecision
from src.config.client import call_gemini_audio, call_structured_llm
from src.prompts import AUDIO_PROMPT, FOLLOWUP_PROMPT
from src.utils import fetch_rubric


def followup_gate_node(state: AgentState) -> dict:
    """Decide - before scoring - whether a clarifying follow-up is needed.

    Runs again after each answered follow-up. Stops asking once the response is
    clear enough or ``max_iterations`` is reached.
    """
    logger.info("Running followup_gate_node (iteration=%s)...", state.iteration)

    prior = ""
    if state.followup_question and state.followup_answer:
        prior = (
            f"A previous follow-up was already asked:\n"
            f"  Q: {state.followup_question}\n"
            f"  A: {state.followup_answer}\n"
        )

    # Free static lookup - no LLM call. Reused by plan_node so it need not refetch.
    rubric = state.rubric or fetch_rubric(state.input.domain)

    prompt = FOLLOWUP_PROMPT.format(
        actual_question=state.input.actual_question,
        user_response=state.input.user_response or "",
        prior_followup=prior,
        criteria=rubric.criteria,
        weightings=rubric.weightings,
    )
    decision = call_structured_llm(prompt, FollowupDecision)

    if decision.should_ask_followup and state.iteration < state.max_iterations:
        logger.info("followup_gate_node: follow-up required.")
        return {
            "rubric": rubric,
            "followup_decision": decision,
            "followup_question": decision.suggested_followup_question,
            "awaiting_followup_answer": True,
        }

    logger.info("followup_gate_node: proceeding to scoring.")
    return {
        "rubric": rubric,
        "followup_decision": decision,
        "awaiting_followup_answer": False,
    }


def ask_followup_node(state: AgentState) -> dict:
    """Pause the graph and surface the follow-up question to the caller.

    ``interrupt`` suspends execution; the value passed to ``resume`` becomes the
    return of this call when the graph is re-invoked. The reply may be a plain
    text string or ``{"audio_path": "..."}`` for a spoken follow-up answer, which
    is transcribed here so downstream nodes see a normal text answer.
    """
    logger.info("Running ask_followup_node - interrupting for follow-up answer.")
    reply = interrupt({"followup_question": state.followup_question})

    if isinstance(reply, dict) and reply.get("audio_path"):
        logger.info("ask_followup_node: transcribing spoken follow-up answer.")
        analysis = call_gemini_audio(
            AUDIO_PROMPT.format(actual_question=state.followup_question or ""),
            reply["audio_path"],
            AudioAnalysis,
        )
        answer = analysis.transcript
    else:
        answer = reply

    return {
        "followup_answer": answer,
        "awaiting_followup_answer": False,
        "iteration": state.iteration + 1,
    }
