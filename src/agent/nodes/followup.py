from logger import logger
from langgraph.types import interrupt

from src.agent.state import AgentState, FollowupDecision
from src.config.client import call_structured_llm
from src.prompts import FOLLOWUP_PROMPT


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

    prompt = FOLLOWUP_PROMPT.format(
        actual_question=state.input.actual_question,
        user_response=state.input.user_response,
        prior_followup=prior,
        plan=state.plan.plan if state.plan else "",
    )
    decision = call_structured_llm(prompt, FollowupDecision)

    if decision.should_ask_followup and state.iteration < state.max_iterations:
        logger.info("followup_gate_node: follow-up required.")
        return {
            "followup_decision": decision,
            "followup_question": decision.suggested_followup_question,
            "awaiting_followup_answer": True,
        }

    logger.info("followup_gate_node: proceeding to scoring.")
    return {"followup_decision": decision, "awaiting_followup_answer": False}


def ask_followup_node(state: AgentState) -> dict:
    """Pause the graph and surface the follow-up question to the caller.

    ``interrupt`` suspends execution; the value passed to ``resume`` becomes the
    return of this call when the graph is re-invoked.
    """
    logger.info("Running ask_followup_node - interrupting for follow-up answer.")
    answer = interrupt({"followup_question": state.followup_question})
    return {
        "followup_answer": answer,
        "awaiting_followup_answer": False,
        "iteration": state.iteration + 1,
    }
