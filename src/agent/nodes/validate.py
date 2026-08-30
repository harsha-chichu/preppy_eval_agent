from logger import logger

from src.agent.state import AgentState, ReviewResult
from src.config.client import call_structured_llm
from src.prompts import VALIDATE_PROMPT


def should_regenerate(state: AgentState) -> bool:
    """True when the last validation failed and a retry is still allowed.

    Shared by ``validate_node`` (to set ``is_final``) and the graph router so
    both agree on the same decision.
    """
    review = state.review_result
    if review is None:
        return False
    failed = review.needs_regeneration or not (
        review.is_grounded and review.is_rubric_consistent
    )
    return failed and state.validate_iteration < state.max_validate_iterations


def validate_node(state: AgentState) -> dict:
    """Audit the EvaluationResult against the plan.

    Does not re-grade the candidate - it checks whether ``execute_node`` did its
    job (score supported by feedback, plan and weightings followed, claims
    grounded). On failure the graph routes back to ``execute``, at most
    ``max_validate_iterations`` times.
    """
    logger.info("Running validate_node (validate_iteration=%s)...", state.validate_iteration)

    followup_block = ""
    if state.followup_answer:
        followup_block = (
            "Follow-up asked to resolve ambiguity:\n"
            f"  Q: {state.followup_question}\n"
            f"  A: {state.followup_answer}\n"
        )

    rubric = state.plan.rubric
    result = state.evaluation_result
    prompt = VALIDATE_PROMPT.format(
        actual_question=state.input.actual_question,
        user_response=state.input.user_response,
        followup_block=followup_block,
        criteria=rubric.criteria,
        weightings=rubric.weightings,
        plan=state.plan.plan,
        score=result.score,
        is_correct=result.is_correct,
        feedback=result.feedback,
        strengths=result.strengths,
        weaknesses=result.weaknesses,
        followup_evaluation=result.followup_evaluation,
    )
    review = call_structured_llm(prompt, ReviewResult, max_tokens=1200)

    next_iteration = state.validate_iteration + 1
    updated = state.model_copy(update={"review_result": review, "validate_iteration": next_iteration})
    retry = should_regenerate(updated)

    logger.info(
        "validate_node: needs_regeneration=%s grounded=%s rubric_consistent=%s -> %s",
        review.needs_regeneration,
        review.is_grounded,
        review.is_rubric_consistent,
        "re-execute" if retry else "done",
    )
    return {
        "review_result": review,
        "validate_iteration": next_iteration,
        "is_final": not retry,
    }
