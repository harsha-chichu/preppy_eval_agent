from logger import logger

from src.agent.state import AgentState, EvaluationResult
from src.config.client import call_structured_llm
from src.config.constants import CONTENT_WEIGHT
from src.config.tracing import traced_node
from src.prompts import EXECUTE_PROMPT


@traced_node("score-answer")
def execute_node(state: AgentState) -> dict:
    """Run the full evaluation using the plan, rubric, and any follow-up answer."""
    logger.info("Running execute_node...")

    followup_block = ""
    if state.followup_answer:
        followup_block = (
            "Follow-up asked to resolve ambiguity:\n"
            f"  Q: {state.followup_question}\n"
            f"  A: {state.followup_answer}\n"
        )

    rubric = state.plan.rubric
    prompt = EXECUTE_PROMPT.format(
        actual_question=state.input.actual_question,
        user_response=state.input.user_response,
        followup_block=followup_block,
        criteria=rubric.criteria,
        weightings=rubric.weightings,
        plan=state.plan.plan,
    )
    result = call_structured_llm(prompt, EvaluationResult, name="score-answer", max_tokens=2000)

    result.evaluation_plan_used = state.plan

    # Voice answers: fold in the delivery assessment and a blended overall score.
    # The content LLM leaves these null; we fill them deterministically here so
    # the model never invents communication sub-scores.
    if state.communication_evaluation:
        comm = state.communication_evaluation
        comm_score = (
            comm.grammar_score
            + comm.pronunciation_score
            + comm.fluency_score
            + comm.clarity_score
        ) / 4
        result.communication_evaluation = comm
        result.overall_score = round(
            CONTENT_WEIGHT * result.score + (1 - CONTENT_WEIGHT) * comm_score, 2
        )

    logger.info("execute_node completed (score=%s, overall=%s).", result.score, result.overall_score)
    return {"evaluation_result": result}
