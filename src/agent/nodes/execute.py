from logger import logger

from src.agent.state import AgentState, EvaluationResult
from src.config.client import call_structured_llm
from src.prompts import EXECUTE_PROMPT


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
    result = call_structured_llm(prompt, EvaluationResult, max_tokens=2000)

    result.evaluation_plan_used = state.plan
    logger.info("execute_node completed (score=%s).", result.score)
    return {"evaluation_result": result}
