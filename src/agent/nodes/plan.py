import sys

from logger import logger
from src.agent.state import AgentState, Difficulty, EvaluationPlan, UserInput
from src.config.client import call_llm
from src.prompts import PLAN_PROMPT
from src.utils import fetch_rubric

sys.stdout.reconfigure(encoding="utf-8")


def plan_node(state: AgentState) -> dict:
    """Build the evaluation plan.

    The rubric is taken verbatim from the static registry; the LLM only writes
    the grading instructions for that rubric.
    """
    logger.info("Running plan_node...")

    rubric = state.rubric or fetch_rubric(state.input.domain)
    prompt = PLAN_PROMPT.format(
        actual_question=state.input.actual_question,
        domain=state.input.domain,
        difficulty=state.input.difficulty.value,
        criteria=rubric.criteria,
        weightings=rubric.weightings,
    )
    plan_text = call_llm(prompt)

    plan = EvaluationPlan(rubric=rubric, plan=plan_text)
    logger.info("plan_node completed.")
    return {"plan": plan}


if __name__ == "__main__":
    sample_input = UserInput(
        actual_question="what is backpropagation?",
        user_response="backpropagation is used to find gradients.",
        difficulty=Difficulty.easy,
        domain="ai",
    )
    result = plan_node(AgentState(input=sample_input))
    print(result["plan"].model_dump_json(indent=2))
