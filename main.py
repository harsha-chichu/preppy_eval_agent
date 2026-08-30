"""Local CLI demo of the evaluation agent over the stateless start()/resume() API."""

import sys

from src.agent.graph import resume, start
from src.agent.state import Difficulty, EvalStatus, UserInput

sys.stdout.reconfigure(encoding="utf-8")


def _print_result(result) -> None:
    if result is None:
        print("No evaluation result.")
        return
    print("\n=== Evaluation ===")
    print(f"score:      {result.score}")
    print(f"is_correct: {result.is_correct}")
    print(f"feedback:   {result.feedback}")
    print(f"strengths:  {result.strengths}")
    print(f"weaknesses: {result.weaknesses}")
    if result.followup_evaluation:
        print(f"followup:   {result.followup_evaluation}")


def _print_review(review) -> None:
    if review is None:
        return
    print("\n=== Validation ===")
    print(f"grounded:          {review.is_grounded}")
    print(f"rubric_consistent: {review.is_rubric_consistent}")
    print(f"needs_regeneration:{review.needs_regeneration}")
    if review.issues_found:
        print(f"issues:            {review.issues_found}")


def main() -> None:
    user_input = UserInput(
        actual_question="Explain how a transformer handles long-range dependencies.",
        user_response="it is in the attention mechanism",
        difficulty=Difficulty.hard,
        domain="ai",
    )

    res = start(user_input)

    while res["status"] == EvalStatus.pending_followup:
        print(f"\nFollow-up needed: {res['followup_question']}")
        answer = input("Your answer: ").strip()
        res = resume(res["thread_id"], answer)

    _print_result(res["evaluation_result"])
    _print_review(res.get("review_result"))


if __name__ == "__main__":
    main()
