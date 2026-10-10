"""Local CLI demo of the evaluation agent over the stateless start()/resume() API."""

import os
import sys

from src.agent.graph import resume, start
from src.agent.state import Difficulty, EvalStatus, UserInput
from src.config.tracing import langfuse

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
    if result.communication_evaluation:
        ce = result.communication_evaluation
        print("\n--- Communication ---")
        print(f"overall:    {result.overall_score}")
        print(
            f"grammar={ce.grammar_score} pronunciation={ce.pronunciation_score} "
            f"fluency={ce.fluency_score} clarity={ce.clarity_score}"
        )
        print(f"fillers={ce.filler_words_used} pauses={ce.pause_count}")
        print(f"feedback:   {ce.communication_feedback}")


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
    # Pass a path to a recorded answer as argv[1] to exercise the voice path;
    # otherwise a hard-coded typed answer is used.
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    user_input = UserInput(
        actual_question="What is machine learning and how does backpropagation work?",
        user_response=None if audio_path else "it is in the attention mechanism",
        audio_path=audio_path,
        difficulty=Difficulty.medium,
        domain="ai",
    )

    res = start(user_input)

    while res["status"] == EvalStatus.pending_followup:
        print(f"\nFollow-up needed: {res['followup_question']}")
        answer = input("Your answer (or a path to an audio file): ").strip()
        if os.path.isfile(answer):
            res = resume(res["thread_id"], audio_path=answer)
        else:
            res = resume(res["thread_id"], answer)

    _print_result(res["evaluation_result"])
    _print_review(res.get("review_result"))
    langfuse.flush()  # short-lived script: send buffered traces before exit


if __name__ == "__main__":
    main()
