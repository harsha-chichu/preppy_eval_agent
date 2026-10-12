"""Prompt for ``execute_node`` - produces the final EvaluationResult."""

EXECUTE_PROMPT = """\
Score the candidate's answer using the grading plan and rubric below.

Question: {actual_question}
Candidate answer: {user_response}
{followup_block}
Rubric criteria: {criteria}
Criterion weightings: {weightings}

Grading plan:
{plan}

Return JSON matching the EvaluationResult schema:
- score: final 0-10 score from the weighted rubric
- is_correct: whether the answer is correct overall
- feedback: specific, actionable feedback grounded in the rubric
- strengths: list of concrete strengths
- weaknesses: list of concrete weaknesses
- followup_evaluation: if a follow-up answer is present above, a short note on
  what it added or failed to add; otherwise null
- suggested_answer: a model answer to the question, written as the candidate
  should have said it out loud (first person, plain prose, no markdown, roughly
  120-250 words, calibrated to the difficulty). Keep what the candidate got
  right and apply every point in feedback and weaknesses - e.g. if a weakness
  is missing edge cases, the suggested answer covers them. If a follow-up was
  asked, fold its answer in so it is no longer needed.
Leave communication_evaluation and overall_score null.
"""
