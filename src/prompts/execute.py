"""Prompt for ``execute_node`` - produces the final EvaluationResult."""

EXECUTE_PROMPT = """\
Score the candidate's answer to a {difficulty} interview question using the
rubric and grading plan below.

Question: {actual_question}
Candidate answer: {user_response}
{followup_block}
Rubric for this difficulty:
{rubric}

Grading plan:
{plan}

Score against what THIS difficulty expects, not against a perfect expert
answer. An answer that fully meets the level's expectations scores 8 or more,
even if it lacks depth a harder level would ask for. Do not take marks off for
anything the rubric says not to penalise at this level, and do not ask for
depth beyond the level in feedback or weaknesses.

Return JSON matching the EvaluationResult schema:
- score: final 0-10 score from the weighted rubric
- is_correct: whether the answer is correct overall
- feedback: specific, actionable feedback grounded in the rubric
- strengths: list of concrete strengths
- weaknesses: list of concrete weaknesses, judged at this difficulty
- followup_evaluation: if follow-ups are listed above, a short note on what the
  answers added or failed to add (a skipped follow-up added nothing); otherwise null
- suggested_answer: a model answer to the question at {difficulty} difficulty,
  written as the candidate should have said it out loud (first person, plain
  prose, no markdown, {answer_words} words). Style: {answer_style}. Keep what
  the candidate got right and apply every point in feedback and weaknesses -
  e.g. if a weakness is missing edge cases, the suggested answer covers them.
  If follow-ups were asked, fold in what they probed so they are no longer needed.
- suggested_points: the suggested answer as exactly {key_points} short key
  points (under 12 words each) the candidate can remember before the real
  interview.
- followup_suggestions: one entry per follow-up listed above, in the same order
  (an empty list if there were none). For each, copy the question and write
  suggested_answer: how the candidate should have answered THAT follow-up out
  loud - first person, plain prose, no markdown, 40-80 words, at {difficulty}
  difficulty. Answer it directly; keep what they got right; if it was skipped,
  write the answer they should have given.
Leave communication_evaluation and overall_score null.
"""
