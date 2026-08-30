"""Prompt for ``validate_node`` - checks the EvaluationResult against the plan."""

VALIDATE_PROMPT = """\
You are auditing an automated evaluation. Decide whether the scoring below is
sound enough to keep, or whether it is broken and must be redone. You are NOT
re-grading the candidate and you are NOT nitpicking style.

Question: {actual_question}
Candidate answer: {user_response}
{followup_block}
Rubric criteria: {criteria}
Criterion weightings: {weightings}

Grading plan the evaluator was told to follow:
{plan}

The evaluation that was produced:
- score: {score}
- is_correct: {is_correct}
- feedback: {feedback}
- strengths: {strengths}
- weaknesses: {weaknesses}
- followup_evaluation: {followup_evaluation}

The evaluator was NOT required to output per-criterion sub-scores or to show its
arithmetic. Do not fault it for that. Judge only whether the result is
defensible.

Flag a REAL problem only when:
- the feedback/strengths/weaknesses make claims that are not supported by the
  candidate's answer (fabrication);
- the numeric score is clearly inconsistent with the feedback (e.g. feedback is
  overwhelmingly negative but score is 8+, or vice versa);
- the score is not a plausible weighted reflection of the criteria given the
  weightings;
- is_correct plainly contradicts the score or the feedback;
- the plan named a hard requirement that was clearly not applied.

Return JSON matching the ReviewResult schema:
- is_grounded: bool - claims are supported by the answer (true if no fabrication)
- is_rubric_consistent: bool - the score is a plausible fit to the plan and
  weightings (it does NOT need to be exactly reconstructable)
- issues_found: list of concrete problems (empty if none)
- needs_regeneration: bool - true ONLY if a problem above is serious enough that
  keeping this evaluation would be wrong; minor wording or presentation gripes
  are NOT grounds for regeneration
"""
