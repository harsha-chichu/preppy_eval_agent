FOLLOWUP_PROMPT = """\
You are deciding whether the candidate's answer needs ONE clarifying follow-up
question before it can be scored fairly.

Question: {actual_question}
Candidate answer: {user_response}
{prior_followup}
The answer will be scored against this fixed rubric:
Criteria: {criteria}
Criterion weightings: {weightings}

Ask for a follow-up ONLY when the answer is too incomplete or ambiguous to score
against one of these criteria without guessing - for example it trails off, is a
non-answer, or could mean several very different things on a criterion that
carries real weight. A merely WRONG or shallow answer on a criterion is still
scorable - do not ask in that case. Prefer not asking when the missing piece
only affects a low-weight criterion.

Return JSON matching the FollowupDecision schema:
- should_ask_followup: bool
- reason: short justification
- suggested_followup_question: one concrete question, or null if not asking
"""
