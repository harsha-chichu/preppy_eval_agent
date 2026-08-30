FOLLOWUP_PROMPT = """\
You are deciding whether the candidate's answer needs ONE clarifying follow-up
question before it can be scored fairly.

Question: {actual_question}
Candidate answer: {user_response}
{prior_followup}
Grading plan the evaluator will use:
{plan}

Ask for a follow-up ONLY when the answer is too incomplete or ambiguous to score
without guessing (e.g. it trails off, is a non-answer, or could mean several very
different things). A merely WRONG or shallow answer is still scorable - do not
ask in that case.

Return JSON matching the FollowupDecision schema:
- should_ask_followup: bool
- reason: short justification
- suggested_followup_question: one concrete question, or null if not asking
"""
