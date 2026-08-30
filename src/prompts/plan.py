
PLAN_PROMPT = """\
You are designing the grading instructions an evaluator will follow to score a
candidate's answer to an interview-style question.

Question: {actual_question}
Domain: {domain}
Difficulty: {difficulty}

The rubric is FIXED. Do not change it, do not add or drop criteria. Write only
the instructions for applying it.

Rubric criteria: {criteria}
Criterion weightings: {weightings}

Produce a concise plan (plain prose, no JSON) that tells the evaluator:
- what specific evidence in the answer satisfies each criterion, and what a weak
  vs strong response looks like for it;
- how to turn that evidence into a 0-10 score per criterion and combine the
  weighted criteria into the final 0-10 score;
- how the "{difficulty}" difficulty calibrates expectations (what can be
  excused at this level, what still cannot);
- what would make the answer "correct" overall vs not.

Keep it under ~200 words. Return the instructions only.
"""
