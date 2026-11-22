"""Prompt for ``plan_node`` - writes the grading instructions for the rubric."""

PLAN_PROMPT = """\
You are designing the grading instructions an evaluator will follow to score a
candidate's answer to an interview-style question.

Question: {actual_question}
Domain: {domain}
Difficulty: {difficulty}

The rubric is FIXED and already tuned for this difficulty. Do not change it, do
not add or drop criteria, do not change the weights. Write only the
instructions for applying it to this question.

Rubric:
{rubric}

Produce a concise plan (plain prose, no JSON) that tells the evaluator:
- what specific evidence in an answer to THIS question satisfies each
  criterion at the "{difficulty}" level, and what a weak vs strong response
  looks like for it;
- how to turn that evidence into a 0-10 score per criterion and combine the
  weighted criteria into the final 0-10 score;
- what can be excused at this level and what still cannot (follow the rubric);
- what would make the answer "correct" overall vs not.

Keep it under ~200 words. Return the instructions only.
"""
