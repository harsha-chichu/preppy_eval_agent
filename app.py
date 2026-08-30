"""Streamlit UI for the evaluation agent.

Run:  uv run streamlit run app.py

Shows the question, takes your answer, lets the agent decide whether to ask a
follow-up (answered in the UI), then runs the evaluation and displays the result.
"""

import streamlit as st

from src.agent.graph import resume, start
from src.agent.state import Difficulty, EvalStatus, UserInput

# The question under test - same as main.py.
QUESTION = "Explain how a transformer handles long-range dependencies."
DIFFICULTY = Difficulty.hard
DOMAIN = "ai"

st.set_page_config(page_title="Eval Agent", page_icon="🧪")


def _reset() -> None:
    for key in list(st.session_state.keys()):
        if key in ("phase", "thread_id", "result", "followups", "answer") or key.startswith("fa_"):
            st.session_state.pop(key, None)


def _run(fn, *args) -> dict:
    """Invoke a graph call with a spinner and stash the outcome in session state."""
    with st.spinner("The agent is thinking… (a few LLM calls, ~10-40s)"):
        res = fn(*args)
    st.session_state.thread_id = res["thread_id"]
    st.session_state.result = res
    st.session_state.phase = (
        "followup" if res["status"] == EvalStatus.pending_followup else "done"
    )
    return res


st.session_state.setdefault("phase", "answer")
st.session_state.setdefault("followups", [])

st.title("🧪 Evaluation Agent")

st.subheader("Question")
st.info(QUESTION)
st.caption(f"difficulty: **{DIFFICULTY.value}**  ·  domain: **{DOMAIN}**")

phase = st.session_state.phase

# ── 1. initial answer ────────────────────────────────────────────────────────
if phase == "answer":
    answer = st.text_area("Your answer", height=160, key="answer")
    if st.button("Submit answer", type="primary", disabled=not answer.strip()):
        _run(
            start,
            UserInput(
                actual_question=QUESTION,
                user_response=answer.strip(),
                difficulty=DIFFICULTY,
                domain=DOMAIN,
            ),
        )
        st.rerun()

# ── 2. follow-up loop ────────────────────────────────────────────────────────
elif phase == "followup":
    for i, (q, a) in enumerate(st.session_state.followups, 1):
        st.markdown(f"**Follow-up {i}:** {q}")
        st.markdown(f"> {a}")

    question = st.session_state.result["followup_question"]
    st.warning(f"**The agent needs a follow-up:**\n\n{question}")
    fa = st.text_area(
        "Your follow-up answer", height=120, key=f"fa_{len(st.session_state.followups)}"
    )
    if st.button("Submit follow-up", type="primary", disabled=not fa.strip()):
        st.session_state.followups.append((question, fa.strip()))
        _run(resume, st.session_state.thread_id, fa.strip())
        st.rerun()

# ── 3. result ────────────────────────────────────────────────────────────────
elif phase == "done":
    res = st.session_state.result
    result = res["evaluation_result"]
    review = res.get("review_result")

    if st.session_state.followups:
        with st.expander("Follow-up exchange", expanded=False):
            for i, (q, a) in enumerate(st.session_state.followups, 1):
                st.markdown(f"**Q{i}:** {q}")
                st.markdown(f"**A{i}:** {a}")

    st.subheader("Result")
    c1, c2 = st.columns([1, 2])
    c1.metric("Score", f"{result.score:.1f} / 10")
    (c2.success if result.is_correct else c2.error)(
        "Correct" if result.is_correct else "Not correct"
    )

    st.markdown("**Feedback**")
    st.write(result.feedback)

    if result.strengths:
        st.markdown("**Strengths**")
        for s in result.strengths:
            st.markdown(f"- {s}")
    if result.weaknesses:
        st.markdown("**Weaknesses**")
        for w in result.weaknesses:
            st.markdown(f"- {w}")
    if result.followup_evaluation:
        st.markdown("**Follow-up evaluation**")
        st.write(result.followup_evaluation)

    if review is not None:
        with st.expander("Validation (self-check of the evaluation)", expanded=False):
            st.write(
                {
                    "is_grounded": review.is_grounded,
                    "is_rubric_consistent": review.is_rubric_consistent,
                    "needs_regeneration": review.needs_regeneration,
                }
            )
            if review.issues_found:
                st.markdown("**Issues found**")
                for issue in review.issues_found:
                    st.markdown(f"- {issue}")

    st.button("Start over", on_click=_reset)
