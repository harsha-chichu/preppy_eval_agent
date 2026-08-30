"""Streamlit UI for the evaluation agent.

Run:  uv run streamlit run app.py

Shows the question, takes your answer, lets the agent decide whether to ask a
follow-up (answered in the UI), then runs the evaluation and displays the result.
"""

import tempfile

import streamlit as st

from src.agent.graph import resume, start
from src.agent.state import Difficulty, EvalStatus, UserInput

# The question under test - same as main.py.
QUESTION = "What is machine learning and how does backpropagation work?"
DIFFICULTY = Difficulty.medium
DOMAIN = "ai"

st.set_page_config(page_title="Eval Agent", page_icon="🧪")


def _reset() -> None:
    for key in list(st.session_state.keys()):
        if key in ("phase", "thread_id", "result", "followups", "answer", "audio") or key.startswith(("fa_", "faa_")):
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
    st.caption("…or answer out loud instead:")
    audio = st.audio_input("Record your answer", key="audio")

    has_answer = bool(answer.strip()) or audio is not None
    if st.button("Submit answer", type="primary", disabled=not has_answer):
        audio_path = None
        if audio is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio.getvalue())
            tmp.close()
            audio_path = tmp.name
        _run(
            start,
            UserInput(
                actual_question=QUESTION,
                # a recording takes precedence; its transcript becomes the answer
                user_response=None if audio_path else answer.strip(),
                audio_path=audio_path,
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
    n = len(st.session_state.followups)
    fa = st.text_area("Your follow-up answer", height=120, key=f"fa_{n}")
    st.caption("…or answer out loud instead:")
    fa_audio = st.audio_input("Record your follow-up answer", key=f"faa_{n}")

    has_reply = bool(fa.strip()) or fa_audio is not None
    if st.button("Submit follow-up", type="primary", disabled=not has_reply):
        if fa_audio is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(fa_audio.getvalue())
            tmp.close()
            st.session_state.followups.append((question, "🎙️ (spoken answer)"))
            _run(resume, st.session_state.thread_id, None, tmp.name)
        else:
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

    if result.communication_evaluation:
        ce = result.communication_evaluation
        st.divider()
        st.subheader("Communication (from your recording)")
        if result.overall_score is not None:
            st.metric("Overall score (80% content / 20% communication)", f"{result.overall_score:.1f} / 10")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Grammar", f"{ce.grammar_score:.1f}")
        g2.metric("Pronunciation", f"{ce.pronunciation_score:.1f}")
        g3.metric("Fluency", f"{ce.fluency_score:.1f}")
        g4.metric("Clarity", f"{ce.clarity_score:.1f}")
        st.write(ce.communication_feedback)
        if ce.filler_words_used:
            st.caption(f"Filler words: {', '.join(ce.filler_words_used)}")
        st.caption(f"Noticeable pauses: {ce.pause_count}")
        if ce.mispronounced_words:
            st.caption(f"Mispronounced: {', '.join(ce.mispronounced_words)}")
        if res.get("transcript"):
            with st.expander("Transcript", expanded=False):
                st.write(res["transcript"])

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
