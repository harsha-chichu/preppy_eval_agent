import sys

from logger import logger
from src.agent.state import AgentState, AudioAnalysis, Difficulty, UserInput
from src.config.client import call_gemini_audio
from src.config.tracing import traced_node
from src.prompts import AUDIO_PROMPT

sys.stdout.reconfigure(encoding="utf-8")


def _analyse(state: AgentState) -> AudioAnalysis:
    return call_gemini_audio(
        AUDIO_PROMPT.format(actual_question=state.input.actual_question),
        state.input.audio_path,
        AudioAnalysis,
    )


@traced_node("transcribe-answer", when=lambda state: bool(state.input.audio_path))
def transcribe_node(state: AgentState) -> dict:
    """Transcribe + assess a spoken answer that arrived WITHOUT text.

    Runs before the follow-up gate, because the gate needs the words. Gemini's
    transcript is written into ``input.user_response`` so every downstream node
    scores the content exactly as it would a typed answer. The delivery
    assessment from the same call is kept, so ``assess_delivery`` skips later.

    When the caller sent reviewed text with the recording, the graph routes
    around this node and ``assess_delivery_node`` handles the audio instead.
    """
    if not state.input.audio_path:
        logger.info("transcribe_node: no audio, skipping.")
        return {}

    logger.info("Running transcribe_node on %s ...", state.input.audio_path)
    analysis = _analyse(state)
    logger.info("transcribe_node completed (%d chars transcribed).", len(analysis.transcript))
    if state.input.user_response:
        # The user's reviewed text is the answer; the audio was for delivery only.
        return {"transcript": state.input.user_response, "communication_evaluation": analysis.communication}
    updated_input = state.input.model_copy(update={"user_response": analysis.transcript})
    return {
        "input": updated_input,
        "transcript": analysis.transcript,
        "communication_evaluation": analysis.communication,
    }


def _needs_delivery(state: AgentState) -> bool:
    return bool(state.input.audio_path) and state.communication_evaluation is None


@traced_node("assess-delivery", when=_needs_delivery)
def assess_delivery_node(state: AgentState) -> dict:
    """Delivery assessment for a spoken answer whose text the user already sent.

    Runs AFTER the follow-up gate, in parallel with ``plan``, so the slow
    audio call is off the path the user waits on. Delivery scores are an
    extra: if Gemini fails here, content scoring still goes ahead without them.
    """
    if not _needs_delivery(state):
        return {}
    logger.info("Running assess_delivery_node on %s ...", state.input.audio_path)
    try:
        analysis = _analyse(state)
    except Exception:
        logger.exception("assess_delivery_node failed; scoring without delivery scores.")
        return {}
    logger.info("assess_delivery_node completed.")
    return {"communication_evaluation": analysis.communication}


if __name__ == "__main__":
    import sys as _sys

    path = _sys.argv[1] if len(_sys.argv) > 1 else "sample.wav"
    sample_input = UserInput(
        actual_question="what is backpropagation?",
        audio_path=path,
        difficulty=Difficulty.easy,
        domain="ai",
    )
    result = transcribe_node(AgentState(input=sample_input))
    print(result["communication_evaluation"].model_dump_json(indent=2))
    print("\nTranscript:\n", result["transcript"])
