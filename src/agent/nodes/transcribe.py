import sys

from logger import logger
from src.agent.state import AgentState, AudioAnalysis, Difficulty, UserInput
from src.config.client import call_gemini_audio
from src.config.tracing import traced_node
from src.prompts import AUDIO_PROMPT

sys.stdout.reconfigure(encoding="utf-8")


@traced_node("transcribe-answer", when=lambda state: bool(state.input.audio_path))
def transcribe_node(state: AgentState) -> dict:
    """Transcribe + assess a spoken answer.

    No-op for typed answers. For voice answers it makes one Gemini call that
    returns both the transcript and the delivery assessment. If the caller sent
    text with the recording (the user reviewed/edited the transcript), that text
    stays the answer and the audio is only used for the delivery assessment;
    otherwise Gemini's transcript is written into ``input.user_response`` so every
    downstream node scores the content exactly as it would a typed answer.
    """
    if not state.input.audio_path:
        logger.info("transcribe_node: no audio, skipping.")
        return {}

    logger.info("Running transcribe_node on %s ...", state.input.audio_path)
    analysis = call_gemini_audio(
        AUDIO_PROMPT.format(actual_question=state.input.actual_question),
        state.input.audio_path,
        AudioAnalysis,
    )

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
