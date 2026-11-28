from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class EvalStatus(str, Enum):
    pending_followup = "pending_followup"
    completed = "completed"


class Criterion(BaseModel):
    name: str
    weight: float = Field(..., ge=0, le=1, description="Share of the score; a rubric's weights add up to 1")
    looks_for: str = Field(..., description="What a strong answer shows on this criterion at this difficulty")


class Rubric(BaseModel):
    """One domain's rubric at one difficulty (see ``src/rubrics.py``)."""

    topic: str
    difficulty: Difficulty
    criteria: list[Criterion]
    expectations: str = Field(..., description="What a strong answer looks like overall at this difficulty")
    forgiven: list[str] = Field(default_factory=list, description="Gaps this difficulty does not penalise")
    not_forgiven: list[str] = Field(default_factory=list, description="Gaps this difficulty always penalises")
    answer_words: str = Field(..., description="Length of the suggested answer, e.g. '60-120'")
    key_points: int = Field(..., description="How many key points the suggested answer is summarised into")
    answer_style: str = Field(..., description="How deep and how technical the suggested answer should be")
    source: Optional[str] = None

    def render(self) -> str:
        """The rubric as prompt text, so every node scores against the same level."""
        lines = [f"Difficulty: {self.difficulty.value}", f"What this level expects: {self.expectations}", "Criteria:"]
        lines += [f"- {c.name} ({round(c.weight * 100)}%): {c.looks_for}" for c in self.criteria]
        lines.append(f"Do not penalise at this level: {'; '.join(self.forgiven)}.")
        lines.append(f"Always penalise: {'; '.join(self.not_forgiven)}.")
        return "\n".join(lines)


class EvaluationPlan(BaseModel):
    rubric: Rubric
    plan: str = Field(..., description="Instructions for how to evaluate the user's response based on the rubric")


class FollowupDecision(BaseModel):
    should_ask_followup: bool = Field(
        ..., description="True if user_response is incomplete/ambiguous and needs probing before scoring"
    )
    reason: str
    suggested_followup_question: Optional[str] = None


class CommunicationEvaluation(BaseModel):
    grammar_score: float = Field(..., ge=0, le=10, description="Score for grammatical correctness")
    grammar_issues: list[str] = Field(
        default_factory=list, description="Specific grammar mistakes identified"
    )

    pronunciation_score: float = Field(
        ..., ge=0, le=10, description="Score for pronunciation clarity/accuracy"
    )
    mispronounced_words: list[str] = Field(
        default_factory=list, description="Words that were mispronounced, if detectable"
    )

    fluency_score: float = Field(..., ge=0, le=10, description="Overall fluency of the speech")

    filler_words_used: list[str] = Field(
        default_factory=list, description="Distinct filler words detected"
    )

    pause_count: int = Field(default=0, ge=0, description="Number of noticeable pauses")

    clarity_score: float = Field(
        ..., ge=0, le=10, description="Overall clarity/articulation score"
    )

    communication_feedback: str = Field(
        ..., description="Summarized feedback on communication quality"
    )


class AudioAnalysis(BaseModel):
    """Structured reply from the Gemini call on a spoken answer."""

    transcript: str = Field(..., description="Verbatim transcript of the spoken answer")
    communication: CommunicationEvaluation


class FollowupTurn(BaseModel):
    """One follow-up the candidate was asked, and their answer (None if skipped)."""

    question: str
    answer: Optional[str] = None


class FollowupSuggestion(BaseModel):
    question: str = Field(..., description="The follow-up question, copied as asked")
    suggested_answer: str = Field(..., description="A short model answer to this follow-up, spoken in first person")


def render_followups(followups: list[FollowupTurn]) -> str:
    """Every follow-up asked so far, for prompts. Empty string when there were none."""
    if not followups:
        return ""
    lines = ["Follow-ups asked, in order:"]
    for i, f in enumerate(followups, 1):
        lines.append(f"  {i}. Q: {f.question}")
        lines.append(f"     A: {f.answer}" if f.answer else "     A: (skipped - the candidate did not answer)")
    lines.append("Judge the whole exchange: the main answer plus every follow-up answer. A skipped follow-up shows nothing extra.")
    return "\n".join(lines) + "\n"


class EvaluationResult(BaseModel):
    score: float = Field(..., ge=0, le=10, description="Score out of 10 for the response")
    is_correct: bool = Field(..., description="Whether the response is considered correct")
    feedback: str = Field(..., description="Detailed feedback on the user's response")
    strengths: list[str] = Field(default_factory=list, description="Strengths identified in the response")
    weaknesses: list[str] = Field(default_factory=list, description="Weaknesses identified in the response")
    evaluation_plan_used: Optional[EvaluationPlan] = None
    followup_evaluation: Optional[str] = Field(
        default=None, description="Evaluation specifically for the followup answer, if provided"
    )
    # Required (no default) so the structured output must include them; when they
    # were optional the model sometimes left them out, mostly on runs with a
    # follow-up, and the app showed no suggested answer.
    suggested_answer: str = Field(
        ...,
        description="A model answer to the question that applies the feedback and fixes the weaknesses above",
    )
    suggested_points: list[str] = Field(
        ...,
        description="3-5 short key points a strong answer covers (the suggested answer in brief)",
    )
    # Required for the same reason; an empty list when no follow-up was asked.
    followup_suggestions: list[FollowupSuggestion] = Field(
        ...,
        description="One short model answer per follow-up listed in the prompt, in the same order; empty if none",
    )
    communication_evaluation: Optional[CommunicationEvaluation] = Field(
        default=None, description="Evaluation of communication skills (grammar, pronunciation, filler words, pauses, etc.)"
    )
    overall_score: Optional[float] = Field(
        default=None, ge=0, le=10, description="Combined score factoring in both content and communication"
    )


class ReviewResult(BaseModel):
    is_grounded: bool
    is_rubric_consistent: bool
    issues_found: list[str] = Field(default_factory=list)
    needs_regeneration: bool = False


class UserInput(BaseModel):
    actual_question: str = Field(..., description="The original question asked to the user")
    user_response: Optional[str] = Field(
        default=None,
        description="The user's typed response; None when the answer was given by voice (see audio_path)",
    )
    audio_path: Optional[str] = Field(
        default=None,
        description="Local path to the candidate's recorded answer (wav/mp3/m4a/ogg/webm/flac)",
    )
    difficulty: Difficulty = Field(..., description="Difficulty level of the question")
    domain: str = Field(..., description="Domain selected in the UI, e.g. 'ai-ml', 'backend', 'behavioral' (see src/rubrics.py)")

    @model_validator(mode="after")
    def _require_an_answer(self):
        if not self.user_response and not self.audio_path:
            raise ValueError("provide user_response or audio_path")
        return self


class AgentState(BaseModel):
    # inputs
    input: UserInput
    user_id: Optional[str] = Field(
        default=None,
        description="Opaque caller-provided user identifier (e.g. the LMS account id), attached to Langfuse traces",
    )

    # transcribe stage (voice answers only; no-op for typed answers)
    transcript: Optional[str] = Field(
        default=None, description="Transcript of the spoken answer, once produced"
    )
    communication_evaluation: Optional[CommunicationEvaluation] = Field(
        default=None, description="Delivery assessment from the audio (grammar, pronunciation, fluency, ...)"
    )

    # rubric (free static lookup; fetched by followup_gate, reused by plan)
    rubric: Optional[Rubric] = None

    # plan stage
    plan: Optional[EvaluationPlan] = None

    # followup-gate stage (runs BEFORE scoring)
    followup_decision: Optional[FollowupDecision] = None
    followup_question: Optional[str] = Field(
        default=None, description="Followup question generated by the agent, if the response needed probing"
    )
    followup_answer: Optional[str] = Field(
        default=None, description="The user's answer to the followup question, once provided"
    )
    followups: list[FollowupTurn] = Field(
        default_factory=list, description="Every follow-up asked so far and its answer (None if skipped)"
    )
    awaiting_followup_answer: bool = Field(
        default=False, description="True when a followup was asked but not yet answered"
    )

    # execute stage (only runs once followup gate clears)
    evaluation_result: Optional[EvaluationResult] = Field(
        default=None, description="The result of evaluating the user's response(s)"
    )

    # review stage
    review_result: Optional[ReviewResult] = None

    # control flow
    iteration: int = Field(default=0, description="Number of followup_decision <-> execute loops so far")
    max_iterations: int = Field(default=2)

    # validate stage (runs AFTER scoring; loops back to execute when the result
    # is not grounded in / consistent with the plan)
    validate_iteration: int = Field(default=0, description="Number of validate runs so far")
    max_validate_iterations: int = Field(default=2)

    is_final: bool = Field(default=False)

    @model_validator(mode="after")
    def guard_scoring_before_followup_resolved(self):
        if self.evaluation_result is not None and self.awaiting_followup_answer:
            raise ValueError(
                "evaluation_result cannot be set while awaiting_followup_answer is True - "
                "resolve the followup before scoring"
            )
        return self


class SessionSummary(BaseModel):
    """Structured reply for the end-of-session summary."""

    summary: str
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    clarity_level: float
    clarity_label: str
    confidence_level: float
    confidence_label: str
