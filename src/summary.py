"""End-of-session summary: one LLM call over all of a session's answers."""

from typing import Optional

from pydantic import BaseModel, Field

from logger import logger
from src.agent.state import SessionSummary
from src.config.client import call_structured_llm
from src.prompts import SUMMARY_PROMPT

MAX_QUESTIONS = 10
MAX_TEXT = 20_000


class SummaryTurn(BaseModel):
    text: str = Field(max_length=MAX_TEXT)
    mode: str = "text"


class SummaryFollowUp(BaseModel):
    question: str = Field(max_length=MAX_TEXT)
    answer: Optional[SummaryTurn] = None


class SummaryQuestion(BaseModel):
    question: str = Field(max_length=MAX_TEXT)
    skipped: bool = False
    answer: Optional[SummaryTurn] = None
    followups: list[SummaryFollowUp] = Field(default_factory=list, max_length=5)
    score: Optional[float] = None
    feedback: Optional[str] = Field(default=None, max_length=MAX_TEXT)


class SummaryMetrics(BaseModel):
    wpm: Optional[float] = None
    fillers_total: int = 0
    fillers_per_min: Optional[float] = None
    hedges: int = 0


class SummaryRequest(BaseModel):
    domain: str = Field(max_length=200)
    difficulty: str = Field(max_length=20)
    questions: list[SummaryQuestion] = Field(min_length=1, max_length=MAX_QUESTIONS)
    metrics: SummaryMetrics = Field(default_factory=SummaryMetrics)


def _transcript(req: SummaryRequest) -> str:
    blocks = []
    for i, q in enumerate(req.questions, 1):
        lines = [f"q{i}: {q.question}"]
        if q.skipped:
            lines.append("(skipped)")
        if q.answer:
            lines.append(f"answer ({q.answer.mode}): {q.answer.text}")
        for f in q.followups:
            lines.append(f"follow-up: {f.question}")
            lines.append(f"follow-up answer: {f.answer.text}" if f.answer else "(follow-up skipped)")
        if q.score is not None:
            lines.append(f"score: {q.score}/10 - {q.feedback or ''}".rstrip(" -"))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _delivery(m: SummaryMetrics) -> str:
    lines = [f"speaking pace: {m.wpm} wpm" if m.wpm is not None else "all answers were typed"]
    if m.fillers_per_min is not None:
        lines.append(f"filler words: {m.fillers_total} ({m.fillers_per_min}/min)")
    lines.append(f"hedging phrases: {m.hedges}")
    return "\n".join(lines)


def _level(n: float) -> int:
    return int(round(min(5, max(1, n))))


def summarize(req: SummaryRequest) -> dict:
    prompt = SUMMARY_PROMPT.format(
        difficulty=req.difficulty,
        domain=req.domain,
        transcript=_transcript(req),
        delivery=_delivery(req.metrics),
    )
    out = call_structured_llm(prompt, SessionSummary, name="write-session-summary", temperature=0.4, max_tokens=1500)
    if not out.summary.strip():
        raise ValueError("model returned an empty summary")
    logger.info("summary written (%d chars).", len(out.summary))
    return {
        "summary": out.summary.strip(),
        "strengths": [s.strip() for s in out.strengths if s.strip()][:3],
        "improvements": [s.strip() for s in out.improvements if s.strip()][:3],
        "clarity": {"level": _level(out.clarity_level), "label": out.clarity_label.strip()},
        "confidence": {"level": _level(out.confidence_level), "label": out.confidence_label.strip()},
    }
