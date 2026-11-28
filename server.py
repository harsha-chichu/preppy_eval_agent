"""HTTP wrapper around the evaluation agent, used by the LMS web app.

Run:  uv run uvicorn server:app --host 127.0.0.1 --port 8000

The graph runs in a background thread per evaluation. Each call returns as soon
as the follow-up gate has decided - either with the follow-up question, or with
``status: "scoring"`` while plan -> execute -> validate keep running. The caller
polls ``GET /evaluations/{id}`` for the final result.

Every request must carry ``x-agent-token: $EVAL_AGENT_TOKEN``. The optional
``user_id`` form field (sent by the web app) is attached to every Langfuse
trace of the evaluation so per-user analytics work in the dashboard.
State lives in the agent's in-memory checkpointer, so restarting this server
drops in-flight evaluations (the web app falls back to its own scorer).
"""

import asyncio
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from langgraph.types import Command

from logger import logger
from src.agent.graph import _GRAPH
from src.agent.state import AgentState, Difficulty, FollowupTurn, UserInput
from src.config import constants as _constants  # noqa: F401  (loads .env)
from src.rubrics import fetch_rubric
from src.summary import SummaryRequest, summarize
from src.config.tracing import record_outcome, resume_input, score_intermediate, start_input, trace_run

TOKEN = os.getenv("EVAL_AGENT_TOKEN")
if not TOKEN:
    raise RuntimeError("EVAL_AGENT_TOKEN is not set")

# The decision is one gate LLM call (plus Gemini transcription for a recording
# sent without text). Return before the web app gives up (45s) so a slow call
# degrades to "scoring" instead of a failed request.
DECISION_TIMEOUT_S = float(os.getenv("EVAL_DECISION_TIMEOUT_S", "40"))
RUN_TTL_S = 60 * 60
MAX_AUDIO_BYTES = 15 * 1024 * 1024
# user_id is trace metadata, not identity data: bound it so a malformed client
# cannot push arbitrary payloads into Langfuse. 200 matches the SDK limit -
# propagate_attributes silently drops longer user_id values.
USER_ID_MAX_LEN = 200

_AUDIO_SUFFIX = {
    "audio/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
}


@dataclass
class Run:
    # running -> pending_followup | scoring -> completed | failed
    status: str = "running"
    user_id: Optional[str] = None
    followup_question: Optional[str] = None
    transcript: Optional[str] = None
    communication: Optional[dict] = None
    result: Optional[dict] = None
    review: Optional[dict] = None
    error: Optional[str] = None
    decided: threading.Event = field(default_factory=threading.Event)
    # Audio files to delete once the run is finished. They outlive a follow-up
    # pause: the delivery assessment runs after the gate, in a later segment.
    files: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def snapshot(self, thread_id: str) -> dict:
        return {
            "thread_id": thread_id,
            "status": self.status,
            "followup_question": self.followup_question,
            "transcript": self.transcript,
            "communication": self.communication,
            "result": self.result,
            "error": self.error,
        }


_RUNS: dict[str, Run] = {}
_LOCK = threading.Lock()


def _remove_files(paths: list[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass
    paths.clear()


def _prune() -> None:
    cutoff = time.time() - RUN_TTL_S
    with _LOCK:
        stale = [(tid, _RUNS.pop(tid)) for tid, run in list(_RUNS.items()) if run.updated_at < cutoff]
    for tid, run in stale:
        _remove_files(run.files)
        try:
            _GRAPH.checkpointer.delete_thread(tid)
        except Exception:  # best effort
            pass


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _clean_user_id(value: Optional[str]) -> Optional[str]:
    """Normalize the optional ``user_id`` form field sent by the web app."""
    if not value:
        return None
    value = value.strip()
    return value[:USER_ID_MAX_LEN] or None


def _drive(
    thread_id: str, run: Run, graph_input: Any, user_input: UserInput, segment: str, run_input: dict
) -> None:
    """Stream the graph until it pauses for a follow-up or finishes."""
    with trace_run(thread_id, user_input, segment, run_input, user_id=run.user_id) as root:
        _stream(thread_id, run, graph_input, root)
        record_outcome(
            root,
            evaluation_result=run.result,
            review_result=run.review,
            followup_question=run.followup_question if run.status == "pending_followup" else None,
        )


def _stream(thread_id: str, run: Run, graph_input: Any, root: Any = None) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        for update in _GRAPH.stream(graph_input, config, stream_mode="updates"):
            run.updated_at = time.time()
            if "__interrupt__" in update:
                run.followup_question = update["__interrupt__"][0].value["followup_question"]
                run.status = "pending_followup"
                run.decided.set()
                continue
            for node, values in update.items():
                values = values or {}
                if node == "transcribe" and values.get("transcript"):
                    run.transcript = values["transcript"]
                    run.communication = _dump(values.get("communication_evaluation"))
                elif node == "assess_delivery" and values.get("communication_evaluation"):
                    run.communication = _dump(values["communication_evaluation"])
                elif node == "ask_followup" and values.get("followup_answer"):
                    # Spoken follow-ups are transcribed inside ask_followup.
                    run.transcript = values["followup_answer"]
                elif node == "followup_gate" and not values.get("awaiting_followup_answer"):
                    run.status = "scoring"
                    run.followup_question = None
                    run.decided.set()
                elif node == "execute" and values.get("evaluation_result") is not None:
                    # validate may send it back to execute; the last one wins.
                    result = _dump(values["evaluation_result"])
                    result.pop("evaluation_plan_used", None)
                    run.result = result
                elif node == "validate" and values.get("review_result") is not None:
                    run.review = _dump(values["review_result"])
                    # Record each validate-loop attempt so retries are visible
                    # in Langfuse Scores (e.g. grounded@1, rubric-consistent@2).
                    if root is not None:
                        iteration = values.get("validate_iteration", 1)
                        score_intermediate(root, run.review, iteration)

        if run.status != "pending_followup":
            run.status = "completed" if run.result else "failed"
            if not run.result:
                run.error = "graph finished without an evaluation result"
    except Exception as exc:  # surface to the caller instead of hanging
        logger.exception("eval run %s failed", thread_id)
        run.status = "failed"
        run.error = str(exc)
    finally:
        run.updated_at = time.time()
        run.decided.set()
        if run.status != "pending_followup":
            _remove_files(run.files)


async def _start_segment(
    thread_id: str, run: Run, graph_input: Any, segment: str, run_input: Optional[dict] = None
) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    if isinstance(graph_input, AgentState):
        user_input = graph_input.input
    else:
        user_input = _GRAPH.get_state(config).values["input"]
    run_input = run_input or start_input(user_input)
    run.decided = threading.Event()
    run.status = "running"
    worker = threading.Thread(
        target=_drive, args=(thread_id, run, graph_input, user_input, segment, run_input), daemon=True
    )
    worker.start()
    # Wait in a worker thread: a blocking wait here would freeze the event loop,
    # so every other request (other users' answers, score polls) queued behind it.
    await asyncio.to_thread(run.decided.wait, DECISION_TIMEOUT_S)
    return run.snapshot(thread_id)


VOICE_ENABLED = bool(_constants.GEMINI_API_KEY)
VOICE_UNAVAILABLE = "voice answers are unavailable, please type your answer"


async def _save_audio(audio: Optional[UploadFile], answer: Optional[str] = None) -> Optional[str]:
    if audio is None:
        return None
    if not VOICE_ENABLED:
        # Audio needs Gemini. With typed text, score the text and drop the
        # recording; with a recording alone there is nothing we can score.
        if answer and answer.strip():
            logger.info("Voice disabled (no GEMINI_API_KEY): ignoring recording, using typed answer.")
            return None
        raise HTTPException(503, VOICE_UNAVAILABLE)
    data = await audio.read()
    if not data:
        return None
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio too large")
    content_type = (audio.content_type or "").split(";")[0].strip().lower()
    suffix = _AUDIO_SUFFIX.get(content_type) or os.path.splitext(audio.filename or "")[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _require_token(x_agent_token: str = Header(default="")) -> None:
    if x_agent_token != TOKEN:
        raise HTTPException(401, "invalid agent token")


def _get_run(thread_id: str) -> Run:
    with _LOCK:
        run = _RUNS.get(thread_id)
    if run is None:
        raise HTTPException(404, "unknown evaluation")
    return run


app = FastAPI(title="eval-agent", dependencies=[Depends(_require_token)])


@app.on_event("shutdown")
def _flush_langfuse() -> None:
    """Drain the Langfuse event queue so in-flight traces are not lost."""
    from src.config.tracing import langfuse as _lf

    logger.info("Flushing Langfuse event queue...")
    _lf.flush()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "runs": len(_RUNS), "voice": VOICE_ENABLED}


@app.get("/rubrics")
def get_rubric(domain: str, difficulty: Difficulty) -> dict:
    """How answers in this domain are scored at this difficulty, for showing before a session."""
    return fetch_rubric(domain, difficulty).model_dump(mode="json")


@app.post("/evaluations")
async def create_evaluation(
    question: str = Form(...),
    difficulty: Difficulty = Form(...),
    domain: str = Form(...),
    answer: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    user_id: Optional[str] = Form(None),
) -> dict:
    _prune()
    audio_path = await _save_audio(audio, answer)
    if not audio_path and not (answer and answer.strip()):
        raise HTTPException(422, "provide answer or audio")

    user_input = UserInput(
        actual_question=question,
        # Text the user reviewed wins; with a recording alone its transcript becomes
        # the answer. Either way the recording is still used for delivery scores.
        user_response=answer.strip() if answer and answer.strip() else None,
        audio_path=audio_path,
        difficulty=difficulty,
        domain=domain,
    )
    thread_id = uuid4().hex
    run = Run(user_id=_clean_user_id(user_id), files=[audio_path] if audio_path else [])
    with _LOCK:
        _RUNS[thread_id] = run
    return await _start_segment(thread_id, run, AgentState(input=user_input, user_id=run.user_id), "start")


@app.post("/evaluations/{thread_id}/followup")
async def answer_followup(
    thread_id: str,
    answer: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    user_id: Optional[str] = Form(None),
) -> dict:
    run = _get_run(thread_id)
    if run.status != "pending_followup":
        raise HTTPException(409, f"evaluation is {run.status}, not waiting for a follow-up")
    # First writer wins: a run keeps the identity it was created with, so a
    # follow-up cannot re-attribute an in-flight evaluation to another user.
    run.user_id = run.user_id or _clean_user_id(user_id)
    audio_path = await _save_audio(audio, answer)
    if not audio_path and not (answer and answer.strip()):
        raise HTTPException(422, "provide answer or audio")
    text = answer.strip() if answer and answer.strip() else None
    payload = {"audio_path": audio_path, "answer": text} if audio_path else text
    if audio_path:
        run.files.append(audio_path)
    run_input = resume_input(run.followup_question, text, audio_path)
    return await _start_segment(thread_id, run, Command(resume=payload), "followup", run_input)


@app.post("/evaluations/{thread_id}/skip-followup")
async def skip_followup(thread_id: str) -> dict:
    run = _get_run(thread_id)
    if run.status != "pending_followup":
        raise HTTPException(409, f"evaluation is {run.status}, not waiting for a follow-up")
    config = {"configurable": {"thread_id": thread_id}}
    values = _GRAPH.get_state(config).values
    max_iterations = values.get("max_iterations", 2) if isinstance(values, dict) else 2
    followups = list(values.get("followups") or []) if isinstance(values, dict) else []
    run_input = resume_input(run.followup_question, "(skipped)", None)
    # Close the pending follow-up as skipped (so scoring sees it was asked but not
    # answered) and exhaust the follow-up budget, so the gate proceeds to scoring.
    _GRAPH.update_state(
        config,
        {
            "awaiting_followup_answer": False,
            "iteration": max_iterations,
            "followups": [*followups, FollowupTurn(question=run.followup_question or "", answer=None)],
            "followup_question": None,
            "followup_answer": None,
        },
        as_node="ask_followup",
    )
    return await _start_segment(thread_id, run, None, "skip-followup", run_input)


@app.get("/evaluations/{thread_id}")
def get_evaluation(thread_id: str) -> dict:
    return _get_run(thread_id).snapshot(thread_id)


@app.post("/summaries")
def create_summary(req: SummaryRequest) -> dict:
    """End-of-session summary: one LLM call, returned directly (no polling)."""
    try:
        return summarize(req)
    except Exception as exc:
        logger.exception("summary failed")
        raise HTTPException(502, f"summary failed: {exc}") from exc
