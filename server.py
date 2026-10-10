"""HTTP wrapper around the evaluation agent, used by the LMS web app.

Run:  uv run uvicorn server:app --host 127.0.0.1 --port 8000

The graph runs in a background thread per evaluation. Each call returns as soon
as the follow-up gate has decided - either with the follow-up question, or with
``status: "scoring"`` while plan -> execute -> validate keep running. The caller
polls ``GET /evaluations/{id}`` for the final result.

Every request must carry ``x-agent-token: $EVAL_AGENT_TOKEN``.
State lives in the agent's in-memory checkpointer, so restarting this server
drops in-flight evaluations (the web app falls back to its own scorer).
"""

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
from src.agent.state import AgentState, Difficulty, UserInput
from src.config import constants as _constants  # noqa: F401  (loads .env)
from src.config.tracing import record_outcome, resume_input, start_input, trace_run

TOKEN = os.getenv("EVAL_AGENT_TOKEN")
if not TOKEN:
    raise RuntimeError("EVAL_AGENT_TOKEN is not set")

# Transcription (Gemini) + the follow-up gate usually finish well inside this.
DECISION_TIMEOUT_S = float(os.getenv("EVAL_DECISION_TIMEOUT_S", "90"))
RUN_TTL_S = 60 * 60
MAX_AUDIO_BYTES = 15 * 1024 * 1024

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
    followup_question: Optional[str] = None
    transcript: Optional[str] = None
    communication: Optional[dict] = None
    result: Optional[dict] = None
    review: Optional[dict] = None
    error: Optional[str] = None
    decided: threading.Event = field(default_factory=threading.Event)
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


def _prune() -> None:
    cutoff = time.time() - RUN_TTL_S
    with _LOCK:
        stale = [tid for tid, run in _RUNS.items() if run.updated_at < cutoff]
        for tid in stale:
            _RUNS.pop(tid, None)
    for tid in stale:
        try:
            _GRAPH.checkpointer.delete_thread(tid)
        except Exception:  # best effort
            pass


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _drive(
    thread_id: str, run: Run, graph_input: Any, cleanup: list[str], user_input: UserInput, segment: str, run_input: dict
) -> None:
    """Stream the graph until it pauses for a follow-up or finishes."""
    with trace_run(thread_id, user_input, segment, run_input) as root:
        _stream(thread_id, run, graph_input, cleanup)
        record_outcome(
            root,
            evaluation_result=run.result,
            review_result=run.review,
            followup_question=run.followup_question if run.status == "pending_followup" else None,
        )


def _stream(thread_id: str, run: Run, graph_input: Any, cleanup: list[str]) -> None:
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
        for path in cleanup:
            try:
                os.remove(path)
            except OSError:
                pass


def _start_segment(
    thread_id: str, run: Run, graph_input: Any, cleanup: list[str], segment: str, run_input: Optional[dict] = None
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
        target=_drive, args=(thread_id, run, graph_input, cleanup, user_input, segment, run_input), daemon=True
    )
    worker.start()
    run.decided.wait(DECISION_TIMEOUT_S)
    return run.snapshot(thread_id)


async def _save_audio(audio: Optional[UploadFile]) -> Optional[str]:
    if audio is None:
        return None
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


@app.get("/health")
def health() -> dict:
    return {"ok": True, "runs": len(_RUNS)}


@app.post("/evaluations")
async def create_evaluation(
    question: str = Form(...),
    difficulty: Difficulty = Form(...),
    domain: str = Form(...),
    answer: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
) -> dict:
    _prune()
    audio_path = await _save_audio(audio)
    if not audio_path and not (answer and answer.strip()):
        raise HTTPException(422, "provide answer or audio")

    user_input = UserInput(
        actual_question=question,
        # a recording takes precedence; its transcript becomes the answer
        user_response=None if audio_path else answer.strip(),
        audio_path=audio_path,
        difficulty=difficulty,
        domain=domain,
    )
    thread_id = uuid4().hex
    run = Run()
    with _LOCK:
        _RUNS[thread_id] = run
    cleanup = [audio_path] if audio_path else []
    return _start_segment(thread_id, run, AgentState(input=user_input), cleanup, "start")


@app.post("/evaluations/{thread_id}/followup")
async def answer_followup(
    thread_id: str,
    answer: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
) -> dict:
    run = _get_run(thread_id)
    if run.status != "pending_followup":
        raise HTTPException(409, f"evaluation is {run.status}, not waiting for a follow-up")
    audio_path = await _save_audio(audio)
    if not audio_path and not (answer and answer.strip()):
        raise HTTPException(422, "provide answer or audio")
    payload = {"audio_path": audio_path} if audio_path else answer.strip()
    cleanup = [audio_path] if audio_path else []
    run_input = resume_input(run.followup_question, None if audio_path else answer.strip(), audio_path)
    return _start_segment(thread_id, run, Command(resume=payload), cleanup, "followup", run_input)


@app.post("/evaluations/{thread_id}/skip-followup")
def skip_followup(thread_id: str) -> dict:
    run = _get_run(thread_id)
    if run.status != "pending_followup":
        raise HTTPException(409, f"evaluation is {run.status}, not waiting for a follow-up")
    config = {"configurable": {"thread_id": thread_id}}
    values = _GRAPH.get_state(config).values
    max_iterations = values.get("max_iterations", 2) if isinstance(values, dict) else 2
    run_input = resume_input(run.followup_question, "(skipped)", None)
    # Close the pending follow-up as if it had been answered and exhaust the
    # follow-up budget, so the gate proceeds straight to scoring.
    _GRAPH.update_state(
        config,
        {"awaiting_followup_answer": False, "iteration": max_iterations, "followup_question": None},
        as_node="ask_followup",
    )
    return _start_segment(thread_id, run, None, [], "skip-followup", run_input)


@app.get("/evaluations/{thread_id}")
def get_evaluation(thread_id: str) -> dict:
    return _get_run(thread_id).snapshot(thread_id)
