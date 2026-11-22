# preppy eval agent

The evaluation agent behind the [LMS web app](https://github.com/harsha-chichu/LMS_UI). For each interview answer it:
- decides whether to ask a follow-up
- scores the answer against a per-domain rubric
- writes feedback, strengths and weaknesses
- writes a **suggested answer** with key points
- scores delivery for voice answers (grammar, pronunciation, fluency, clarity)

It also writes session summaries. It runs on Gemini via LiteLLM, as a LangGraph graph behind a small FastAPI server (`server.py`).

The web app treats this agent as its main scorer. If the agent is down or errors, the app falls back to its own Claude scorer.

## Run it

Create a `.env` with at least `EVAL_AGENT_TOKEN` and `GEMINI_API_KEY` (see below), then:

```bash
uv run uvicorn server:app --host 127.0.0.1 --port 8000
```

In the web app, set `EVAL_AGENT_URL` (e.g. `http://127.0.0.1:8000`) and the same `EVAL_AGENT_TOKEN`.

Other entry points, for trying the graph by hand:
- `uv run streamlit run app.py`: a small UI
- `uv run python main.py`: a CLI demo

## Environment variables

`.env` is loaded on start and wins over shell variables.

| Variable | Default | What it does |
|---|---|---|
| `EVAL_AGENT_TOKEN` | **required** | Shared secret; every request must send it as `x-agent-token`. The server won't start without it. |
| `GEMINI_API_KEY` | unset | Gemini access. Also turns on voice (transcription and delivery scores); `/health` reports `voice: true`. |
| `MODEL` | `gemini/gemini-3.5-flash-lite` | Text model for follow-up, plan, execute (scoring) and validate. LiteLLM route: `gemini/…` or `anthropic/…`. |
| `GEMINI_MODEL` | `gemini/gemini-3.5-flash-lite` | Audio model for transcribing and assessing spoken answers. |
| `ANTHROPIC_API_KEY` | unset | Only needed when `MODEL` starts with `anthropic/`. |
| `EVAL_DECISION_TIMEOUT_S` | `40` | How long a request waits for the follow-up decision before answering `status: "scoring"`. Keep it under the web app's 45 s limit. |
| `CONTENT_WEIGHT` | `0.8` | `overall_score = CONTENT_WEIGHT × content + (1 − CONTENT_WEIGHT) × communication` (voice answers). |
| `LLM_RETRIES` | `3` | Retries for temporary provider errors (Gemini 503 "high demand", 429, 5xx, timeouts, dropped connections), with backoff of about 1 s, 2 s, 4 s. Bad requests and auth errors are not retried. |
| `LLM_RETRY_BASE_S` | `1.0` | First backoff delay in seconds; doubles on each retry. |
| `FALLBACK_MODEL` | unset | A LiteLLM model (e.g. a different Gemini model) tried once after the retries are used up. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | unset | Optional Langfuse tracing (standard Langfuse SDK variables). Traces carry the web app's `user_id`. Tracing is a no-op without them. |

## API

Every request needs the header `x-agent-token: $EVAL_AGENT_TOKEN`.

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | `{ ok, runs, voice }` |
| `GET` | `/rubrics?domain=…&difficulty=easy\|medium\|hard` | How answers in a domain are scored at a level (shown before a session) |
| `POST` | `/evaluations` | Start an evaluation. Form fields: `question`, `difficulty`, `domain`, `answer` and/or `audio` (a file, up to 15 MB), and an optional `user_id`. |
| `POST` | `/evaluations/{id}/followup` | Answer the follow-up: `answer` and/or `audio` |
| `POST` | `/evaluations/{id}/skip-followup` | Skip the follow-up and go straight to scoring |
| `GET` | `/evaluations/{id}` | Poll a run |
| `POST` | `/summaries` | Write the session summary from all questions and answers (JSON body) |

**Start, follow-up and skip** reply as soon as the follow-up decision is made. The reply looks like this:

```json
{
  "thread_id": "…",
  "status": "running | pending_followup | scoring | completed | failed",
  "followup_question": "… or null",
  "transcript": "… or null",
  "communication": { "grammar_score": 8, "pronunciation_score": 7, "fluency_score": 7, "clarity_score": 8, "…": "…" },
  "result": null,
  "error": null
}
```

What each status means:
- **`pending_followup`:** the agent wants an answer to `followup_question`. Send it, or skip it.
- **`scoring`:** scoring continues in the background. Poll `GET /evaluations/{id}`.
- **`completed`:** `result` holds:
  - `score` (0–10)
  - `is_correct`, `feedback`
  - `strengths`, `weaknesses`
  - `followup_evaluation`
  - **`suggested_answer`** and **`suggested_points`**, which are always present
  - and, for voice answers, `communication_evaluation` and `overall_score`

Runs live in memory for an hour. Restarting the server drops in-flight runs; the web app's fallback scorer covers them.

## How it works

The graph is in `src/agent/graph.py`, and `graph.png` shows it:

```
[transcribe] → followup_gate → (ask_followup → followup_gate)* → (plan ∥ assess_delivery) → execute → validate → (execute → validate)* → END
```

- **`transcribe`** runs first only for a recording sent without text. When the user's reviewed text comes with the recording, delivery is assessed later (`assess_delivery`), in parallel with `plan`.
- **`followup_gate`** decides whether a follow-up is needed (never on easy; at most `max_iterations`, default 2). It's the only step the user waits for.
- **`plan`** builds a grading plan from the rubric (`src/rubrics.py`).
- **`execute`** scores and writes the suggested answer (`src/prompts/execute.py`).
- **`validate`** checks the result against the plan. If the score isn't grounded or doesn't match the rubric, it sends the run back to `execute`, at most twice.

## Tests

```bash
uv run --with pytest --with httpx pytest tests
```

No API keys or network needed.

## Changes

- **Suggested answer is required.** `suggested_answer` and `suggested_points` used to be optional. On longer runs, usually ones with a follow-up, the model sometimes returned a score without them, so the web app showed no suggested answer. They're now required in `EvaluationResult`, the scoring call's output limit went from 3000 to 6000 tokens, and an empty suggestion is logged as a warning.
