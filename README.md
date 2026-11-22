# preppy eval agent

Evaluation agent behind the LMS web app (`server.py`): decides follow-ups and scores answers with Gemini via LiteLLM.

```bash
uv run uvicorn server:app --host 127.0.0.1 --port 8000
```

## Reliability settings

| Variable | Default | What it does |
|---|---|---|
| `LLM_RETRIES` | `3` | Retries for temporary provider errors (Gemini 503 "high demand", 429, 5xx, timeouts, dropped connections), with backoff of about 1 s, 2 s, 4 s. Bad requests and auth errors are not retried. |
| `LLM_RETRY_BASE_S` | `1.0` | First backoff delay in seconds; doubles on each retry. |
| `FALLBACK_MODEL` | unset | A LiteLLM model (e.g. a different Gemini model) tried once after the retries are used up. |

## Tests

```bash
uv run --with pytest --with httpx pytest tests
```

No API keys or network needed.
