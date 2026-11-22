import base64
import mimetypes
import os
import random
import time
from typing import Type, TypeVar

import litellm

from pydantic import BaseModel

from logger import logger
from src.config.constants import (
    ANTHROPIC_API_KEY,
    FALLBACK_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_RETRIES,
    LLM_RETRY_BASE_S,
    MODEL,
)
from src.config.tracing import trace_generation
from litellm import completion


def _is_anthropic(model: str) -> bool:
    return model.startswith("anthropic/")


def _api_key_for(model: str) -> str:
    """Pick the provider key from the litellm model prefix."""
    key = ANTHROPIC_API_KEY if _is_anthropic(model) else GEMINI_API_KEY
    if not key:
        name = "ANTHROPIC_API_KEY" if _is_anthropic(model) else "GEMINI_API_KEY"
        raise RuntimeError(f"{name} is not set (needed for model {model!r})")
    return key


# Fail fast at startup if the main text model's key is missing.
_api_key_for(MODEL)

T = TypeVar("T", bound=BaseModel)


# Errors worth retrying: the provider is overloaded or briefly unreachable
# (Gemini's 503 "model is currently experiencing high demand", 429, 5xx,
# timeouts). Bad requests and auth errors are not retried.
TRANSIENT_ERRORS = (
    litellm.ServiceUnavailableError,
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.Timeout,
    litellm.APIConnectionError,
)


def _backoff(retry: int) -> float:
    """Exponential backoff with jitter: ~1s, 2s, 4s ... for retry 1, 2, 3 ..."""
    return LLM_RETRY_BASE_S * (2 ** (retry - 1)) * random.uniform(0.75, 1.25)


def _complete(name: str, model: str, messages: list, temperature: float, max_tokens: int, attempt: int = 1, **kwargs):
    """``litellm.completion`` recorded as a Langfuse generation named ``name``.

    Temporary provider errors are retried up to ``LLM_RETRIES`` times with
    backoff, then once on ``FALLBACK_MODEL`` if one is set. Without this, one
    503 from Gemini failed the whole evaluation and the answer went unscored.
    """
    for retry in range(LLM_RETRIES + 1):
        try:
            return _complete_once(name, model, messages, temperature, max_tokens, attempt, **kwargs)
        except TRANSIENT_ERRORS as exc:
            if retry == LLM_RETRIES:
                if FALLBACK_MODEL and FALLBACK_MODEL != model:
                    logger.warning("%s: %s still failing after %d retries (%s); trying %s.",
                                   name, model, LLM_RETRIES, type(exc).__name__, FALLBACK_MODEL)
                    kwargs.pop("api_key", None)  # the fallback may use another provider's key
                    return _complete_once(name, FALLBACK_MODEL, messages, temperature, max_tokens, attempt, **kwargs)
                raise
            delay = _backoff(retry + 1)
            logger.warning("%s: %s on %s, retry %d/%d in %.1fs.",
                           name, type(exc).__name__, model, retry + 1, LLM_RETRIES, delay)
            time.sleep(delay)


def _complete_once(name: str, model: str, messages: list, temperature: float, max_tokens: int, attempt: int = 1, **kwargs):
    params = {"temperature": temperature}
    if _is_anthropic(model):
        # Current Claude models (e.g. Haiku 5.5) reject non-default temperature,
        # and thinking is on by default and counts toward max_tokens.
        params = {}
        max_tokens = max(max_tokens, 8000)
    kwargs.setdefault("api_key", _api_key_for(model))
    with trace_generation(name, model, messages, temperature, max_tokens, attempt) as record:
        response = completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **params,
            **kwargs,
        )
        record(response)
    return response


def call_llm(
    prompt: str,
    name: str,
    temperature: float = 0.5,
    max_tokens: int = 1000,
):
    response = _complete(
        name,
        MODEL,
        [{"role": "user", "content": prompt}],
        temperature,
        max_tokens,
    )

    return response.choices[0].message.content


def call_structured_llm(
    prompt: str,
    schema: Type[T],
    name: str,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> T:
    """Call the LLM and parse the reply into ``schema``.

    Uses litellm's ``response_format`` so the model is constrained to emit JSON
    matching the pydantic model. If the provider ever returns non-JSON we retry
    once with the schema spelled out in the prompt.
    """
    messages = [{"role": "user", "content": prompt}]
    response = _complete(
        name,
        MODEL,
        messages,
        temperature,
        max_tokens,
        response_format=schema,
    )
    content = response.choices[0].message.content
    try:
        return schema.model_validate_json(content)
    except ValueError:
        retry = _complete(
            name,
            MODEL,
            messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Reply with ONLY a JSON object matching this schema, no prose:\n"
                        f"{schema.model_json_schema()}"
                    ),
                }
            ],
            temperature,
            max_tokens,
            attempt=2,
                response_format=schema,
        )
        return schema.model_validate_json(retry.choices[0].message.content)


_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}


def call_gemini_audio(
    prompt: str,
    audio_path: str,
    schema: Type[T],
    name: str = "transcribe-and-assess-audio",
    temperature: float = 1.0,
    max_tokens: int = 1500,
) -> T:
    """Send an audio file to Gemini (via litellm) and parse the reply into ``schema``.

    The raw audio bytes are sent inline so the model actually hears the answer -
    that is what makes the delivery assessment (pronunciation, fluency, pauses,
    filler words) meaningful. Same JSON-with-one-retry contract as
    ``call_structured_llm``.
    """
    ext = os.path.splitext(audio_path)[1].lower()
    mime = _AUDIO_MIME.get(ext) or mimetypes.guess_type(audio_path)[0] or "audio/wav"
    with open(audio_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()

    audio_part = {"type": "file", "file": {"file_data": f"data:{mime};base64,{b64}"}}
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, audio_part],
        }
    ]
    response = _complete(
        name,
        GEMINI_MODEL,
        messages,
        temperature,
        max_tokens,
        response_format=schema,
    )
    content = response.choices[0].message.content
    try:
        return schema.model_validate_json(content)
    except ValueError:
        retry = _complete(
            name,
            GEMINI_MODEL,
            messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Reply with ONLY a JSON object matching this schema, no prose:\n"
                        f"{schema.model_json_schema()}"
                    ),
                }
            ],
            temperature,
            max_tokens,
            attempt=2,
                response_format=schema,
        )
        return schema.model_validate_json(retry.choices[0].message.content)
