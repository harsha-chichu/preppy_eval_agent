import base64
import mimetypes
import os
from typing import Type, TypeVar

from pydantic import BaseModel

from src.config.constants import ANTHROPIC_API_KEY, GEMINI_API_KEY, GEMINI_MODEL, MODEL
from litellm import completion

# Validate required keys
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set")

T = TypeVar("T", bound=BaseModel)


def call_llm(
    prompt: str,
    temperature: float = 0.5,
    max_tokens: int = 1000,
):
    response = completion(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        api_key=ANTHROPIC_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def call_structured_llm(
    prompt: str,
    schema: Type[T],
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> T:
    """Call the LLM and parse the reply into ``schema``.

    Uses litellm's ``response_format`` so the model is constrained to emit JSON
    matching the pydantic model. If the provider ever returns non-JSON we retry
    once with the schema spelled out in the prompt.
    """
    messages = [{"role": "user", "content": prompt}]
    response = completion(
        model=MODEL,
        messages=messages,
        api_key=ANTHROPIC_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=schema,
    )
    content = response.choices[0].message.content
    try:
        return schema.model_validate_json(content)
    except ValueError:
        retry = completion(
            model=MODEL,
            messages=messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Reply with ONLY a JSON object matching this schema, no prose:\n"
                        f"{schema.model_json_schema()}"
                    ),
                }
            ],
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
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
    temperature: float = 1.0,
    max_tokens: int = 1500,
) -> T:
    """Send an audio file to Gemini (via litellm) and parse the reply into ``schema``.

    The raw audio bytes are sent inline so the model actually hears the answer -
    that is what makes the delivery assessment (pronunciation, fluency, pauses,
    filler words) meaningful. Same JSON-with-one-retry contract as
    ``call_structured_llm``.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

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
    response = completion(
        model=GEMINI_MODEL,
        messages=messages,
        api_key=GEMINI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=schema,
    )
    content = response.choices[0].message.content
    try:
        return schema.model_validate_json(content)
    except ValueError:
        retry = completion(
            model=GEMINI_MODEL,
            messages=messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Reply with ONLY a JSON object matching this schema, no prose:\n"
                        f"{schema.model_json_schema()}"
                    ),
                }
            ],
            api_key=GEMINI_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=schema,
        )
        return schema.model_validate_json(retry.choices[0].message.content)
