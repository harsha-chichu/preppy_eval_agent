from typing import Type, TypeVar

from pydantic import BaseModel

from src.config.constants import ANTHROPIC_API_KEY, MODEL
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
