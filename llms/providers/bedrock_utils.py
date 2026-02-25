"""Tools to generate from Claude models via AWS Bedrock.

Required environment variables:
  AWS_REGION            — e.g. "us-east-1" (required)
  AWS_ACCESS_KEY_ID     — standard AWS credentials (or use instance profile/SSO)
  AWS_SECRET_ACCESS_KEY — standard AWS credentials (or use instance profile/SSO)
  AWS_SESSION_TOKEN     — (optional) for temporary credentials
  BEDROCK_MODEL_ID      — (optional) fallback default model if not specified via --model
"""

import json
import os
import random
import time
from typing import Any

import boto3
import botocore.exceptions

MODEL_ID_MAP: dict[str, str] = {
    "claude-3-5-v2-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-7-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-opus-4-5": "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-4-sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4-1": "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "claude-4-1-opus": "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "claude-opus-4": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-4-opus": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
}

_bedrock_client: Any = None


def _get_bedrock_client() -> Any:
    global _bedrock_client
    if _bedrock_client is None:
        region = os.environ.get("AWS_REGION")
        if not region:
            raise ValueError(
                "AWS_REGION environment variable must be set when using Bedrock."
            )
        _bedrock_client = boto3.client(
            "bedrock-runtime", region_name=region
        )
    return _bedrock_client


def _retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
) -> Any:
    """Retry a function with exponential backoff on throttling errors."""

    def wrapper(*args, **kwargs):  # type: ignore
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except botocore.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code not in (
                    "ThrottlingException",
                    "TooManyRequestsException",
                    "ServiceUnavailableException",
                ):
                    raise
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay} seconds.")
                time.sleep(delay)

    return wrapper


def _invoke_model(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_token: str | None = None,
) -> str:
    """Call the Bedrock invoke_model API and return the response text."""
    model_id = MODEL_ID_MAP.get(model, model)

    # Separate system message from the rest
    system_text: str | None = None
    anthropic_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system" and system_text is None:
            system_text = content
        else:
            anthropic_messages.append(
                {
                    "role": role if role in ("user", "assistant") else "user",
                    "content": [{"type": "text", "text": content}],
                }
            )

    request_body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
    }

    # Bedrock Claude does not allow both temperature and top_p simultaneously.
    # Prefer temperature; only fall back to top_p if temperature is not set.
    if temperature is not None:
        request_body["temperature"] = temperature
    elif top_p is not None:
        request_body["top_p"] = top_p

    if system_text is not None:
        request_body["system"] = system_text
    if stop_token:
        request_body["stop_sequences"] = [stop_token]

    client = _get_bedrock_client()
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body),
    )
    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


_invoke_model_with_retry = _retry_with_exponential_backoff(_invoke_model)


def generate_from_bedrock_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    """Generate a response from Claude via AWS Bedrock.

    Args:
        messages: List of {"role": ..., "content": ...} dicts (OpenAI format).
        model: Friendly model name (looked up in MODEL_ID_MAP) or full Bedrock ARN.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        top_p: Top-p sampling value.
        context_length: Unused; kept for interface parity with openai utils.
        stop_token: Optional stop sequence.

    Returns:
        The generated text string.
    """
    return _invoke_model_with_retry(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop_token=stop_token,
    )
