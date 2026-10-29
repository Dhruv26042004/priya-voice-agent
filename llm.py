"""LLM access with automatic failover across free tiers.

Free tiers are small: Groq allows ~8,000 tokens/minute and Cerebras 5 requests/minute,
and Cerebras' free tier also sheds load when busy. Neither alone covers a busy call.
When one answers 429 the same request goes to the other, and if both are limited we
wait as long as Groq asks (usually about a second) and try again, rather than leave
the caller in silence. A 429 arrives before streaming starts, so nothing is half-spoken.

Used both by the live call (FailoverGroqLLMService) and the post-call analysis
(json_completion).
"""

import asyncio
import json
import os
import time

from loguru import logger
from openai import APIConnectionError, AsyncOpenAI, BadRequestError, InternalServerError, RateLimitError

from pipecat.services.groq.llm import GroqLLMService
from pipecat.utils.types import assert_given

GROQ_URL, GROQ_MODEL = "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"
CEREBRAS_URL, CEREBRAS_MODEL = "https://api.cerebras.ai/v1", "gpt-oss-120b"

# Longest we'll keep a caller waiting while both providers are rate limited.
MAX_WAIT_SECS = 6.0


# Worth trying the other provider for: rate limits, provider outages, network trouble.
RETRYABLE = (RateLimitError, InternalServerError, APIConnectionError)


def _retry_after(error: Exception) -> float:
    try:
        return float(error.response.headers.get("retry-after", 1.0))
    except (AttributeError, TypeError, ValueError):
        return 1.0


async def create_with_failover(targets: list[tuple[str, AsyncOpenAI, str]], params: dict, max_wait: float = MAX_WAIT_SECS):
    """Run a chat completion on the first provider that isn't rate limited.

    targets: (name, client, model) in order of preference.
    """
    deadline = time.monotonic() + max_wait
    while True:
        wait = None
        for name, client, model in targets:
            try:
                return await client.chat.completions.create(**{**params, "model": model})
            except RETRYABLE as e:
                logger.warning(f"{name} unavailable ({type(e).__name__})")
                last_error = e
                wait = _retry_after(e) if wait is None else min(wait, _retry_after(e))
        wait = min(wait, 2.0)
        if time.monotonic() + wait > deadline:
            raise last_error
        logger.warning(f"All LLM providers rate limited, retrying in {wait:.1f}s")
        await asyncio.sleep(wait)


def default_targets() -> list[tuple[str, AsyncOpenAI, str]]:
    """Groq first, then Cerebras if a key is set. The SDK's own retries would sleep on
    one provider; we switch instead, so they're off."""
    targets = [("Groq", AsyncOpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_URL, max_retries=0), GROQ_MODEL)]
    if key := os.getenv("CEREBRAS_API_KEY"):
        targets.append(("Cerebras", AsyncOpenAI(api_key=key, base_url=CEREBRAS_URL, max_retries=0), CEREBRAS_MODEL))
    return targets


async def json_completion(system: str, user: str, max_wait: float = 60.0, targets=None) -> dict:
    """One non-streaming completion that returns a JSON object. Not latency-critical,
    so it can wait out rate limits for longer than a live call would."""
    params = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "reasoning_effort": "low",
    }
    for attempt in range(3):
        try:
            resp = await create_with_failover(targets or default_targets(), params, max_wait=max_wait)
            return json.loads(resp.choices[0].message.content)
        except (BadRequestError, json.JSONDecodeError) as e:
            # Now and then the model emits invalid JSON (Groq rejects it with a 400); just ask again.
            if attempt == 2 or (isinstance(e, BadRequestError) and "json" not in str(e).lower()):
                raise
            logger.warning(f"Invalid JSON from the model, retrying: {e}")


class FailoverGroqLLMService(GroqLLMService):
    def __init__(self, *, backup_api_key: str | None, backup_base_url: str, backup_model: str, **kwargs):
        super().__init__(**kwargs)
        self._targets = [("Groq", self._client.with_options(max_retries=0), None)]
        if backup_api_key:
            backup = AsyncOpenAI(api_key=backup_api_key, base_url=backup_base_url, max_retries=0)
            self._targets.append(("Cerebras", backup, backup_model))

    async def get_chat_completions(self, context):
        from prompts import language_hint

        system = assert_given(self._settings.system_instruction) + language_hint(context.messages)
        params_from_context = await self.get_llm_adapter().get_llm_invocation_params(
            context, system_instruction=system, convert_developer_to_user=not self.supports_developer_role
        )
        params = self.build_chat_completion_params(params_from_context)
        targets = [(name, client, model or params["model"]) for name, client, model in self._targets]
        return await create_with_failover(targets, params)
