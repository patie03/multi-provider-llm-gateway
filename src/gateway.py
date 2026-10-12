"""UnifiedLLMGateway: one async streaming interface over OpenAI and Anthropic."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
from functools import lru_cache
from typing import AsyncGenerator, Optional

import anthropic
import openai
import tiktoken
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .pricing import ModelInfo, calculate_cost, get_model_info
from .schemas import ChunkTelemetry, Provider, StreamChunk

logger = logging.getLogger(__name__)

_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")


def redact(text: str) -> str:
    """Remove anything that looks like an API key from a string."""
    return _KEY_PATTERN.sub("sk-***REDACTED***", text)


# ---------------------------------------------------------------- exceptions
class GatewayError(Exception):
    """Base class. Messages are always redacted before being stored."""

    def __init__(self, message: str, provider: Optional[str] = None):
        self.provider = provider
        super().__init__(redact(message))


class GatewayProviderError(GatewayError):
    """Provider returned an error, or the connection failed."""


class AuthenticationError(GatewayError):
    """API key is missing, malformed, or rejected by the provider."""


class RateLimitExceededError(GatewayError):
    """Provider rate limit (HTTP 429) was hit."""


class UnknownModelError(GatewayError):
    """Model is not in the pricing catalog."""


# ---------------------------------------------------------------- tokenizer
@lru_cache(maxsize=16)
def _encoding_for(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # tiktoken doesn't know newer model names; o200k_base is the current OpenAI encoding.
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str) -> int:
    return len(_encoding_for(model).encode(text))


async def count_tokens_async(text: str, model: str) -> int:
    """AUDIT FIX #1: tokenizing can download a file and is CPU work, so run it off the event loop."""
    return await asyncio.to_thread(count_tokens, text, model)


# ---------------------------------------------------------------- gateway
class UnifiedLLMGateway:
    """Streams normalized StreamChunk objects from OpenAI or Anthropic.

    Async clients are created once per gateway and reused, so HTTP connections
    are pooled across requests.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        max_tokens: int = 2048,
        timeout_s: float = 60.0,
    ) -> None:
        load_dotenv()
        self.max_tokens = max_tokens
        openai_key = openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY")
        anthropic_key = (
            anthropic_api_key if anthropic_api_key is not None else os.getenv("ANTHROPIC_API_KEY")
        )

        self._openai: Optional[AsyncOpenAI] = None
        self._anthropic: Optional[AsyncAnthropic] = None
        self._openai_key_problem: Optional[str] = self._check_key(openai_key, "sk-", "OPENAI_API_KEY")
        self._anthropic_key_problem: Optional[str] = self._check_key(
            anthropic_key, "sk-ant-", "ANTHROPIC_API_KEY"
        )

        if self._openai_key_problem is None:
            self._openai = AsyncOpenAI(api_key=openai_key, timeout=timeout_s, max_retries=2)
        if self._anthropic_key_problem is None:
            self._anthropic = AsyncAnthropic(api_key=anthropic_key, timeout=timeout_s, max_retries=2)

        # AUDIT FIX #1: load the tokenizer once at startup, not during the first request.
        try:
            _encoding_for("o200k_base")
        except Exception:  # offline: fall back to the provider's own count later
            logger.warning("Could not preload the tiktoken encoding; will retry on first request.")

    @staticmethod
    def _check_key(key: Optional[str], prefix: str, env_name: str) -> Optional[str]:
        if not key or not key.strip():
            return f"{env_name} is not set."
        if not key.startswith(prefix):
            return f"{env_name} is malformed (expected it to start with '{prefix}')."
        return None

    def __repr__(self) -> str:  # never include keys
        return (
            f"UnifiedLLMGateway(openai={'ready' if self._openai else 'unavailable'}, "
            f"anthropic={'ready' if self._anthropic else 'unavailable'})"
        )

    async def aclose(self) -> None:
        if self._openai is not None:
            await self._openai.close()
        if self._anthropic is not None:
            await self._anthropic.close()

    # ------------------------------------------------------------ public API
    async def stream(
        self, prompt: str, model: str, temperature: float = 0.7
    ) -> AsyncGenerator[StreamChunk, None]:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        try:
            info = get_model_info(model)
        except KeyError as exc:
            raise UnknownModelError(str(exc)) from None

        if info.provider == "openai":
            generator = self._stream_openai(prompt, model, info, temperature)
        else:
            generator = self._stream_anthropic(prompt, model, info, temperature)

        # AUDIT FIX #2 (part b): `aclosing` makes sure the inner provider generator, and the
        # HTTP stream it holds, is closed right away if the consumer stops early.
        async with contextlib.aclosing(generator) as chunks:
            async for chunk in chunks:
                yield chunk

    # ------------------------------------------------------------ OpenAI
    async def _stream_openai(
        self, prompt: str, model: str, info: ModelInfo, temperature: float
    ) -> AsyncGenerator[StreamChunk, None]:
        if self._openai is None:
            raise AuthenticationError(self._openai_key_problem or "OpenAI unavailable.", "openai")

        try:
            local_prompt_tokens: Optional[int] = await count_tokens_async(prompt, model)
        except Exception:
            local_prompt_tokens = None
        request: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": self.max_tokens,
        }
        if info.supports_temperature:
            request["temperature"] = temperature

        start = time.perf_counter()
        first_text_at: Optional[float] = None
        usage = None
        try:
            stream = await self._openai.chat.completions.create(**request)
            # AUDIT FIX #2: `async with` closes the HTTP stream even if the consumer stops early.
            async with stream:
                async for chunk in stream:
                    if chunk.usage is not None:  # final usage chunk has an empty choices list
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    text = chunk.choices[0].delta.content if chunk.choices[0].delta else None
                    if not text:  # role-only and finish chunks have content=None
                        continue
                    if first_text_at is None:
                        first_text_at = time.perf_counter()
                    yield StreamChunk(delta_text=text, provider="openai", model=model)
        except openai.AuthenticationError:
            raise AuthenticationError("OpenAI rejected the API key.", "openai") from None
        except openai.RateLimitError:
            raise RateLimitExceededError("OpenAI rate limit exceeded. Try again shortly.", "openai") from None
        except (openai.APIConnectionError, openai.APITimeoutError):
            raise GatewayProviderError("Could not connect to OpenAI.", "openai") from None
        except openai.APIStatusError as exc:
            logger.warning("OpenAI HTTP %s: %s", exc.status_code, redact(str(exc)))
            raise GatewayProviderError(f"OpenAI returned HTTP {exc.status_code}.", "openai") from None
        end = time.perf_counter()

        input_tokens = usage.prompt_tokens if usage else (local_prompt_tokens or 0)
        output_tokens = usage.completion_tokens if usage else 0
        yield self._final_chunk(
            "openai", model, input_tokens, output_tokens, start, first_text_at, end, local_prompt_tokens
        )

    # ------------------------------------------------------------ Anthropic
    async def _stream_anthropic(
        self, prompt: str, model: str, info: ModelInfo, temperature: float
    ) -> AsyncGenerator[StreamChunk, None]:
        if self._anthropic is None:
            raise AuthenticationError(
                self._anthropic_key_problem or "Anthropic unavailable.", "anthropic"
            )

        request: dict = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        if info.supports_temperature:
            request["temperature"] = temperature

        start = time.perf_counter()
        first_text_at: Optional[float] = None
        input_tokens = 0
        output_tokens = 0
        try:
            stream = await self._anthropic.messages.create(**request)
            # AUDIT FIX #2: `async with` closes the HTTP stream even if the consumer stops early.
            async with stream:
                async for event in stream:
                    if event.type == "message_start":
                        input_tokens = event.message.usage.input_tokens
                        output_tokens = event.message.usage.output_tokens or 0
                    elif event.type == "content_block_delta":
                        # Only text deltas are user-visible; thinking/signature deltas are skipped.
                        if event.delta.type == "text_delta" and event.delta.text:
                            if first_text_at is None:
                                first_text_at = time.perf_counter()
                            yield StreamChunk(delta_text=event.delta.text, provider="anthropic", model=model)
                    elif event.type == "message_delta":
                        # usage.output_tokens on message_delta is cumulative, so take the latest value.
                        if event.usage is not None and event.usage.output_tokens is not None:
                            output_tokens = event.usage.output_tokens
        except anthropic.AuthenticationError:
            raise AuthenticationError("Anthropic rejected the API key.", "anthropic") from None
        except anthropic.RateLimitError:
            raise RateLimitExceededError(
                "Anthropic rate limit exceeded. Try again shortly.", "anthropic"
            ) from None
        except (anthropic.APIConnectionError, anthropic.APITimeoutError):
            raise GatewayProviderError("Could not connect to Anthropic.", "anthropic") from None
        except anthropic.APIStatusError as exc:
            logger.warning("Anthropic HTTP %s: %s", exc.status_code, redact(str(exc)))
            raise GatewayProviderError(
                f"Anthropic returned HTTP {exc.status_code}.", "anthropic"
            ) from None
        end = time.perf_counter()

        yield self._final_chunk("anthropic", model, input_tokens, output_tokens, start, first_text_at, end)

    # ------------------------------------------------------------ telemetry
    @staticmethod
    def _final_chunk(
        provider: Provider,
        model: str,
        input_tokens: int,
        output_tokens: int,
        start: float,
        first_text_at: Optional[float],
        end: float,
        local_prompt_tokens: Optional[int] = None,
    ) -> StreamChunk:
        latency_s = end - start
        # TTFT is measured to the first non-empty text delta, not to the HTTP response.
        ttft_s = (first_text_at - start) if first_text_at is not None else latency_s
        # AUDIT FIX #3: measure speed over the generation window (first text -> end),
        # not over the whole request, which also includes the wait before the first token.
        generation_s = (end - first_text_at) if first_text_at is not None else latency_s
        if generation_s <= 0:
            generation_s = latency_s
        tokens_per_second = output_tokens / generation_s if generation_s > 0 else 0.0
        input_cost, output_cost, total_cost = calculate_cost(model, input_tokens, output_tokens)

        return StreamChunk(
            delta_text="",
            provider=provider,
            model=model,
            is_final=True,
            telemetry=ChunkTelemetry(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                local_prompt_tokens=local_prompt_tokens,
                ttft_ms=round(ttft_s * 1000, 1),
                latency_s=round(latency_s, 3),
                tokens_per_second=round(tokens_per_second, 1),
                input_cost_usd=input_cost,
                output_cost_usd=output_cost,
                total_cost_usd=total_cost,
            ),
        )
