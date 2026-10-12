"""Fake provider streams so most tests run offline, for free, in under a second.

They mimic the exact shapes the real SDKs return:
- OpenAI: a role-only chunk (content=None), text chunks, then a final usage chunk with
  an EMPTY choices list (the case that crashes naive parsers).
- Anthropic: message_start (input tokens), a thinking delta that must be ignored,
  text_delta events, message_delta (cumulative output tokens), message_stop.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from src.gateway import UnifiedLLMGateway

FAKE_OPENAI_KEY = "sk-test-openai-1234567890abcdef"
FAKE_ANTHROPIC_KEY = "sk-ant-test-1234567890abcdef"


class FakeStream:
    """Async-iterable + async-context-manager, like openai/anthropic AsyncStream."""

    def __init__(self, items, delay: float = 0.005):
        self.items = items
        self.delay = delay
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for item in self.items:
            await asyncio.sleep(self.delay)
            yield item


def openai_chunks(words=("Hello", " world"), prompt_tokens=12, completion_tokens=2):
    def text(t):
        return NS(choices=[NS(delta=NS(content=t))], usage=None)

    return [
        text(None),  # role-only first chunk
        *[text(w) for w in words],
        NS(choices=[NS(delta=NS(content=None))], usage=None),  # finish_reason chunk
        NS(choices=[], usage=NS(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)),
    ]


def anthropic_events(words=("Hello", " world"), input_tokens=12, output_tokens=3):
    return [
        NS(type="message_start", message=NS(usage=NS(input_tokens=input_tokens, output_tokens=1))),
        NS(type="content_block_start"),
        NS(type="content_block_delta", delta=NS(type="thinking_delta", thinking="hmm")),
        *[NS(type="content_block_delta", delta=NS(type="text_delta", text=w)) for w in words],
        NS(type="content_block_stop"),
        NS(type="message_delta", usage=NS(output_tokens=output_tokens)),
        NS(type="message_stop"),
    ]


def make_gateway(openai_stream=None, anthropic_stream=None, openai_error=None, anthropic_error=None):
    """A real gateway whose network clients are swapped for fakes."""
    gw = UnifiedLLMGateway(openai_api_key=FAKE_OPENAI_KEY, anthropic_api_key=FAKE_ANTHROPIC_KEY)
    gw._openai = NS(
        chat=NS(completions=NS(create=AsyncMock(return_value=openai_stream, side_effect=openai_error))),
        close=AsyncMock(),
    )
    gw._anthropic = NS(
        messages=NS(create=AsyncMock(return_value=anthropic_stream, side_effect=anthropic_error)),
        close=AsyncMock(),
    )
    return gw
