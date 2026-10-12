"""Gateway behaviour: unified streaming, token counts, telemetry, and the 3 audit fixes.

Tests marked `live` call the real APIs (a fraction of a cent per run) and are skipped
automatically when the API keys aren't available.
"""

from __future__ import annotations

import os
import threading

import pytest
from dotenv import load_dotenv

import src.gateway as gateway_module
from src.gateway import UnifiedLLMGateway, count_tokens, count_tokens_async
from src.pricing import calculate_cost
from src.schemas import StreamChunk
from tests.fakes import FakeStream, anthropic_events, make_gateway, openai_chunks

load_dotenv()
HAS_KEYS = bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("ANTHROPIC_API_KEY"))
live = pytest.mark.skipif(not HAS_KEYS, reason="API keys not set; skipping live test")

OPENAI_MODEL = "gpt-6-luna"
ANTHROPIC_MODEL = "claude-haiku-5-5"


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch, request):
    """Mocked tests don't need tiktoken's download; live tests use the real tokenizer."""
    if "live" not in request.keywords:
        monkeypatch.setattr(gateway_module, "count_tokens", lambda text, model: len(text.split()))


async def collect(gw, model, prompt="Say hello"):
    return [chunk async for chunk in gw.stream(prompt, model)]


# --------------------------------------------------------------- Test 1
@pytest.mark.parametrize(
    "model, fake",
    [
        (OPENAI_MODEL, lambda: {"openai_stream": FakeStream(openai_chunks())}),
        (ANTHROPIC_MODEL, lambda: {"anthropic_stream": FakeStream(anthropic_events())}),
    ],
)
async def test_1_unified_stream_protocol_mocked(model, fake):
    """Both providers yield the same StreamChunk shape and join into the same text."""
    chunks = await collect(make_gateway(**fake()), model)

    assert all(isinstance(c, StreamChunk) for c in chunks)
    assert [c.is_final for c in chunks].count(True) == 1 and chunks[-1].is_final
    assert "".join(c.delta_text for c in chunks) == "Hello world"
    # The OpenAI None/empty-choices chunks and the Anthropic thinking delta produced no text chunks.
    assert all(c.delta_text for c in chunks[:-1])


@pytest.mark.live
@live
@pytest.mark.parametrize("model", [OPENAI_MODEL, ANTHROPIC_MODEL])
async def test_1_unified_stream_protocol_live(model):
    gw = UnifiedLLMGateway()
    chunks = await collect(gw, model, "Reply with exactly these two words and nothing else: Hello world")
    await gw.aclose()

    assert all(isinstance(c, StreamChunk) for c in chunks)
    assert chunks[-1].is_final and chunks[-1].telemetry is not None
    assert "hello world" in "".join(c.delta_text for c in chunks).lower()


# --------------------------------------------------------------- Test 2
# 100 fixed words. (The assignment suggests ~50; 100 keeps OpenAI's small fixed
# per-message overhead of a few tokens inside the +/-10% tolerance.)
DETERMINISTIC_PROMPT = (
    "Please read the following sentence carefully and then reply with only the word done. "
    "The quick brown fox jumps over the lazy dog near the quiet river bank. "
    "Every morning the baker opens the small shop and sells fresh bread to neighbors. "
    "Students walk to school with heavy backpacks full of books and colorful pencils. "
    "In the evening families gather around the table to share stories about their day. "
    "Clouds drift slowly across the blue sky while birds sing in the tall green trees. "
    "This prompt exists only to test token counting accuracy across both providers here today, thank you."
)


def test_2_prompt_has_100_words():
    assert len(DETERMINISTIC_PROMPT.split()) == 100


@pytest.mark.live
@live
async def test_2_token_count_matches_tiktoken_live():
    """Provider-reported prompt tokens are within 10% of the local tiktoken count."""
    local = count_tokens(DETERMINISTIC_PROMPT, OPENAI_MODEL)
    gw = UnifiedLLMGateway()
    chunks = await collect(gw, OPENAI_MODEL, DETERMINISTIC_PROMPT)
    await gw.aclose()

    reported = chunks[-1].telemetry.input_tokens
    assert chunks[-1].telemetry.local_prompt_tokens == local
    assert abs(reported - local) / local <= 0.10, f"provider={reported} tiktoken={local}"


# --------------------------------------------------------------- Test 3
@pytest.mark.parametrize(
    "model, fake",
    [
        (OPENAI_MODEL, lambda: {"openai_stream": FakeStream(openai_chunks())}),
        (ANTHROPIC_MODEL, lambda: {"anthropic_stream": FakeStream(anthropic_events())}),
    ],
)
async def test_3_telemetry_sanity_mocked(model, fake):
    t = (await collect(make_gateway(**fake()), model))[-1].telemetry
    assert t.total_cost_usd > 0 and t.tokens_per_second > 0 and t.ttft_ms > 0
    assert t.input_tokens == 12
    assert t.total_tokens == t.input_tokens + t.output_tokens
    # Anthropic: output tokens come from message_delta (cumulative 3), not message_start (1).
    if model == ANTHROPIC_MODEL:
        assert t.output_tokens == 3


@pytest.mark.live
@live
@pytest.mark.parametrize("model", [OPENAI_MODEL, ANTHROPIC_MODEL])
async def test_3_telemetry_sanity_live(model):
    gw = UnifiedLLMGateway()
    chunks = await collect(gw, model, "Name three colors.")
    await gw.aclose()

    t = chunks[-1].telemetry
    assert chunks[-1].is_final
    assert t.total_cost_usd > 0
    assert t.tokens_per_second > 0
    assert t.ttft_ms > 0


def test_cost_is_exact_to_six_decimals():
    # 1,234 input + 567 output on Claude Sonnet 5.5 ($2 / $10 per 1M)
    # = 0.002468 + 0.00567 = 0.008138
    assert calculate_cost("claude-sonnet-5-5", 1234, 567) == (0.002468, 0.00567, 0.008138)
    # Float math would give 0.1 + 0.2 style drift; Decimal keeps it exact.
    assert calculate_cost("gpt-6-luna", 1, 1) == (0.0, 0.000001, 0.000001)


# --------------------------------------------------------------- Audit fixes
async def test_audit_fix_1_tokenizer_runs_off_the_event_loop(monkeypatch):
    seen = {}

    def fake_count(text, model):
        seen["thread"] = threading.get_ident()
        return 3

    monkeypatch.setattr(gateway_module, "count_tokens", fake_count)
    assert await count_tokens_async("a b c", OPENAI_MODEL) == 3
    assert seen["thread"] != threading.get_ident()


@pytest.mark.parametrize(
    "model, kind",
    [(OPENAI_MODEL, "openai_stream"), (ANTHROPIC_MODEL, "anthropic_stream")],
)
async def test_audit_fix_2_stream_closed_when_consumer_stops_early(model, kind):
    fake = FakeStream(openai_chunks() if kind == "openai_stream" else anthropic_events())
    agen = make_gateway(**{kind: fake}).stream("Say hello", model)
    first = await agen.__anext__()
    assert first.delta_text == "Hello"
    await agen.aclose()  # e.g. the user clicked Stop in the UI
    assert fake.closed


def test_audit_fix_3_tokens_per_second_uses_generation_time():
    # Request starts at t=0, first text at t=1s, ends at t=2s, 100 output tokens.
    chunk = UnifiedLLMGateway._final_chunk("openai", OPENAI_MODEL, 10, 100, 0.0, 1.0, 2.0)
    assert chunk.telemetry.tokens_per_second == 100.0  # draft reported 50.0
    assert chunk.telemetry.ttft_ms == 1000.0
    assert chunk.telemetry.latency_s == 2.0
