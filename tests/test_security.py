"""Key handling and secret redaction. All offline: no real API calls."""

from __future__ import annotations

import logging
import traceback

import openai
import anthropic
import httpx2 as httpx
import pytest

import src.gateway as gateway_module
from src.gateway import (
    AuthenticationError,
    GatewayError,
    GatewayProviderError,
    RateLimitExceededError,
    UnifiedLLMGateway,
    redact,
)
from tests.fakes import FAKE_ANTHROPIC_KEY, FAKE_OPENAI_KEY, make_gateway

OPENAI_REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
ANTHROPIC_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
LEAKY_KEY = "sk-proj-SuperSecretKey1234567890"


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    monkeypatch.setattr(gateway_module, "count_tokens", lambda text, model: len(text.split()))


async def drain(gw, model):
    return [c async for c in gw.stream("hi", model)]


def full_text_of(exc: BaseException) -> str:
    """Everything a user or a log could ever see: message, repr and traceback."""
    return str(exc) + repr(exc) + "".join(traceback.format_exception(exc))


# --------------------------------------------------------------- Test 4
@pytest.mark.parametrize(
    "openai_key, anthropic_key, model, expected",
    [
        ("", FAKE_ANTHROPIC_KEY, "gpt-6-luna", "OPENAI_API_KEY is not set"),
        ("   ", FAKE_ANTHROPIC_KEY, "gpt-6-luna", "OPENAI_API_KEY is not set"),
        ("not-a-key", FAKE_ANTHROPIC_KEY, "gpt-6-luna", "OPENAI_API_KEY is malformed"),
        (FAKE_OPENAI_KEY, "", "claude-haiku-5-5", "ANTHROPIC_API_KEY is not set"),
        (FAKE_OPENAI_KEY, "sk-wrong-prefix", "claude-haiku-5-5", "ANTHROPIC_API_KEY is malformed"),
    ],
)
async def test_4_missing_or_malformed_key_raises_authentication_error(
    openai_key, anthropic_key, model, expected
):
    gw = UnifiedLLMGateway(openai_api_key=openai_key, anthropic_api_key=anthropic_key)
    with pytest.raises(AuthenticationError, match=expected):
        await drain(gw, model)


async def test_4_missing_environment_variables(monkeypatch):
    monkeypatch.setattr(gateway_module, "load_dotenv", lambda *a, **k: None)  # ignore the real .env
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    gw = UnifiedLLMGateway()  # constructing must not crash
    with pytest.raises(AuthenticationError, match="not set"):
        await drain(gw, "gpt-6-luna")
    with pytest.raises(AuthenticationError, match="not set"):
        await drain(gw, "claude-haiku-5-5")


async def test_4_key_rejected_by_provider():
    err = anthropic.AuthenticationError(
        "invalid x-api-key", response=httpx.Response(401, request=ANTHROPIC_REQ), body=None
    )
    with pytest.raises(AuthenticationError, match="Anthropic rejected the API key"):
        await drain(make_gateway(anthropic_error=err), "claude-haiku-5-5")


@pytest.mark.parametrize(
    "error, expected",
    [
        (openai.RateLimitError("slow down", response=httpx.Response(429, request=OPENAI_REQ), body=None),
         RateLimitExceededError),
        (openai.APIConnectionError(request=OPENAI_REQ), GatewayProviderError),
    ],
)
async def test_provider_errors_become_typed_exceptions(error, expected):
    with pytest.raises(expected):
        await drain(make_gateway(openai_error=error), "gpt-6-luna")


# --------------------------------------------------------------- Test 5
@pytest.mark.parametrize(
    "secret",
    [
        "sk-proj-AbCdEf1234567890",
        "sk-ant-api03-AbCdEf_1234-567890",
        "sk-test-openai-1234567890abcdef",
    ],
)
def test_5_redact_removes_key_formats(secret):
    out = redact(f"request failed with key {secret} at 10:00")
    assert secret not in out
    assert "sk-***REDACTED***" in out


def test_5_exception_messages_are_redacted():
    exc = GatewayError(f"bad key {LEAKY_KEY}")
    assert LEAKY_KEY not in full_text_of(exc)


def test_5_gateway_repr_never_shows_keys():
    gw = UnifiedLLMGateway(openai_api_key=FAKE_OPENAI_KEY, anthropic_api_key=FAKE_ANTHROPIC_KEY)
    text = repr(gw) + str(gw)
    assert FAKE_OPENAI_KEY not in text and FAKE_ANTHROPIC_KEY not in text


async def test_5_provider_error_containing_key_is_not_leaked():
    """OpenAI echoes part of a bad key in its error. Neither the raised error nor its traceback may show it."""
    err = openai.AuthenticationError(
        f"Incorrect API key provided: {LEAKY_KEY}",
        response=httpx.Response(401, request=OPENAI_REQ),
        body=None,
    )
    with pytest.raises(AuthenticationError) as info:
        await drain(make_gateway(openai_error=err), "gpt-6-luna")
    assert LEAKY_KEY not in full_text_of(info.value)


async def test_5_logs_are_redacted(caplog):
    err = openai.APIStatusError(
        f"server error while using {LEAKY_KEY}",
        response=httpx.Response(500, request=OPENAI_REQ),
        body=None,
    )
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(GatewayProviderError, match="HTTP 500"):
            await drain(make_gateway(openai_error=err), "gpt-6-luna")
    assert caplog.records, "the 500 should be logged for debugging"
    assert LEAKY_KEY not in caplog.text
