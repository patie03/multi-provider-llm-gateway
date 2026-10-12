"""Pydantic v2 models shared by the gateway, the web UI, and the tests."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Provider = Literal["openai", "anthropic"]


class ChunkTelemetry(BaseModel):
    """Metrics attached to the final chunk of a stream (is_final=True)."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0, description="Prompt tokens reported by the provider")
    output_tokens: int = Field(ge=0, description="Completion tokens reported by the provider")
    total_tokens: int = Field(ge=0)
    local_prompt_tokens: Optional[int] = Field(
        default=None, description="Local tiktoken estimate of the prompt (OpenAI only)"
    )
    ttft_ms: float = Field(ge=0, description="Time to first visible text token, in milliseconds")
    latency_s: float = Field(ge=0, description="Total wall-clock time of the request, in seconds")
    tokens_per_second: float = Field(ge=0, description="Output tokens / total latency")
    input_cost_usd: float = Field(ge=0)
    output_cost_usd: float = Field(ge=0)
    total_cost_usd: float = Field(ge=0)


class StreamChunk(BaseModel):
    """One normalized piece of a streamed response, identical for every provider."""

    model_config = ConfigDict(frozen=True)

    delta_text: str = ""
    provider: Provider
    model: str
    is_final: bool = False
    telemetry: Optional[ChunkTelemetry] = None
