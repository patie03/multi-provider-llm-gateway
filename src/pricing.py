"""Static model catalog and cost calculation.

Prices are USD per 1,000,000 tokens (standard tier, no batch/caching discounts),
taken from the providers' official pricing pages on 2026-10-09.
Decimal is used internally so tiny per-request costs don't pick up float error.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

PRICING_VERIFIED_ON = "2026-10-09"
_PER_MILLION = Decimal(1_000_000)
_SIX_PLACES = Decimal("0.000001")


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    display_name: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    # Some models reject non-default sampling parameters with a 400 error.
    supports_temperature: bool = True
    # Optional long-prompt pricing tier (used when input tokens exceed the threshold).
    long_prompt_threshold: Optional[int] = None
    long_input_per_mtok: Optional[Decimal] = None
    long_output_per_mtok: Optional[Decimal] = None


MODELS: dict[str, ModelInfo] = {
    # OpenAI — https://developers.openai.com/api/docs/models
    "gpt-6-luna": ModelInfo(
        provider="openai",
        display_name="GPT-6 Luna",
        input_per_mtok=Decimal("0.10"),
        output_per_mtok=Decimal("0.50"),
        supports_temperature=False,  # reasoning model
        long_prompt_threshold=272_000,
        long_input_per_mtok=Decimal("0.20"),
        long_output_per_mtok=Decimal("0.75"),
    ),
    "gpt-6.1-sol": ModelInfo(
        provider="openai",
        display_name="GPT-6.1 Sol",
        input_per_mtok=Decimal("2.00"),
        output_per_mtok=Decimal("10.00"),
        supports_temperature=False,  # reasoning model
    ),
    # Anthropic — https://platform.claude.com/docs/en/models/overview
    "claude-haiku-5-5": ModelInfo(
        provider="anthropic",
        display_name="Claude Haiku 5.5",
        input_per_mtok=Decimal("0.10"),
        output_per_mtok=Decimal("0.50"),
        supports_temperature=False,  # docs: omit temperature/top_p/top_k
        long_prompt_threshold=100_000,
        long_input_per_mtok=Decimal("0.50"),
        long_output_per_mtok=Decimal("2.50"),
    ),
    "claude-sonnet-5-5": ModelInfo(
        provider="anthropic",
        display_name="Claude Sonnet 5.5",
        input_per_mtok=Decimal("2.00"),
        output_per_mtok=Decimal("10.00"),
        supports_temperature=False,  # docs: non-default temperature returns 400
    ),
}


def get_model_info(model: str) -> ModelInfo:
    try:
        return MODELS[model]
    except KeyError:
        raise KeyError(f"Unknown model '{model}'. Supported: {', '.join(MODELS)}") from None


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> tuple[float, float, float]:
    """Return (input_cost, output_cost, total_cost) in USD, rounded to 6 decimal places."""
    info = get_model_info(model)
    in_rate, out_rate = info.input_per_mtok, info.output_per_mtok
    if info.long_prompt_threshold is not None and input_tokens > info.long_prompt_threshold:
        in_rate = info.long_input_per_mtok or in_rate
        out_rate = info.long_output_per_mtok or out_rate

    input_cost = Decimal(input_tokens) * in_rate / _PER_MILLION
    output_cost = Decimal(output_tokens) * out_rate / _PER_MILLION
    total_cost = input_cost + output_cost

    def _round(value: Decimal) -> float:
        return float(value.quantize(_SIX_PLACES, rounding=ROUND_HALF_UP))

    return _round(input_cost), _round(output_cost), _round(total_cost)
