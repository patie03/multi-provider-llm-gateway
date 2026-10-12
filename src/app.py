"""Streamlit UI for the UnifiedLLMGateway.

Run locally:  uv run streamlit run src/app.py

Design note: Streamlit runs this script synchronously and re-runs it on every click.
The gateway is async and keeps pooled HTTP clients, so it lives on ONE long-running
asyncio event loop in a background thread (cached across reruns). Chunks are handed
to the UI thread through a queue, so the page updates token by token without freezing.
"""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gateway import GatewayError, UnifiedLLMGateway  # noqa: E402
from src.pricing import MODELS, PRICING_VERIFIED_ON  # noqa: E402

_DONE = object()
INVALID_DEMO_KEYS = {
    "openai": "sk-invalid-demo-key-0000000000",
    "anthropic": "sk-ant-invalid-demo-key-0000000000",
}

st.set_page_config(page_title="Multi-Provider LLM Gateway", page_icon="⚡", layout="wide")


# ------------------------------------------------------------------ runtime
def _load_cloud_secrets() -> None:
    """On Streamlit Cloud, keys come from the Secrets manager instead of a .env file."""
    try:
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            if name in st.secrets and not os.getenv(name):
                os.environ[name] = st.secrets[name]
    except Exception:  # no secrets.toml locally: that's fine, .env is used instead
        pass


@st.cache_resource
def get_runtime() -> tuple[asyncio.AbstractEventLoop, UnifiedLLMGateway]:
    _load_cloud_secrets()
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True, name="gateway-loop").start()

    async def build() -> UnifiedLLMGateway:
        return UnifiedLLMGateway()

    gateway = asyncio.run_coroutine_threadsafe(build(), loop).result()
    return loop, gateway


def start_stream(prompt: str, model: str, temperature: float, use_invalid_key: bool):
    """Start streaming on the background loop. Returns (queue, future)."""
    loop, gateway = get_runtime()
    out: queue.Queue = queue.Queue()

    async def pump() -> None:
        gw = gateway
        if use_invalid_key:
            provider = MODELS[model].provider
            gw = UnifiedLLMGateway(**{f"{provider}_api_key": INVALID_DEMO_KEYS[provider]})
        try:
            async for chunk in gw.stream(prompt, model, temperature):
                out.put(chunk)
        except Exception as exc:  # handed to the UI, which shows a friendly message
            out.put(exc)
        finally:
            if gw is not gateway:
                await gw.aclose()
            out.put(_DONE)

    return out, asyncio.run_coroutine_threadsafe(pump(), loop)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    provider_names = {"openai": "OpenAI", "anthropic": "Anthropic"}
    labels = {m: f"{provider_names[info.provider]} · {info.display_name}" for m, info in MODELS.items()}
    model = st.selectbox("Model", list(MODELS), format_func=labels.get)
    info = MODELS[model]
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.7, 0.1,
        disabled=not info.supports_temperature,
        help="Some current models only accept their default temperature.",
    )
    if not info.supports_temperature:
        st.caption(f"{info.display_name} uses its default temperature, so this slider is ignored.")

    st.markdown(
        f"**Pricing** (per 1M tokens)  \nInput \\${info.input_per_mtok} · Output \\${info.output_per_mtok}"
    )
    st.caption(f"Prices verified {PRICING_VERIFIED_ON}")

    st.divider()
    with st.expander("Error-handling demo"):
        use_invalid_key = st.toggle(
            "Send this request with an invalid API key",
            help="Shows how a rejected key is handled. Your real key is not changed.",
        )

# ------------------------------------------------------------------ main
st.title("⚡ Multi-Provider LLM Gateway")
st.caption("One async streaming interface over OpenAI and Anthropic, with live token and cost telemetry.")

prompt = st.text_area("Prompt", "Explain what an API gateway is in three short sentences.", height=110)
run = st.button("Run", type="primary", disabled=not prompt.strip())

answer_col, metrics_col = st.columns([3, 2], gap="large")
with answer_col:
    st.subheader("Response")
    response_box = st.empty()
with metrics_col:
    st.subheader("Telemetry")
    status_box = st.empty()
    row1 = st.columns(2)
    row2 = st.columns(2)
    ttft_box, tps_box = row1[0].empty(), row1[1].empty()
    in_box, out_box = row2[0].empty(), row2[1].empty()
    cost_box = st.empty()


def show_metrics(ttft="–", tps="–", tokens_in="–", tokens_out="–", cost="–") -> None:
    ttft_box.metric("Time to first token", ttft)
    tps_box.metric("Speed", tps)
    in_box.metric("Prompt tokens", tokens_in)
    out_box.metric("Completion tokens", tokens_out)
    cost_box.metric("Cost (USD)", cost)


show_metrics()

if run:
    out, future = start_stream(prompt, model, temperature, use_invalid_key)
    text, started, first_at, final = "", time.perf_counter(), None, None
    try:
        while True:
            try:
                item = out.get(timeout=0.1)
            except queue.Empty:
                status_box.info(f"Waiting for {info.display_name}… {time.perf_counter() - started:.1f}s")
                continue
            if item is _DONE:
                break
            if isinstance(item, Exception):
                name = type(item).__name__
                if isinstance(item, GatewayError):
                    status_box.error(f"**{name}:** {item}")
                else:
                    status_box.error(f"**{name}:** the request could not be completed.")
                break
            if item.is_final:
                final = item.telemetry
                continue
            if first_at is None:
                first_at = time.perf_counter()
            text += item.delta_text
            response_box.markdown(text + "▌")
            elapsed = time.perf_counter() - started
            status_box.info(f"Streaming from {info.display_name}… {elapsed:.1f}s")
            show_metrics(ttft=f"{(first_at - started) * 1000:.0f} ms", tokens_out="counting…")
    finally:
        future.cancel()  # if the user clicks away mid-stream, the gateway closes the HTTP stream

    response_box.markdown(text or "_No text returned._")
    if final is not None:
        status_box.success(f"Done in {final.latency_s:.2f}s")
        show_metrics(
            ttft=f"{final.ttft_ms:.0f} ms",
            tps=f"{final.tokens_per_second:.1f} tok/s",
            tokens_in=f"{final.input_tokens:,}",
            tokens_out=f"{final.output_tokens:,}",
            cost=f"${final.total_cost_usd:.6f}",
        )
        st.session_state.setdefault("history", []).insert(0, {
            "Model": labels[model],
            "TTFT (ms)": round(final.ttft_ms),
            "Tokens/s": final.tokens_per_second,
            "Prompt tok": final.input_tokens,
            "Completion tok": final.output_tokens,
            "Latency (s)": final.latency_s,
            "Cost (USD)": f"${final.total_cost_usd:.6f}",
        })

if st.session_state.get("history"):
    st.subheader("Run history")
    st.caption("Compare providers side by side: latency, throughput and cost.")
    st.dataframe(st.session_state["history"], hide_index=True, width="stretch")
