"""Emit a `model_usage` event for one model call.

There are three model call sites -- the planner, each skill's tool loop, and the
prose lead-in -- and only the skill loop was ever traced. Counting just that one
would undercount a run's cost by every planning and rendering call it makes.
All three now route through here.

`model_usage` is its own event rather than a field on `model_result` because
the planner and prose passes are not skills and have no skill card; the ledger
buckets them as __planner__ / __prose__.
"""

from __future__ import annotations

from typing import Any, Callable

from app.obs.usage import cost_usd, extract_usage

def emit_model_usage(
    writer: Callable[[dict], Any] | None,
    message: Any,
    *,
    role: str,
    model_hint: str = "",
    skill_id: str | None = None,
    latency_ms: int | None = None,
) -> None:
    """Best-effort: a provider that reports no usage, or any failure in here,
    must never break the run it is measuring."""
    if writer is None:
        return
    try:
        usage = extract_usage(message, fallback_model=model_hint)
        if usage is None:
            return
        writer({
            "type": "model_usage",
            "role": role,
            "skill_id": skill_id,
            "model": usage.model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": usage.cached_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": cost_usd(usage),
            "latency_ms": latency_ms,
        })
    except Exception:  # noqa: BLE001 -- telemetry must not break the pipeline
        return
