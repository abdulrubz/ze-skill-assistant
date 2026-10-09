"""Token accounting and pricing.

One rule holds everything together: an unpriced model reports `cost_usd = None`,
never 0.0. "We don't know what this cost" and "this was free" are different
facts, and collapsing them turns a missing price into a silently free query.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PRICING_FILE = Path(__file__).parent / "pricing.yaml"


@dataclass(frozen=True)
class Usage:
    """What one model call consumed."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0      # subset of input_tokens that was served from cache
    reasoning_tokens: int = 0   # subset of output_tokens spent thinking

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            model=self.model if self.model == other.model else "mixed",
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


def extract_usage(message: Any, fallback_model: str = "") -> Usage | None:
    """Pull token counts off a LangChain AIMessage.

    Returns None rather than a zeroed Usage when the provider reported nothing,
    so "the provider is silent" stays distinguishable from "this call was free".
    """
    meta = getattr(message, "usage_metadata", None)
    if not meta:
        return None

    model = (getattr(message, "response_metadata", None) or {}).get("model_name") or fallback_model
    input_details = meta.get("input_token_details") or {}
    output_details = meta.get("output_token_details") or {}
    return Usage(
        model=model,
        input_tokens=int(meta.get("input_tokens") or 0),
        output_tokens=int(meta.get("output_tokens") or 0),
        cached_tokens=int(input_details.get("cache_read") or 0),
        reasoning_tokens=int(output_details.get("reasoning") or 0),
    )


@lru_cache
def _rate_table() -> dict[str, list[dict]]:
    data = yaml.safe_load(_PRICING_FILE.read_text()) or {}
    table: dict[str, list[dict]] = {}
    for model, tiers in (data.get("models") or {}).items():
        table[model] = sorted(tiers, key=lambda t: str(t["effective_from"]))
    return table


def _rates_for(model: str, at: date) -> dict | None:
    """The tier in force on `at`. Matches the exact model id, then any alias
    that is a prefix of it -- `gemini-3.8-flash-001` prices as `gemini-3.8-flash`."""
    table = _rate_table()
    tiers = table.get(model)
    if tiers is None:
        candidates = [k for k in table if model.startswith(k)]
        if not candidates:
            return None
        tiers = table[max(candidates, key=len)]

    applicable = [t for t in tiers if _as_date(t["effective_from"]) <= at]
    return applicable[-1] if applicable else None


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def cost_usd(usage: Usage, *, at: date | None = None) -> float | None:
    """Cost of one usage record, or None when the model has no published rate.

    `at` defaults to the REAL date: providers bill on the actual calendar,
    never on a pinned AS_OF_DATE.

    Cached input is billed at its own (cheaper) rate where the provider reports
    it: `input_tokens` is the TOTAL including cached ones, so the uncached
    portion is the difference.
    """
    rates = _rates_for(usage.model, at or date.today())
    if rates is None:
        return None

    uncached = max(usage.input_tokens - usage.cached_tokens, 0)
    total = (
        uncached * float(rates["input"])
        + usage.cached_tokens * float(rates.get("cached_input", rates["input"]))
        + usage.output_tokens * float(rates["output"])
    ) / 1_000_000
    return round(total, 8)


def is_priced(model: str, *, at: date | None = None) -> bool:
    return _rates_for(model, at or date.today()) is not None
