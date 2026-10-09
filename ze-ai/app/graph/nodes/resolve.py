"""Resolve: deterministic extraction from the question text, no model.

Fills `business_date` ("today": AS_OF_DATE when pinned, else the real date),
`time_window` from common relative phrases, and every id pattern the domain
declares in domain/vocabulary.yaml `resolve`. The planner and skills receive
these as known context instead of re-deriving them."""

from __future__ import annotations

import re
from datetime import date, timedelta

from app.config import get_settings
from app.graph.state import PipelineState
from app.vocabulary import vocabulary

_WINDOWS = [
    ("last 7 days", 7), ("last week", 7), ("last 30 days", 30), ("this month", 30),
    ("last month", 60), ("last 90 days", 90), ("last quarter", 90), ("last 3 months", 90),
    ("last 6 months", 180), ("last 12 months", 365), ("this year", 365), ("last year", 365),
]


def today() -> date:
    pinned = get_settings().as_of_date
    return date.fromisoformat(pinned) if pinned else date.today()


def extract(question: str) -> dict:
    anchor = today()
    resolved: dict = {"business_date": anchor.isoformat(), "time_window": None}
    lowered = question.lower()
    for phrase, days in _WINDOWS:
        if phrase in lowered:
            resolved["time_window"] = {"start": (anchor - timedelta(days=days)).isoformat(),
                                       "end": anchor.isoformat()}
            break
    for slot, spec in vocabulary().resolve.items():
        found = [m.group(1) if m.groups() else m.group(0) for m in re.finditer(spec.pattern, question)]
        if spec.upper:
            found = [f.upper() for f in found]
        found = list(dict.fromkeys(found))
        resolved[slot] = found if spec.many else (found[0] if found else None)
    return resolved


async def resolve(state: PipelineState) -> dict:
    return {"resolved": extract(state["question"])}
