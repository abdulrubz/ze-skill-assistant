"""Per-run cost and quality ledger.

Built by observing the UIEvent stream rather than by instrumenting the graph a
second time. Every model call, tool call and validation result already flows
through `stream_ui_events`; the ledger just keeps score. That keeps the
"adapter is the only framework-coupled file" property intact -- the ledger
speaks UIEvent, not LangGraph.

It records what a generic LLM dashboard cannot know about this system: which
groundedness check failed, whether the run replanned, whether the answer shipped
degraded, and how much of the tool-step budget each skill actually used.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.obs.usage import Usage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    thread_id        TEXT PRIMARY KEY,
    request_id       TEXT,
    question         TEXT,
    agent_id         TEXT,
    started_at       TEXT,
    finished_at      TEXT,
    duration_ms      INTEGER,
    status           TEXT,      -- ok | refused | error | incomplete
    degraded         INTEGER,
    retry_count      INTEGER,
    validation_ok    INTEGER,
    validation_problems TEXT,   -- json array
    skills_used      TEXT,      -- json array
    model_calls      INTEGER,
    tool_calls       INTEGER,
    tool_errors      INTEGER,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    cached_tokens    INTEGER,
    reasoning_tokens INTEGER,
    cost_usd         REAL,      -- NULL when any model in the run is unpriced
    priced           INTEGER
);

CREATE TABLE IF NOT EXISTS run_units (
    thread_id     TEXT,
    unit          TEXT,        -- skill id, or __planner__ / __prose__
    model         TEXT,
    model_calls   INTEGER,
    tool_calls    INTEGER,
    tool_errors   INTEGER,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    cost_usd      REAL,
    PRIMARY KEY (thread_id, unit)
);

CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
"""

PLANNER = "__planner__"
PROSE = "__prose__"


@dataclass
class _Unit:
    """One cost-bearing participant: a skill, the planner, or the prose pass."""

    model: str = ""
    model_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    usage: Usage = field(default_factory=lambda: Usage(model=""))
    cost: float | None = 0.0
    priced: bool = True

    def add_usage(self, usage: Usage, line_cost: float | None) -> None:
        """`line_cost` comes from the event, priced at emit time -- it is NOT
        recomputed here. Recomputing would silently re-price an unpriced call as
        soon as a rate was added to pricing.yaml, and would let the ledger
        disagree with the per-skill figure the UI already showed the user.
        A None line cost poisons the total on purpose."""
        self.model = usage.model or self.model
        self.usage = self.usage + usage if self.usage.model else usage
        self.model_calls += 1
        if line_cost is None:
            self.priced = False
            self.cost = None
        elif self.cost is not None:
            self.cost = round(self.cost + line_cost, 8)


@dataclass
class RunLedger:
    thread_id: str = ""
    request_id: str = ""
    question: str = ""
    agent_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _t0: float = field(default_factory=time.monotonic)

    units: dict[str, _Unit] = field(default_factory=lambda: defaultdict(_Unit))
    retry_count: int = 0
    validation_ok: bool | None = None
    validation_problems: list[str] = field(default_factory=list)
    degraded: bool | None = None
    skills_used: list[str] = field(default_factory=list)
    answered: bool = False

    def observe(self, event_type: str, data: dict[str, Any]) -> None:
        """Feed one UIEvent. Unknown events are ignored on purpose -- the ledger
        must never be the reason a run fails."""
        if event_type == "run_start":
            self.thread_id = data.get("thread_id", self.thread_id)
            self.request_id = data.get("request_id", "")
            self.question = data.get("question", "")
            self.agent_id = data.get("agent_id", "")

        elif event_type == "model_usage":
            unit = data.get("skill_id") or data.get("role") or PLANNER
            self.units[unit].add_usage(
                Usage(
                    model=data.get("model", ""),
                    input_tokens=int(data.get("input_tokens") or 0),
                    output_tokens=int(data.get("output_tokens") or 0),
                    cached_tokens=int(data.get("cached_tokens") or 0),
                    reasoning_tokens=int(data.get("reasoning_tokens") or 0),
                ),
                data.get("cost_usd"),
            )

        elif event_type == "tool_call":
            self.units[data.get("skill_id") or PLANNER].tool_calls += 1

        elif event_type == "tool_result":
            if not data.get("ok", True):
                self.units[data.get("skill_id") or PLANNER].tool_errors += 1

        elif event_type == "validate_result":
            self.validation_ok = bool(data.get("ok"))
            self.retry_count = max(self.retry_count, int(data.get("retry_count") or 0))
            self.validation_problems = list(data.get("problems") or [])

        elif event_type == "answer_ready":
            self.answered = True
            self.degraded = bool(data.get("degraded"))
            self.skills_used = list(data.get("skills_used") or [])

    # --- rollups -----------------------------------------------------------

    @property
    def total_usage(self) -> Usage:
        total = Usage(model="")
        for unit in self.units.values():
            if unit.usage.model:
                total = total + unit.usage if total.model else unit.usage
        return total

    @property
    def priced(self) -> bool:
        return all(u.priced for u in self.units.values())

    @property
    def total_cost(self) -> float | None:
        if not self.priced:
            return None
        return round(sum(u.cost or 0.0 for u in self.units.values()), 8)

    def summary(self) -> dict:
        """The payload for the `run_cost` event and the UI."""
        usage = self.total_usage
        return {
            "thread_id": self.thread_id,
            "duration_ms": round((time.monotonic() - self._t0) * 1000),
            "model_calls": sum(u.model_calls for u in self.units.values()),
            "tool_calls": sum(u.tool_calls for u in self.units.values()),
            "tool_errors": sum(u.tool_errors for u in self.units.values()),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": usage.cached_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": self.total_cost,
            "priced": self.priced,
            "retry_count": self.retry_count,
            "degraded": self.degraded,
            "by_unit": {
                name: {
                    "model": u.model,
                    "model_calls": u.model_calls,
                    "tool_calls": u.tool_calls,
                    "input_tokens": u.usage.input_tokens,
                    "output_tokens": u.usage.output_tokens,
                    "cached_tokens": u.usage.cached_tokens,
                    "cost_usd": u.cost,
                }
                for name, u in sorted(self.units.items())
            },
        }

    def status(self) -> str:
        if not self.answered:
            return "incomplete"
        if self.degraded:
            return "degraded"
        return "ok"


class LedgerStore:
    """SQLite sink. Separate file from the checkpointer: this outlives any one
    run and is queried analytically, while checkpoints are hot run state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    async def save(self, ledger: RunLedger) -> None:
        await asyncio.to_thread(self._save_sync, ledger)

    def _save_sync(self, ledger: RunLedger) -> None:
        if not ledger.thread_id:
            return
        s = ledger.summary()
        usage = ledger.total_usage
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ledger.thread_id, ledger.request_id, ledger.question, ledger.agent_id,
                    ledger.started_at.isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                    s["duration_ms"], ledger.status(),
                    int(bool(ledger.degraded)), ledger.retry_count,
                    None if ledger.validation_ok is None else int(ledger.validation_ok),
                    json.dumps(ledger.validation_problems),
                    json.dumps(ledger.skills_used),
                    s["model_calls"], s["tool_calls"], s["tool_errors"],
                    usage.input_tokens, usage.output_tokens,
                    usage.cached_tokens, usage.reasoning_tokens,
                    ledger.total_cost, int(ledger.priced),
                ),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO run_units VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (ledger.thread_id, name, u.model, u.model_calls, u.tool_calls,
                     u.tool_errors, u.usage.input_tokens, u.usage.output_tokens,
                     u.usage.cached_tokens, u.cost)
                    for name, u in ledger.units.items()
                ],
            )

    async def recent(self, limit: int = 50) -> list[dict]:
        return await asyncio.to_thread(self._recent_sync, limit)

    def _recent_sync(self, limit: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    async def rollup(self) -> dict:
        return await asyncio.to_thread(self._rollup_sync)

    def _rollup_sync(self) -> dict:
        with self._connect() as conn:
            totals = dict(conn.execute(
                "SELECT COUNT(*) runs, COALESCE(SUM(cost_usd),0) cost, "
                "COALESCE(SUM(input_tokens+output_tokens),0) tokens, "
                "COALESCE(SUM(cached_tokens),0) cached, "
                "COALESCE(AVG(duration_ms),0) avg_ms, "
                "COALESCE(SUM(retry_count>0),0) replanned, "
                "COALESCE(SUM(degraded),0) degraded, "
                "COALESCE(SUM(tool_errors),0) tool_errors "
                "FROM runs"
            ).fetchone())
            by_unit = [dict(r) for r in conn.execute(
                "SELECT unit, COUNT(*) runs, COALESCE(SUM(cost_usd),0) cost, "
                "COALESCE(SUM(input_tokens+output_tokens),0) tokens "
                "FROM run_units GROUP BY unit ORDER BY cost DESC"
            ).fetchall()]
        return {"totals": totals, "by_unit": by_unit}
