"""Cost accounting.

The invariant that matters most: an unpriced model reports None, never 0.0.
"We don't know what this cost" and "this was free" are different facts, and
collapsing them turns a missing price entry into a silently free query -- the
one failure mode a cost dashboard must never have.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.obs.ledger import PLANNER, LedgerStore, RunLedger
from app.obs.usage import Usage, cost_usd, extract_usage


# --- pricing ----------------------------------------------------------------

def test_published_rates_are_applied():
    u = Usage(model="gemini-3.8-flash", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd(u, at=date(2026, 9, 11)) == pytest.approx(0.75 + 3.75)


def test_rates_are_effective_dated():
    """Gemini Flash pricing doubles on 2027-01-01. A flat constant would quietly
    under-report every run from that day on."""
    u = Usage(model="gemini-3.8-flash", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd(u, at=date(2026, 12, 31)) == pytest.approx(4.50)
    assert cost_usd(u, at=date(2027, 1, 1)) == pytest.approx(9.00)


def test_cached_input_is_billed_at_the_cached_rate():
    """Gemini reports input_tokens as the TOTAL including cached ones, so the
    uncached portion is the difference. Billing all input at the full rate would
    overstate the cost of every cache hit -- exactly backwards from the signal
    you want when tuning caching."""
    u = Usage(model="gemini-3.8-flash", input_tokens=1_000_000, cached_tokens=1_000_000)
    assert cost_usd(u, at=date(2026, 9, 11)) == pytest.approx(0.075)

    half = Usage(model="gemini-3.8-flash", input_tokens=1_000_000, cached_tokens=500_000)
    assert cost_usd(half, at=date(2026, 9, 11)) == pytest.approx(0.375 + 0.0375)


def test_an_unpriced_model_is_none_not_zero():
    assert cost_usd(Usage(model="some-new-model", input_tokens=5000)) is None


def test_an_unpriced_model_reports_no_cost_rather_than_zero():
    """A Databricks model nobody priced yet must read as unknown, not free."""
    assert cost_usd(Usage(model="databricks-some-new-model", input_tokens=5000)) is None


def test_versioned_model_ids_price_off_their_base():
    u = Usage(model="gemini-3.8-flash-001", input_tokens=1_000_000)
    assert cost_usd(u, at=date(2026, 9, 11)) == pytest.approx(0.75)


# --- usage extraction -------------------------------------------------------

class _Msg:
    def __init__(self, usage, model="gemini-3.8-flash"):
        self.usage_metadata = usage
        self.response_metadata = {"model_name": model}


def test_extract_reads_the_langchain_shape():
    u = extract_usage(_Msg({
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
        "input_token_details": {"cache_read": 40},
    }))
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (100, 20, 40)


def test_a_silent_provider_yields_none_not_zeros():
    """None means 'unreported'. A zeroed Usage would look like a free call."""
    assert extract_usage(_Msg(None)) is None


# --- ledger -----------------------------------------------------------------

def _usage_event(unit, inp, out, cost, cached=0):
    return "model_usage", {
        "skill_id": unit, "model": "gemini-3.8-flash", "input_tokens": inp,
        "output_tokens": out, "cached_tokens": cached, "cost_usd": cost,
    }


def test_ledger_rolls_up_across_units():
    led = RunLedger()
    led.observe(*_usage_event("skill_a", 1000, 100, 0.001))
    led.observe(*_usage_event("skill_a", 2000, 200, 0.002))
    led.observe(*_usage_event("skill_b", 500, 50, 0.0005))

    s = led.summary()
    assert s["input_tokens"] == 3500
    assert s["output_tokens"] == 350
    assert s["model_calls"] == 3
    assert s["cost_usd"] == pytest.approx(0.0035)
    assert s["by_unit"]["skill_a"]["model_calls"] == 2


def test_one_unpriced_call_makes_the_whole_run_unpriced():
    """A partial total would read as authoritative while being wrong."""
    led = RunLedger()
    led.observe(*_usage_event("skill_a", 1000, 100, 0.001))
    led.observe(*_usage_event("skill_a", 1000, 100, None))
    assert led.total_cost is None
    assert led.priced is False


def test_planner_and_prose_are_bucketed_outside_skills():
    """They are real cost but have no skill card. Counting only skills would
    under-report every run by its planning and rendering calls."""
    led = RunLedger()
    led.observe("model_usage", {
        "role": PLANNER, "skill_id": None, "model": "gemini-3.8-flash",
        "input_tokens": 2000, "output_tokens": 90, "cost_usd": 0.002,
    })
    assert PLANNER in led.summary()["by_unit"]


def test_ledger_tracks_quality_signals():
    led = RunLedger()
    led.observe("validate_result", {"ok": False, "problems": ["nope"], "retry_count": 2})
    led.observe("answer_ready", {"degraded": True, "skills_used": ["a", "b"]})
    assert led.retry_count == 2
    assert led.status() == "degraded"
    assert led.validation_problems == ["nope"]


def test_tool_errors_are_counted():
    led = RunLedger()
    led.observe("tool_call", {"skill_id": "s"})
    led.observe("tool_result", {"skill_id": "s", "ok": False})
    led.observe("tool_call", {"skill_id": "s"})
    led.observe("tool_result", {"skill_id": "s", "ok": True})
    s = led.summary()
    assert s["tool_calls"] == 2 and s["tool_errors"] == 1


def test_an_unfinished_run_is_incomplete_not_ok():
    """Client disconnect still cost money; it must not be filed as a success."""
    led = RunLedger()
    led.observe(*_usage_event("skill_a", 1000, 100, 0.001))
    assert led.status() == "incomplete"


def test_unknown_events_are_ignored():
    """The ledger must never be the reason a run fails."""
    RunLedger().observe("something_new", {"whatever": 1})


# --- persistence ------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_round_trips_a_run(tmp_path):
    store = LedgerStore(tmp_path / "obs.sqlite")
    led = RunLedger()
    led.observe("run_start", {
        "thread_id": "t1", "request_id": "r1", "question": "q?", "agent_id": "general",
    })
    led.observe(*_usage_event("skill_a", 1000, 100, 0.001))
    led.observe("answer_ready", {"degraded": False, "skills_used": ["skill_a"]})
    await store.save(led)

    rows = await store.recent()
    assert len(rows) == 1 and rows[0]["thread_id"] == "t1"
    assert rows[0]["status"] == "ok"

    rollup = await store.rollup()
    assert rollup["totals"]["runs"] == 1
    assert rollup["by_unit"][0]["unit"] == "skill_a"


@pytest.mark.asyncio
async def test_saving_the_same_thread_twice_does_not_duplicate(tmp_path):
    """A resumed run saves again under the same thread id."""
    store = LedgerStore(tmp_path / "obs.sqlite")
    led = RunLedger()
    led.observe("run_start", {"thread_id": "t1", "question": "q", "agent_id": "general"})
    await store.save(led)
    await store.save(led)
    assert len(await store.recent()) == 1
