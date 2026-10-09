#!/usr/bin/env python
"""End-to-end cost AND quality harness.

`eval_coverage.py` measures planner routing -- one model call per question, no
tools, no skill loops. That is 3.5% of a run's tokens. This runs the whole
pipeline so the numbers cover the part that actually costs money, and records
enough quality signal alongside to tell a real saving apart from a regression
that happens to be cheaper.

    python scripts/eval_cost.py --out baseline.json
    python scripts/eval_cost.py --out after-phase1.json --compare baseline.json
    python scripts/eval_cost.py --compare baseline.json --filter cohort

⚠️ This spends real API credit. Use --filter while iterating.

Runs go through `stream_ui_events`, the same path POST /ask uses, and land in
the same `var/observability.sqlite` the API writes to. That is deliberate: a
baseline captured here is directly comparable with the rows the system already
produced in normal use.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

import yaml
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.manifest import load_all_agent_manifests  # noqa: E402
from app.api.adapter import stream_ui_events  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.deps import Caller, Deps  # noqa: E402
from app.graph.build import build_graph, initial_state  # noqa: E402
from app.obs.ledger import LedgerStore  # noqa: E402
from app.skills.registry import load_and_validate, get_skill  # noqa: E402

from app.paths import EVALS_DIR  # noqa: E402

BANK = EVALS_DIR / "answer_bank.yaml"

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
GREEN, RED, YELLOW = "\033[32m", "\033[31m", "\033[33m"


async def run_one(graph, deps, agent, question: str, ledger_store, expect: list[str] | None = None,
                  extra_config: dict | None = None) -> dict:
    """One end-to-end run, reduced to the metrics a phase is accepted on."""
    thread_id = str(uuid.uuid4())
    state = initial_state(
        request_id=str(uuid.uuid4()), thread_id=thread_id,
        question=question, agent=agent, caller=deps.caller,
    )
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": deps.settings.recursion_limit,
        **(extra_config or {}),
    }

    cost: dict = {}
    skills: list[dict] = []
    problems: list[str] = []
    degraded = False
    answered = False
    facts = 0
    answer_blob = ""
    answer_payload: dict = {}
    started = time.monotonic()

    async for ev in stream_ui_events(
        graph, state, context=deps, config=config, ledger_store=ledger_store
    ):
        d = ev.data
        if ev.type == "run_cost":
            cost = d
        elif ev.type == "skill_done":
            skills.append({
                "skill_id": d["skill_id"],
                "status": d["status"],
                "steps_used": d.get("steps_used", 0),
                "facts": len(d.get("facts") or []),
            })
        elif ev.type == "validate_result" and not d.get("ok"):
            problems.extend(d.get("problems") or [])
        elif ev.type == "answer_ready":
            answered = True
            degraded = bool(d.get("degraded"))
            facts = len(d.get("facts") or [])
            # Rendered text AND cited fact values: a correct answer may state a
            # value only through a citation marker.
            answer_blob = json.dumps(d, default=str).lower()
            answer_payload = d

    # Budget exhaustion is the quality signal the ledger does not carry, and it
    # is the one Phase 3 is supposed to move: a skill that spends its whole tool
    # budget was cut off mid-investigation and answered from partial data.
    for s in skills:
        try:
            s["max_tool_steps"] = get_skill(s["skill_id"]).manifest.max_tool_steps
        except KeyError:
            s["max_tool_steps"] = None
        s["exhausted"] = (
            s["max_tool_steps"] is not None and s["steps_used"] >= s["max_tool_steps"]
        )

    by_unit = cost.get("by_unit") or {}
    return {
        "question": question,
        "thread_id": thread_id,
        "answer_payload": answer_payload,
        "wall_ms": round((time.monotonic() - started) * 1000),
        "answered": answered,
        "degraded": degraded,
        "facts": facts,
        "validation_problems": problems,
        "skills": skills,
        "skills_used": [s["skill_id"] for s in skills],
        "exhausted_skills": [s["skill_id"] for s in skills if s["exhausted"]],
        "input_tokens": cost.get("input_tokens", 0),
        "output_tokens": cost.get("output_tokens", 0),
        "cached_tokens": cost.get("cached_tokens", 0),
        "model_calls": cost.get("model_calls", 0),
        "tool_calls": cost.get("tool_calls", 0),
        "tool_errors": cost.get("tool_errors", 0),
        "cost_usd": cost.get("cost_usd"),
        "by_unit": by_unit,
        # Optional ground truth from the bank: substrings the answer must contain.
        # Crude on purpose -- it catches a wrong or missing fact, not style.
        "expect": list(expect or []),
        "expect_missing": [e for e in (expect or []) if e.lower() not in answer_blob],
    }


def aggregate(runs: list[dict]) -> dict:
    priced = [r["cost_usd"] for r in runs if r.get("cost_usd") is not None]
    inp = sum(r["input_tokens"] for r in runs)
    return {
        "runs": len(runs),
        "cost_usd": round(sum(priced), 6) if priced else None,
        "unpriced_runs": len(runs) - len(priced),
        "input_tokens": inp,
        "output_tokens": sum(r["output_tokens"] for r in runs),
        "cached_tokens": sum(r["cached_tokens"] for r in runs),
        "cached_pct": round(100 * sum(r["cached_tokens"] for r in runs) / inp, 2) if inp else 0.0,
        "model_calls": sum(r["model_calls"] for r in runs),
        "tool_calls": sum(r["tool_calls"] for r in runs),
        "tool_errors": sum(r["tool_errors"] for r in runs),
        "answered": sum(1 for r in runs if r["answered"]),
        "degraded": sum(1 for r in runs if r["degraded"]),
        "with_validation_problems": sum(1 for r in runs if r["validation_problems"]),
        "exhausted_skill_runs": sum(1 for r in runs if r["exhausted_skills"]),
        "expect_checks": sum(len(r.get("expect", [])) for r in runs),
        "expect_passed": sum(len(r.get("expect", [])) - len(r.get("expect_missing", [])) for r in runs),
        "median_wall_ms": round(statistics.median([r["wall_ms"] for r in runs])) if runs else 0,
    }


def _fmt(v, width=10, money=False):
    if v is None:
        return "n/a".rjust(width)
    return (f"${v:.4f}" if money else f"{v:,}").rjust(width)


def print_report(snapshot: dict) -> None:
    agg = snapshot["totals"]
    print(f"\n{BOLD}per question{RESET}")
    print(f"  {'cost':>9} {'in':>8} {'cached':>7} {'out':>6} {'calls':>6} {'tools':>6} "
          f"{'err':>4}  question")
    for r in snapshot["runs"]:
        flags = ""
        if r["degraded"]:
            flags += f" {YELLOW}degraded{RESET}"
        if r.get("expect_missing"):
            flags += f" {RED}missing {r['expect_missing']}{RESET}"
        if r["validation_problems"]:
            flags += f" {RED}{len(r['validation_problems'])} validation{RESET}"
        if r["exhausted_skills"]:
            flags += f" {YELLOW}budget:{','.join(r['exhausted_skills'])}{RESET}"
        cost = f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "n/a"
        print(f"  {cost:>9} {r['input_tokens']:>8,} {r['cached_tokens']:>7,} "
              f"{r['output_tokens']:>6,} {r['model_calls']:>6} {r['tool_calls']:>6} "
              f"{r['tool_errors']:>4}  {r['question'][:52]}{flags}")

    print(f"\n{BOLD}totals{RESET}")
    for k in ("cost_usd", "input_tokens", "cached_tokens", "cached_pct", "output_tokens",
              "model_calls", "tool_calls", "tool_errors", "answered", "degraded",
              "with_validation_problems", "exhausted_skill_runs", "expect_checks", "expect_passed",
              "median_wall_ms"):
        print(f"  {k:<26} {agg.get(k)}")


def print_comparison(before: dict, after: dict) -> None:
    b, a = before["totals"], after["totals"]
    print(f"\n{BOLD}vs baseline{RESET}  ({before.get('label','baseline')} -> {after.get('label','current')})")
    print(f"  {'metric':<26}{'before':>12}{'after':>12}{'delta':>12}")

    def row(key, money=False, lower_is_better=True):
        bv, av = b.get(key), a.get(key)
        if bv is None or av is None:
            print(f"  {key:<26}{_fmt(bv,12,money)}{_fmt(av,12,money)}{'n/a':>12}")
            return
        delta = av - bv
        pct = (100 * delta / bv) if bv else 0.0
        good = (delta <= 0) if lower_is_better else (delta >= 0)
        color = GREEN if good else RED
        if delta == 0:
            color = DIM
        shown = f"{pct:+.1f}%" if bv else f"{delta:+}"
        print(f"  {key:<26}{_fmt(bv,12,money)}{_fmt(av,12,money)}{color}{shown:>12}{RESET}")

    row("cost_usd", money=True)
    row("input_tokens")
    row("output_tokens")
    row("cached_tokens", lower_is_better=False)
    row("model_calls")
    row("tool_calls")
    row("tool_errors")
    row("median_wall_ms")
    print(f"  {DIM}--- quality gates (must not regress) ---{RESET}")
    row("answered", lower_is_better=False)
    row("degraded")
    row("with_validation_problems")
    if before["totals"].get("expect_checks") or after["totals"].get("expect_checks"):
        row("expect_passed", lower_is_better=False)
    row("exhausted_skill_runs")

    # Routing drift: a cost change must not quietly become a routing change.
    bmap = {r["question"]: set(r["skills_used"]) for r in before["runs"]}
    drift = [
        (r["question"], bmap[r["question"]], set(r["skills_used"]))
        for r in after["runs"]
        if r["question"] in bmap and bmap[r["question"]] != set(r["skills_used"])
    ]
    if drift:
        print(f"\n  {YELLOW}routing drift on {len(drift)} question(s){RESET}")
        for q, was, now in drift:
            print(f"    {DIM}{q[:60]}{RESET}\n      was {sorted(was)} -> now {sorted(now)}")
    else:
        print(f"\n  {GREEN}no routing drift{RESET}")

    gates = {
        "answered": a["answered"] >= b["answered"],
        "degraded": a["degraded"] <= b["degraded"],
        "validation": a["with_validation_problems"] <= b["with_validation_problems"],
        "facts": a.get("expect_passed", 0) >= b.get("expect_passed", 0),
        "cost": (a["cost_usd"] or 0) <= (b["cost_usd"] or 0),
    }
    failed = [k for k, ok in gates.items() if not ok]
    if failed:
        print(f"\n{RED}GATES FAILED: {', '.join(failed)}{RESET}")
    else:
        print(f"\n{GREEN}all gates passed{RESET}")
    return not failed


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="general")
    ap.add_argument("--bank", default=str(BANK))
    ap.add_argument("--filter", help="substring: only run matching questions")
    ap.add_argument("--out", help="write the snapshot JSON here")
    ap.add_argument("--compare", help="baseline JSON to diff against")
    ap.add_argument("--label", default="", help="name this snapshot, e.g. 'phase-1'")
    ap.add_argument(
        "--repeat", type=int, default=1,
        help="run each question N times and aggregate. The SAME question varies "
             "1.4x-4.5x in input tokens between runs on this dataset (measured "
             "across the ledger), so a single pass CANNOT distinguish a 20-30%% "
             "change from noise. Use >=3 before believing any end-to-end delta.",
    )
    args = ap.parse_args()

    bank = yaml.safe_load(Path(args.bank).read_text())["questions"]
    if args.filter:
        bank = [e for e in bank if args.filter.lower() in e["q"].lower()]
    if not bank:
        print("no questions matched --filter")
        return 2

    # Every tool is generated from the service specs. Before load_and_validate.
    from app.tools.openapi.server import install_from_settings
    tool_stack = AsyncExitStack()
    await install_from_settings(tool_stack)

    load_and_validate()
    agents = load_all_agent_manifests()
    agent = agents[args.agent]

    settings = get_settings()
    if not settings.ze_dev_token:
        print(f"{RED}set ZE_DEV_TOKEN{RESET}: the runs call the services, which need a user token")
        return 2
    deps = Deps(
        settings=settings,
        caller=Caller(agent_id=agent.id, tool_deny=tuple(agent.scope.tool_deny)),
        user_token=settings.ze_dev_token,
    )

    ledger_store = LedgerStore(settings.obs_sqlite_path)

    print(f"{BOLD}{len(bank)} question(s){RESET} | agent={args.agent} | "
          f"skill_endpoint={settings.endpoint_for('skill')}")
    runs = []
    try:
        # Same checkpointer the API uses.
        async with AsyncSqliteSaver.from_conn_string(
            settings.checkpoint_sqlite_path
        ) as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            for i, entry in enumerate(bank, 1):
                print(f"{DIM}[{i}/{len(bank)}]{RESET} {entry['q'][:70]}", flush=True)
                for rep in range(args.repeat):
                    # Sequential on purpose: concurrent runs make latency and
                    # provider-side cache numbers uninterpretable.
                    r = await run_one(graph, deps, agent, entry["q"], ledger_store,
                                      expect=entry.get("expect"))
                    r["expected_skills"] = entry.get("skills", [])
                    r["repeat"] = rep
                    runs.append(r)
                    cost = f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "unpriced"
                    tag = f" {DIM}rep {rep + 1}/{args.repeat}{RESET}" if args.repeat > 1 else ""
                    print(f"      {cost} | {r['input_tokens']:,} in "
                          f"({r['cached_tokens']:,} cached) | {r['model_calls']} calls{tag}",
                          flush=True)
    finally:
        await tool_stack.aclose()

    snapshot = {
        "label": args.label or "current",
        "skill_endpoint": settings.endpoint_for("skill"),
        "planner_endpoint": settings.endpoint_for("planner"),
        "bank": str(Path(args.bank).name),
        "repeat": args.repeat,
        "totals": aggregate(runs),
        "runs": runs,
    }
    print_report(snapshot)

    if args.out:
        Path(args.out).write_text(json.dumps(snapshot, indent=2))
        print(f"\n{DIM}wrote {args.out}{RESET}")

    ok = True
    if args.compare:
        baseline = json.loads(Path(args.compare).read_text())
        ok = print_comparison(baseline, snapshot)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
