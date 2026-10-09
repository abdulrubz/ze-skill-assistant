#!/usr/bin/env python
"""Query-coverage eval: runs the PLANNER ONLY over the question bank and
checks it routes each question to a sensible skill.

Planner-only is deliberate -- it is one model call per question (a few
seconds) rather than a full tool-calling run (a minute), so the whole bank
runs in minutes and can be re-run after every manifest edit. Routing is also
where breadth actually lives: if the planner can place a question, the skill
behind it has the tools and playbook to answer it.

  python scripts/eval_coverage.py
  python scripts/eval_coverage.py --agent customer_directory
  python scripts/eval_coverage.py --filter spend
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.manifest import load_all_agent_manifests  # noqa: E402
from app.deps import Caller  # noqa: E402
from app.graph.nodes.plan import plan as plan_node  # noqa: E402
from app.skills.registry import load_and_validate  # noqa: E402

from app.paths import EVALS_DIR  # noqa: E402

BANKS = {
    "default": EVALS_DIR / "routing_bank.yaml",
}
GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="general")
    ap.add_argument("--filter", default=None, help="only questions containing this substring")
    ap.add_argument("--bank", default="default", choices=sorted(BANKS), help="which question bank")
    args = ap.parse_args()

    load_and_validate()
    agent = load_all_agent_manifests()[args.agent]
    caller = Caller(agent_id=agent.id)

    questions = yaml.safe_load(BANKS[args.bank].read_text())
    if args.filter:
        questions = [q for q in questions if args.filter.lower() in q["q"].lower()]

    passed = failed = 0
    failures: list[str] = []
    started = time.monotonic()

    for i, case in enumerate(questions, 1):
        state = {
            "question": case["q"], "agent": agent, "caller": caller,
            "resolved": None, "retry_count": 0, "last_validation_problems": [],
        }
        try:
            out = await plan_node(state)  # type: ignore[arg-type]
            p = out["plan"]
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append(f"{case['q']}\n    EXCEPTION: {exc}")
            print(f"{RED}[{i:2}/{len(questions)}] ERROR{RESET} {case['q'][:70]}")
            continue

        chosen = [s["skill_id"] for s in p["steps"]]
        expected = case["expect"]

        if p["unanswerable"] or not chosen:
            ok = "UNANSWERABLE" in expected
            label = "declined"
        else:
            ok = any(c in expected for c in chosen)
            label = "+".join(chosen)

        if ok:
            passed += 1
            print(f"{GREEN}[{i:2}/{len(questions)}] ok  {RESET}{case['q'][:62]:64} {DIM}-> {label}{RESET}")
        else:
            failed += 1
            failures.append(f"{case['q']}\n    got {label}, expected one of {expected}")
            print(f"{RED}[{i:2}/{len(questions)}] MISS{RESET} {case['q'][:62]:64} {YELLOW}-> {label}{RESET} "
                  f"{DIM}(want {'/'.join(expected)}){RESET}")

    elapsed = time.monotonic() - started
    total = passed + failed
    pct = (passed / total * 100) if total else 0
    print(f"\n{BOLD}coverage: {passed}/{total} ({pct:.0f}%){RESET} in {elapsed:.0f}s "
          f"{DIM}({elapsed/max(total,1):.1f}s per question){RESET}")
    if failures:
        print(f"\n{BOLD}misses:{RESET}")
        for f in failures:
            print(f"  - {f}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
