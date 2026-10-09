#!/usr/bin/env python
"""Run one question through the pipeline from the command line, printing the
live trace and the final answer. The same code path the HTTP API uses.

  python scripts/ask.py "who are our top 3 customers by spend"
  python scripts/ask.py --agent customer_directory "who is Globex"
  python scripts/ask.py --quiet "..."      # answer only, no trace

There is no end user on the command line, so service calls carry ZE_DEV_TOKEN
(e.g. from ze-mock-services/scripts/mint_token.py). Traced to Langfuse when its
keys are set, like an API run.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.manifest import load_all_agent_manifests  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.deps import Caller, Deps  # noqa: E402
from app.graph.build import build_graph, initial_state  # noqa: E402
from app.obs.tracing import callbacks_for, flush, trace_scope  # noqa: E402
from app.skills.registry import load_and_validate  # noqa: E402

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
GREEN, RED, YELLOW, CYAN = "\033[32m", "\033[31m", "\033[33m", "\033[36m"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--agent", default="general")
    ap.add_argument("--quiet", action="store_true", help="answer only")
    args = ap.parse_args()

    from contextlib import AsyncExitStack

    from app.tools.openapi.server import install_from_settings

    tool_stack = AsyncExitStack()
    await install_from_settings(tool_stack)  # every tool comes from a service spec
    load_and_validate()
    agents = load_all_agent_manifests()
    if args.agent not in agents:
        print(f"unknown agent '{args.agent}'. Available: {', '.join(agents)}")
        return 2
    agent = agents[args.agent]

    settings = get_settings()
    if not settings.ze_dev_token:
        print(f"{RED}set ZE_DEV_TOKEN{RESET}: every service call carries a user token, and the CLI has no user")
        return 2
    deps = Deps(
        settings=settings,
        caller=Caller(agent_id=agent.id, tool_deny=tuple(agent.scope.tool_deny)),
        user_token=settings.ze_dev_token,
    )

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    state = initial_state(
        request_id=str(uuid.uuid4()), thread_id=thread_id,
        question=args.question, agent=agent, caller=deps.caller,
    )
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.recursion_limit,
              **callbacks_for(thread_id=thread_id, agent_id=agent.id, question=args.question)}

    final: dict = {}
    with trace_scope(config):
        await _stream(graph, state, deps, config, args, final)
    flush()

    print()
    answer = final.get("answer")
    if answer is None:
        print(f"{RED}no answer produced{RESET}")
        await tool_stack.aclose()
        return 1

    print(f"{BOLD}ANSWER{RESET}" + (f" {YELLOW}(degraded){RESET}" if answer.degraded else ""))
    for block in answer.blocks:
        if block.kind == "text":
            print(block.text)
        elif block.kind == "metric":
            print(f"  {DIM}[{block.ref}]{RESET} {block.label}: {BOLD}{block.display}{RESET}")
        elif block.kind == "table":
            print(f"\n  {BOLD}{block.title or 'table'}{RESET}")
            keys = [c.key for c in block.columns]
            print("  " + " | ".join(c.label for c in block.columns))
            for row in block.rows[:25]:
                print("  " + " | ".join(str(row.get(k, "")) for k in keys))
    if answer.caveats:
        print(f"\n{DIM}caveats:{RESET}")
        for c in answer.caveats:
            print(f"  {DIM}- {c}{RESET}")
    print(f"\n{DIM}skills: {', '.join(answer.skills_used)} | facts: {len(answer.facts)}{RESET}")

    await tool_stack.aclose()
    return 0


async def _stream(graph, state, deps, config, args, final: dict) -> None:
    async for mode, chunk in graph.astream(
        state, context=deps, config=config, stream_mode=["custom", "updates"]
    ):
        if mode == "custom" and not args.quiet:
            _print_trace(chunk)
        elif mode == "updates":
            for node, update in chunk.items():
                if isinstance(update, dict):
                    final.update(update)
                    if not args.quiet:
                        _print_update(node, update)


def _print_trace(event: dict) -> None:
    t = event.get("type")
    if t == "node_start":
        print(f"{CYAN}> {event['node']}{RESET}", flush=True)
    elif t == "node_end" and not event.get("ok", True):
        print(f"{RED}  x {event['node']}: {event.get('error')}{RESET}", flush=True)
    elif t == "tool_call":
        args = str(event.get("args", {}))
        print(f"{DIM}    -> {event['tool']}({args[:120]}){RESET}", flush=True)
    elif t == "tool_result":
        if event.get("ok"):
            rows = f", {event['row_count']} rows" if event.get("row_count") is not None else ""
            print(f"{DIM}       {GREEN}ok{RESET}{DIM} {event.get('latency_ms')}ms{rows}{RESET}", flush=True)
        else:
            label = "DENIED" if event.get("denied") else "error"
            print(f"       {RED}{label}: {event.get('error')}{RESET}", flush=True)


def _print_update(node: str, update: dict) -> None:
    if node == "plan" and update.get("plan"):
        p = update["plan"]
        if p["unanswerable"]:
            print(f"{YELLOW}  plan: UNANSWERABLE -- {p['unanswerable_reason']}{RESET}")
        else:
            for s in p["steps"]:
                print(f"  plan: {BOLD}{s['skill_id']}{RESET} {DIM}-- {s['rationale'][:110]}{RESET}")
            print(f"  plan: output via {p['output_skill_id']}")
    elif node == "run_skill" and update.get("skill_results"):
        for ex in update["skill_results"]:
            color = GREEN if ex.result.status == "ok" else YELLOW
            print(f"  skill {BOLD}{ex.skill_id}{RESET}: {color}{ex.result.status}{RESET} "
                  f"{DIM}({len(ex.result.facts)} facts, {ex.steps_used} steps){RESET}")
    elif node == "validate":
        problems = update.get("last_validation_problems") or []
        if problems:
            print(f"  validate: {YELLOW}{len(problems)} problem(s){RESET}")
            for p in problems[:5]:
                print(f"    {DIM}- {p}{RESET}")
        else:
            print(f"  validate: {GREEN}passed{RESET}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
