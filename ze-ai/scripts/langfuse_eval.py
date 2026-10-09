#!/usr/bin/env python
"""Evals on Langfuse: datasets from the golden banks, experiments, scores.

    python scripts/langfuse_eval.py sync                  # banks -> Langfuse datasets
    python scripts/langfuse_eval.py run --bank answer --label baseline
    python scripts/langfuse_eval.py run --bank routing

`sync` uploads every bank question that has a `truth` (the reference answer,
verified against the source data) as a dataset item: input = the question, expected
output = the truth, metadata = the pitfall, the routing expectation and the
substring checks. Item ids are derived from the question, so re-running `sync`
updates items in place instead of duplicating them. Run it again after editing
a bank.

`run` sends every item through the same pipeline POST /ask uses and scores it.
Each run appears in Langfuse under Datasets, where two runs (say before and
after a prompt edit) can be compared side by side, question by question.

Scores, per question:
  code checks (free, deterministic)
    answered, validation_clean, not_degraded, budget_ok   1 or 0
    expect_substrings   fraction of the bank's `expect` strings found in the answer
    cost_usd            the run's cost from the ledger
  LLM judge (one model call; the rubric is the Langfuse prompt `evals/answer-judge`)
    judge/correctness, judge/completeness, judge/no_fabrication   0, 0.5 or 1
    judge/overall       their mean
  The judge's reasoning is stored as each score's comment.

The rubric is text in Langfuse Prompt Management, so it is edited there like any
other prompt. The judge runs on the MLflow endpoint GATEWAY_JUDGE_ENDPOINT, overridden
by --judge-endpoint. --also-judge ENDPOINT grades with a second endpoint as well
(judge_ref/*) so the light judge can be checked against a stronger one.

    python scripts/langfuse_eval.py report --run <run-name>   # pass/fail vs eval_targets.yaml
    python scripts/langfuse_eval.py target                    # publish targets as `target-v1` runs

`report` prints each score's mean against the target in domain/evals/eval_targets.yaml,
within a tolerance (10%), and lists any question the judge scored below 0.5. Only the
answer-quality scores listed under `gate:` can fail it; process scores (validation,
tool budget, cost) are shown as info. `target`
publishes those same numbers as a synthetic run in each dataset so Langfuse's Compare
view can put a real run beside them.

This spends real API credit: one full pipeline run plus one judge call per item.
Runs are sequential on purpose (prompt-cache numbers stay interpretable).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

import yaml
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402

from app import prompt_store  # noqa: E402
from app.agents.manifest import load_all_agent_manifests  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.paths import EVALS_DIR as GOLDEN, VAR  # noqa: E402
from app.deps import Caller, Deps  # noqa: E402
from app.graph.build import build_graph  # noqa: E402
from app.graph.nodes.plan import plan as plan_node  # noqa: E402
from app.llm import judge_llm, named  # noqa: E402
from app.obs.ledger import LedgerStore  # noqa: E402
from app.obs.tracing import _client, callbacks_for  # noqa: E402
from app.skills.registry import load_and_validate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_cost import run_one  # noqa: E402

# Dataset names carry LANGFUSE_PROMPT_PREFIX, like the prompts, so apps sharing
# a Langfuse project do not share datasets by accident.
_DS = get_settings().langfuse_prompt_prefix
# kind "answer": end-to-end run, graded against a reference answer.
# kind "routing": planner only, graded on whether it picked an acceptable skill.
BANKS = {
    "answer": dict(dataset=f"{_DS}answer-bank", path=GOLDEN / "answer_bank.yaml", kind="answer", agent="general"),
    "routing": dict(dataset=f"{_DS}routing-bank", path=GOLDEN / "routing_bank.yaml", kind="routing", agent="general"),
}

TARGETS_FILE = GOLDEN / "eval_targets.yaml"
RUNS_DIR = VAR / "evals"

DIM, BOLD, RESET, GREEN, RED, YELLOW = "\033[2m", "\033[1m", "\033[0m", "\033[32m", "\033[31m", "\033[33m"


def _langfuse():
    client = _client()
    if client is None:
        sys.exit("Langfuse is not configured: set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST")
    return client


def _item_id(dataset: str, question: str) -> str:
    # Langfuse item ids are unique across the whole project, and the same
    # question appears in more than one bank, so the dataset is part of the id.
    return "q-" + hashlib.sha1(f"{dataset}\n{question}".encode()).hexdigest()[:16]


def _fold(text: str) -> str:
    """YAML `>` blocks fold to one line already; this just trims."""
    return " ".join(text.split())


def _bank_entries(bank: str) -> list[dict]:
    cfg = BANKS[bank]
    raw = yaml.safe_load(cfg["path"].read_text())
    if cfg["kind"] == "routing":
        return raw  # a bare list of {q, expect: [skill, ...]}
    entries = raw["questions"]
    missing = [e["q"] for e in entries if not e.get("truth")]
    if missing:
        print(f"{YELLOW}skipping {len(missing)} question(s) with no `truth`:{RESET} " +
              "; ".join(m[:40] for m in missing))
    return [e for e in entries if e.get("truth")]


# --------------------------------------------------------------------- sync


def sync(bank_names: list[str]) -> int:
    client = _langfuse()
    for bank in bank_names:
        cfg = BANKS[bank]
        dataset, path = cfg["dataset"], cfg["path"]
        entries = _bank_entries(bank)
        what = ("acceptable skills for the planner to route to" if cfg["kind"] == "routing"
                else "database-verified reference answers")
        client.create_dataset(
            name=dataset,
            description=f"Golden questions from domain/evals/{path.name}, with {what}.",
            metadata={"source": f"domain/evals/{path.name}", "kind": cfg["kind"]},
        )
        for e in entries:
            if cfg["kind"] == "routing":
                expected = {"acceptable_skills": e["expect"]}
                meta = {"bank": path.name}
            else:
                expected = _fold(e["truth"])
                meta = {"trap": e.get("trap") or "", "skills": e.get("skills", []),
                        "expect": e.get("expect", []), "note": e.get("note") or "", "bank": path.name}
            client.create_dataset_item(
                dataset_name=dataset, id=_item_id(dataset, e["q"]), input={"question": e["q"]},
                expected_output=expected, metadata=meta,
            )
        keep = {_item_id(dataset, e["q"]) for e in entries}
        stale = [i for i in client.get_dataset(dataset).items if i.id not in keep]
        for i in stale:
            client.api.dataset_items.delete(i.id)
        if stale:
            print(f"  {DIM}removed {len(stale)} stale item(s) from {dataset}{RESET}")
        print(f"  {GREEN}synced{RESET}  {dataset}  {DIM}{len(entries)} items{RESET}")
    client.flush()
    return 0


# ------------------------------------------------------------- evaluators


_REF = re.compile(r"\[([A-Za-z0-9_]+)\]")


def _display(fact: dict) -> str:
    unit = fact.get("currency") or fact.get("unit")
    return f"{fact['value_str']} {unit}" if unit else str(fact["value_str"])


def render_answer(payload: dict) -> str:
    """The final answer as a person would read it: prose with the [f_01] markers
    replaced by the values they cite (the UI shows chips; the judge has no UI),
    plus metrics, tables and caveats."""
    by_ref = {f["ref"]: f for f in payload.get("facts") or []}
    lines: list[str] = []
    for b in payload.get("blocks") or []:
        kind = b.get("kind")
        if kind == "text":
            lines.append(_REF.sub(lambda m: _display(by_ref[m.group(1)]) if m.group(1) in by_ref
                                  else m.group(0), b["text"]))
        elif kind == "metric":
            lines.append(f"{b['label']}: {b['display']}")
        elif kind == "table":
            if b.get("title"):
                lines.append(b["title"])
            cols = [c["key"] for c in b["columns"]]
            lines.append(" | ".join(c["label"] for c in b["columns"]))
            lines += [" | ".join(str(r.get(k, "")) for k in cols) for r in b["rows"]]
    lines += [f"Caveat: {c}" for c in payload.get("caveats") or []]
    return "\n".join(lines).strip()


def code_evaluator(*, input, output, expected_output=None, metadata=None, **_):
    out = output or {}
    md = metadata or {}
    scores = [
        {"name": "answered", "value": 1.0 if out.get("answered") else 0.0},
        {"name": "validation_clean", "value": 0.0 if out.get("validation_problems") else 1.0,
         "comment": "; ".join(out.get("validation_problems") or [])[:500] or None},
        {"name": "not_degraded", "value": 0.0 if out.get("degraded") else 1.0},
        {"name": "budget_ok", "value": 0.0 if out.get("exhausted_skills") else 1.0,
         "comment": ",".join(out.get("exhausted_skills") or []) or None},
    ]
    expect = md.get("expect") or []
    if expect:
        missing = out.get("expect_missing") or []
        scores.append({"name": "expect_substrings", "value": (len(expect) - len(missing)) / len(expect),
                       "comment": f"missing: {missing}" if missing else None})
    if out.get("cost_usd") is not None:
        scores.append({"name": "cost_usd", "value": float(out["cost_usd"])})
    return scores


def routing_evaluator(*, input, output, expected_output=None, metadata=None, **_):
    got = (output or {}).get("skills") or []
    acceptable = (expected_output or {}).get("acceptable_skills") or []
    if (output or {}).get("declined") or not got:
        ok, label = "UNANSWERABLE" in acceptable, "declined"
    else:
        ok, label = any(g in acceptable for g in got), "+".join(got)
    return [{"name": "routing_correct", "value": 1.0 if ok else 0.0,
             "comment": f"got {label}; acceptable: {', '.join(acceptable)}"}]


class Verdict(BaseModel):
    correctness: int = Field(ge=0, le=2)
    completeness: int = Field(ge=0, le=2)
    no_fabrication: int = Field(ge=0, le=2)
    reason: str = Field(description="One or two sentences naming the specific facts that were right or wrong")


def make_judge(model: str, prefix: str = "judge"):
    """An item evaluator that grades with `model` and writes scores as `<prefix>/...`."""

    async def judge_evaluator(*, input, output, expected_output=None, metadata=None, **_):
        return await _judge(model, prefix, input, output, expected_output, metadata)

    return judge_evaluator


async def _judge(model, prefix, input, output, expected_output, metadata):
    if not expected_output:
        return []
    md = metadata or {}
    rubric = prompt_store.pipeline(prompt_store.EVAL_JUDGE)
    user = "\n\n".join([
        f"QUESTION:\n{input['question']}",
        f"REFERENCE:\n{expected_output}",
        f"PITFALL:\n{md['trap']}" if md.get("trap") else "PITFALL:\n(none)",
        f"ANSWER:\n{(output or {}).get('answer') or '(no answer was produced)'}",
    ])
    with prompt_store.linked(rubric):
        raw = await named(judge_llm(model), "judge").with_structured_output(Verdict).ainvoke(
            [{"role": "system", "content": rubric.text}, {"role": "user", "content": user}]
        )
    v: Verdict = raw
    parts = {"correctness": v.correctness, "completeness": v.completeness, "no_fabrication": v.no_fabrication}
    comment = f"[{model}] {v.reason}"
    scores = [{"name": f"{prefix}/{k}", "value": n / 2, "comment": comment} for k, n in parts.items()]
    scores.append({"name": f"{prefix}/overall", "value": sum(parts.values()) / 6, "comment": comment})
    return scores


# -------------------------------------------------------------------- run


async def run(args) -> int:
    client = _langfuse()
    cfg = BANKS[args.bank]
    routing = cfg["kind"] == "routing"
    agent_id = args.agent or cfg["agent"]
    dataset_name = cfg["dataset"]
    items = list(client.get_dataset(dataset_name).items)
    if args.filter:
        items = [i for i in items if args.filter.lower() in i.input["question"].lower()]
    if not items:
        print(f"no items in {dataset_name} (run `sync` first) or none matched --filter")
        return 2

    settings = get_settings()
    judge_model = args.judge_endpoint or settings.endpoint_for("judge")
    stack = AsyncExitStack()
    if not routing:
        # Every tool is generated from the service specs, so this must run BEFORE
        # load_and_validate. Reversed, every tool call fails with ToolNotInstalled
        # and the run scores an honest zero on every answer.
        from app.tools.openapi.server import install_from_settings

        await install_from_settings(stack)
    load_and_validate()
    agent = load_all_agent_manifests()[agent_id]

    # Prompt versions ride on the run, so two runs can be told apart by what
    # was being served, not just by their label.
    served = prompt_store.warm()
    pipeline_versions = ",".join(
        f"{n.split('/')[-1]}={v.rsplit('v', 1)[-1]}" for n, v in served.items() if not n.startswith("skills/"))
    skills_fingerprint = hashlib.sha1(json.dumps(served, sort_keys=True).encode()).hexdigest()[:8]
    run_name = f"{args.label}-{time.strftime('%Y%m%d-%H%M%S')}"

    if routing:
        evaluators = [routing_evaluator]
        judged = "n/a (routing check is exact)"
    else:
        evaluators = [code_evaluator]
        if not args.no_judge:
            evaluators.append(make_judge(judge_model, "judge"))
            if args.also_judge:
                evaluators.append(make_judge(args.also_judge, "judge_ref"))
        judged = "off" if args.no_judge else judge_model + (f" (+ {args.also_judge})" if args.also_judge else "")
    print(f"{BOLD}{len(items)} item(s){RESET} | {dataset_name} | agent={agent_id} | run={run_name} | judge={judged}")

    try:
        if routing:
            caller = Caller(agent_id=agent.id)

            async def task(*, item, **_):
                question = item.input["question"]
                state = {"question": question, "agent": agent, "caller": caller,
                         "resolved": None, "retry_count": 0, "last_validation_problems": []}
                # Through a runnable, so the planner's model call inherits the
                # tracing callbacks and lands in this item's trace.
                out = await RunnableLambda(plan_node).ainvoke(
                    state, config=callbacks_for(thread_id=f"eval-{run_name}-{_item_id(dataset_name, question)}",
                                                agent_id=agent.id, question=question,
                                                tool_errors_as_warnings=True))
                plan = out["plan"]
                return {"skills": [st["skill_id"] for st in plan["steps"]],
                        "declined": bool(plan["unanswerable"])}
        else:
            if not settings.ze_dev_token:
                sys.exit("set ZE_DEV_TOKEN: answer evals call the services, which need a user token")
            deps = Deps(settings=settings, caller=Caller(agent_id=agent.id, tool_deny=tuple(agent.scope.tool_deny)),
                        user_token=settings.ze_dev_token)
            ledger_store = LedgerStore(settings.obs_sqlite_path)
            checkpointer = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(settings.checkpoint_sqlite_path))
            graph = build_graph(checkpointer=checkpointer)

            async def task(*, item, **_):
                question = item.input["question"]
                print(f"{DIM}  {question[:72]}{RESET}", flush=True)
                r = await run_one(
                    graph, deps, agent, question, ledger_store,
                    expect=(item.metadata or {}).get("expect"),
                    extra_config=callbacks_for(thread_id=f"eval-{run_name}-{_item_id(dataset_name, question)}",
                                               agent_id=agent.id, question=question,
                                               tool_errors_as_warnings=True),
                )
                return {
                    "answer": render_answer(r["answer_payload"]),
                    "answered": r["answered"], "degraded": r["degraded"],
                    "validation_problems": r["validation_problems"],
                    "exhausted_skills": r["exhausted_skills"],
                    "expect_missing": r["expect_missing"],
                    "skills_used": r["skills_used"], "cost_usd": r["cost_usd"],
                    "thread_id": r["thread_id"],
                }

        # `run_experiment` is a sync wrapper that always spins up its own event
        # loop -- fatal here, where the checkpointer and tool stack belong to
        # this one. The async core it wraps is called directly. It is a private
        # name: the SDK is pinned (pyproject: langfuse==4.15.2), and this is the
        # place to look if a bump breaks the script.
        result = await client._run_experiment_async(
            name=f"{_DS}{args.bank}-bank",
            run_name=run_name,
            # Full per-prompt versions live here: metadata values are capped at 200 chars.
            description=args.description or f"prompts served: {json.dumps(served)}",
            data=items,
            task=task,
            evaluators=evaluators,
            composite_evaluator=None,
            run_evaluators=[make_run_gate(args.bank)],
            max_concurrency=1,
            metadata={
                "agent": agent_id,
                "skill_endpoint": settings.endpoint_for("skill"),
                "planner_endpoint": settings.endpoint_for("planner"),
                "judge_endpoint": judge_model if not routing else "",
                "prompts": pipeline_versions,
                "prompts_fingerprint": skills_fingerprint,
            },
            dataset_version=None,
        )
    finally:
        await stack.aclose()

    _summarise(result)
    _save(run_name, args.bank, result)
    client.flush()
    url = getattr(result, "dataset_run_url", None)
    if url:
        print(f"\n{DIM}compare runs:{RESET} {url}")
    return 0


def _summarise(result) -> None:
    totals: dict[str, list[float]] = {}
    for r in result.item_results:
        for ev in r.evaluations:
            if isinstance(ev.value, (int, float)):
                totals.setdefault(ev.name, []).append(float(ev.value))
    print(f"\n{BOLD}mean score over {len(result.item_results)} item(s){RESET}")
    for name in sorted(totals):
        vals = totals[name]
        mean = sum(vals) / len(vals)
        shown = f"${sum(vals):.4f} total" if name == "cost_usd" else f"{mean:.2f}"
        print(f"  {name:<24}{shown:>14}  {DIM}n={len(vals)}{RESET}")


def _save(run_name: str, bank: str, result) -> None:
    """Keep each run's per-question scores on disk. The Langfuse v4 API cannot
    read scores back, and `report` needs them."""
    items = []
    for r in result.item_results:
        item = r.item
        inp = item["input"] if isinstance(item, dict) else item.input
        items.append({"question": inp["question"],
                      "scores": {ev.name: float(ev.value) for ev in r.evaluations
                                 if isinstance(ev.value, (int, float))}})
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{run_name}.json").write_text(json.dumps({"run": run_name, "bank": bank, "items": items}, indent=1))


def _targets(bank: str) -> tuple[dict[str, float], float]:
    raw = yaml.safe_load(TARGETS_FILE.read_text())
    return {**raw["scores"], **(raw.get("banks") or {}).get(bank, {})}, float(raw["tolerance"])


def _gate() -> set[str]:
    return set(yaml.safe_load(TARGETS_FILE.read_text()).get("gate") or [])


def _check(bank: str, by_score: dict[str, list[float]]) -> list[dict]:
    """Every score with a target, judged against it. One place, so `report` and the
    run-level score written to Langfuse can never disagree."""
    targets, tol = _targets(bank)
    gate = _gate()
    rows = []
    for name in sorted(by_score):
        if name not in targets:
            continue
        mean = sum(by_score[name]) / len(by_score[name])
        target = targets[name]
        lower_is_better = name == "cost_usd"
        bar = target * (1 + tol) if lower_is_better else target * (1 - tol)
        ok = mean <= bar if lower_is_better else mean >= bar
        rows.append({"name": name, "mean": mean, "target": target, "bar": bar, "ok": ok,
                     "gated": name in gate, "lower_is_better": lower_is_better})
    return rows


def make_run_gate(bank: str):
    """Run-level evaluator: one score for the whole run, 1 when every gated score is
    within tolerance of its target and 0 otherwise, with the misses as the comment.
    It shows in Langfuse's experiments table beside the averages."""

    def run_gate(*, item_results, **_):
        by_score: dict[str, list[float]] = {}
        for r in item_results:
            for ev in r.evaluations:
                if isinstance(ev.value, (int, float)):
                    by_score.setdefault(ev.name, []).append(float(ev.value))
        missed = [f"{r['name']} {r['mean']:.2f} < {r['bar']:.2f}" for r in _check(bank, by_score)
                  if r["gated"] and not r["ok"]]
        return [{"name": "gate/within_target", "value": 0.0 if missed else 1.0,
                 "comment": ("missed: " + "; ".join(missed)) if missed else "all gated scores within tolerance"}]

    return run_gate


# --------------------------------------------------------------------- report


def report(run: str) -> int:
    path = RUNS_DIR / f"{run}.json"
    if not path.exists():
        sys.exit(f"no saved scores for {run!r} in {RUNS_DIR} (runs made before `report` existed must be backfilled)")
    data = json.loads(path.read_text())
    _, tol = _targets(data["bank"])
    by_score: dict[str, list[float]] = {}
    for it in data["items"]:
        for k, v in it["scores"].items():
            by_score.setdefault(k, []).append(v)

    print(f"{BOLD}{run}{RESET}  {DIM}bank={data['bank']}  items={len(data['items'])}  tolerance={tol:.0%}{RESET}\n")
    print(f"  {'score':<24}{'mean':>8}{'target':>9}{'bar':>10}")
    failed = 0
    for r in _check(data["bank"], by_score):
        fmt = (lambda x: f"${x:.3f}") if r["lower_is_better"] else (lambda x: f"{x:.2f}")
        bar_label = f"{'<=' if r['lower_is_better'] else '>='} {fmt(r['bar'])}"
        if r["gated"]:
            failed += not r["ok"]
            mark = f"{GREEN}pass{RESET}" if r["ok"] else f"{RED}FAIL{RESET}"
        else:
            # Shown for context; never fails the run.
            mark = f"{DIM}info, within target{RESET}" if r["ok"] else f"{DIM}info, below target (not gated){RESET}"
        print(f"  {r['name']:<24}{fmt(r['mean']):>8}{fmt(r['target']):>9}{bar_label:>10}   {mark}")

    # A mean can hide a flatly wrong answer. List those regardless.
    key = "judge/overall" if "judge/overall" in by_score else "routing_correct"
    wrong = [it for it in data["items"] if it["scores"].get(key, 1) < 0.5]
    if wrong:
        print(f"\n{YELLOW}{len(wrong)} question(s) scored below 0.5 on {key}, whatever the averages say:{RESET}")
        for it in wrong:
            print(f"  - {it['question'][:90]}")
    print(f"\n{GREEN}all gated scores within target{RESET}" if not failed
          else f"\n{RED}{failed} gated score(s) missed target{RESET}")
    return 1 if failed else 0


# --------------------------------------------------------------------- target


async def publish_targets(bank_names: list[str], label: str) -> int:
    """One synthetic run per dataset whose scores ARE the targets, so a real run can
    be compared against it in Langfuse's own Compare view. No pipeline, no judge:
    the 'answer' is the reference itself and the scores are the numbers from the
    YAML. It is labelled as a target everywhere it can be, because it is not a
    measurement."""
    client = _langfuse()
    for bank in bank_names:
        cfg = BANKS[bank]
        targets, _ = _targets(bank)
        items = list(client.get_dataset(cfg["dataset"]).items)
        has_expect = any((i.metadata or {}).get("expect") for i in items)
        names = ["routing_correct"] if cfg["kind"] == "routing" else [
            n for n in targets if n not in ("routing_correct",) and (has_expect or n != "expect_substrings")]

        def evaluator(*, input, output, expected_output=None, metadata=None, _names=names, _t=targets, **_):
            return [{"name": n, "value": float(_t[n]), "comment": "TARGET (synthetic), not a measurement"}
                    for n in _names]

        async def task(*, item, **_):
            return {"answer": str(item.expected_output)}

        result = await client._run_experiment_async(
            name=f"{_DS}{bank}-bank", run_name=label, data=items, task=task, evaluators=[evaluator],
            composite_evaluator=None, run_evaluators=[], max_concurrency=8,
            description="TARGET values from domain/evals/eval_targets.yaml -- synthetic, NOT a real run. "
                        "Compare a real run against this one.",
            metadata={"synthetic": "true", "kind": "target"}, dataset_version=None,
        )
        print(f"  {GREEN}published{RESET}  {label}  {DIM}{cfg['dataset']}  {len(items)} items, "
              f"scores: {', '.join(names)}{RESET}")
    client.flush()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync", help="upload the banks as Langfuse datasets")
    s.add_argument("--bank", choices=[*BANKS, "all"], default="all")
    r = sub.add_parser("run", help="run a dataset through the pipeline and score it")
    r.add_argument("--bank", choices=list(BANKS), required=True)
    r.add_argument("--agent", default=None, help="default: the bank's own")
    r.add_argument("--label", default="run", help="prefix of the run name, e.g. 'baseline' or 'planner-v2'")
    r.add_argument("--description", default="")
    r.add_argument("--filter", help="substring: only questions containing it")
    r.add_argument("--no-judge", action="store_true", help="code checks only; no judge call")
    r.add_argument("--judge-endpoint", default=None, help="default: GATEWAY_JUDGE_ENDPOINT")
    r.add_argument("--also-judge", default=None, metavar="ENDPOINT",
                   help="grade with a second model too, scored as judge_ref/*, to check the light judge "
                        "against a stronger one")
    rp = sub.add_parser("report", help="score a saved run against domain/evals/eval_targets.yaml")
    rp.add_argument("--run", required=True, help="run name, e.g. baseline-20260929-231930")
    t = sub.add_parser("target", help="publish the targets as a synthetic `target-*` run per dataset")
    t.add_argument("--bank", choices=[*BANKS, "all"], default="all")
    t.add_argument("--label", default="target-v1")
    args = ap.parse_args()

    if args.cmd == "report":
        return report(args.run)
    if args.cmd == "target":
        return asyncio.run(publish_targets(list(BANKS) if args.bank == "all" else [args.bank], args.label))
    if args.cmd == "sync":
        return sync(list(BANKS) if args.bank == "all" else [args.bank])
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
