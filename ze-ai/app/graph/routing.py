"""Pure, unit-testable routing predicates for the pipeline's conditional edges."""

from __future__ import annotations

import re

from langgraph.types import Send

from app.graph.state import PipelineState
from app.skills.envelope import SkillResult

_REF = re.compile(r"\[([a-zA-Z0-9_]+)\]")


def _resolve_refs(result: SkillResult) -> str:
    """Substitute an upstream narrative's [f_xx] markers with the values they
    stand for, before that text is shown to a DIFFERENT skill.

    Passing the markers through verbatim was a real bug, not a cosmetic one.
    Fact refs are only unique within one skill result, so a downstream skill
    reading "retention stands at [f_11] accounts" would copy `[f_11]` into its
    own narrative -- where it means nothing, because that skill's own facts are
    numbered from f_01 independently. Groundedness then rejects the answer
    ("narrative cites unknown fact ref(s) ['f_11', 'f_12']"), which triggers a
    retry, which re-runs the whole skill. Observed exactly that: the leaked
    refs were the upstream skill's, and the retry roughly doubled the run.

    Resolving them also makes the steering context more useful -- "the cohort
    is 2026-08 with 22 accounts" is what the next skill needs to know, and
    "[f_01] with [f_02] accounts" tells it nothing at all.
    """
    by_ref = {f.ref: f for f in result.facts}

    def sub(match: re.Match) -> str:
        fact = by_ref.get(match.group(1))
        if fact is None:
            return ""  # a dangling marker is noise; drop it rather than leak it
        return fact.value_str + (f" {fact.currency}" if fact.currency else "")

    return _REF.sub(sub, result.narrative).strip()


def route_after_admission(state: PipelineState) -> str:
    admission = state.get("admission")
    return "respond" if admission and not admission["allowed"] else "resolve"


def _prior_findings(state: PipelineState, step_index: int) -> str | None:
    """What an upstream step found, for a step that declared it depends on it.

    Deliberately NOT facts to reuse. A downstream skill restating an upstream
    fact would have it dropped: app/skills/factory.py keeps a fact only if its
    provenance points at a tool call in THAT subgraph's own transcript. So this
    is steering context -- it tells the skill which cohort/accounts/window the
    question turned out to be about, so it does not spend half its tool budget
    re-deriving what is already known, and it still retrieves and cites every
    value it reports itself.
    """
    plan = state["plan"]
    deps = plan["steps"][step_index].get("depends_on") or []
    if not deps:
        return None

    by_index = {ex.step_index: ex for ex in (state.get("skill_results") or [])}
    blocks: list[str] = []
    for d in deps:
        ex = by_index.get(d)
        if ex is None or ex.result.status in ("error", "denied"):
            continue
        lines = [f"From the earlier `{ex.skill_id}` step:", _resolve_refs(ex.result)]
        values = [
            f"- {f.name} = {f.value_str}"
            + (f" {f.currency}" if f.currency else "")
            + (f" ({', '.join(f'{k}={v}' for k, v in f.entity.items())})" if f.entity else "")
            for f in ex.result.facts
        ]
        if values:
            lines.append("Values it established:")
            lines += values
        blocks.append("\n".join(lines))

    if not blocks:
        return None
    return "\n\n".join(blocks)


def _sends_for(
    state: PipelineState, step_indexes: list[int], *, feedback: list[str] | None = None
) -> list[Send]:
    """Build the fan-out for the given plan steps.

    Shared by the initial dispatch and by a targeted retry, so a re-run step
    gets byte-identical context to its first attempt plus the validation
    feedback -- nothing about the retry path can drift from the original.

    Note `prior_findings` rides on its OWN key rather than being folded into
    `task`. The task string is the retry path's identity check (a re-run must
    be a prefix of its first attempt), and upstream findings exist at retry
    time whether or not they existed at first dispatch.
    """
    plan = state["plan"]
    resolved = state.get("resolved") or {}
    business_date = resolved.get("business_date")
    context_inputs = {k: v for k, v in resolved.items() if v and k != "business_date"}

    sends = []
    for i in step_indexes:
        step = plan["steps"][i]
        merged = {**context_inputs, **step["inputs"]}
        task = [state["question"]]
        if merged:
            task.append("Known context: " + ", ".join(f"{k}={v}" for k, v in merged.items()))
        if step["rationale"]:
            task.append(f"(You were selected because: {step['rationale']})")
        if feedback:
            task.append(
                "\nYour PREVIOUS attempt failed these grounding checks. Fix exactly these "
                "and change nothing else:\n" + "\n".join(f"- {p}" for p in feedback)
            )
        sends.append(Send("run_skill", {
            "skill_id": step["skill_id"],
            "step_index": i,
            "task": "\n".join(task),
            "business_date": business_date,
            "prior_findings": _prior_findings(state, i),
        }))
    return sends


def ready_step_indexes(state: PipelineState) -> list[int]:
    """Plan steps that have not run yet and whose dependencies have.

    With no `depends_on` anywhere -- the overwhelming majority of plans -- this
    returns every step on the first call and nothing afterwards, which is
    exactly the single parallel wave the pipeline has always had.
    """
    plan = state.get("plan")
    if not plan or not plan.get("steps"):
        return []
    done = {ex.step_index for ex in (state.get("skill_results") or [])}
    return [
        i for i in range(len(plan["steps"]))
        if i not in done
        and all(d in done for d in (plan["steps"][i].get("depends_on") or []))
    ]


def route_after_plan(state: PipelineState) -> str | list[Send]:
    plan = state.get("plan")
    if plan is None or plan["unanswerable"] or not plan["steps"]:
        return "respond"
    return _sends_for(state, ready_step_indexes(state))


def route_after_run_skill(state: PipelineState) -> str | list[Send]:
    """Dispatch the next dependency wave, or fall through to join.

    `depends_on` is only ever a backward reference (enforced in plan.py), so
    every pending step's dependencies are eventually satisfied and this cannot
    spin. The belt-and-braces case is a step whose dependency produced no
    result at all: `ready_step_indexes` would keep it blocked forever, so any
    plan step still pending with nothing ready falls through to join and is
    reported as missing rather than hanging the run.
    """
    ready = ready_step_indexes(state)
    return _sends_for(state, ready) if ready else "join"


def route_after_validate(state: PipelineState) -> str | list[Send]:
    problems = state.get("last_validation_problems") or []
    if not problems:
        return "respond"
    if state["retry_count"] >= state["agent"].max_retries:
        return "respond"  # exhausted -- respond marks it degraded

    # Re-run ONLY the steps that failed, straight back into the fan-out.
    #
    # Going via `plan` re-ran every step: in one observed run revenue_analytics
    # was correct on the first attempt and was discarded and recomputed twice
    # because a DIFFERENT skill failed. That is why multi-skill questions cost
    # so much more -- 4 of 5 replanned, against almost no single-skill run.
    #
    # Skipping `plan` also drops a planner call per retry and, more importantly,
    # stops the planner reconsidering a selection that was never the problem.
    failed = state.get("failed_step_indexes") or []
    if failed and state.get("plan"):
        return _sends_for(state, failed, feedback=problems)

    # No step could be attributed (e.g. a cross-skill invariant failed, which
    # is nobody's individual fault) -- fall back to a full replan.
    return "plan"
