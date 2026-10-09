"""The generic node every planned skill fans out to via Send. Looks the skill
up in the registry and runs its compiled subgraph.

Steps run in dependency waves (see app/graph/routing.py). A step that declared
`depends_on` receives its upstream steps' findings as `prior_findings` -- as
steering context only, never as facts to restate, since a fact is kept only
when its provenance points at a tool call in this skill's own transcript."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.runtime import get_runtime

from app.deps import Deps
from app.skills.envelope import SkillExecutionResult, SkillResult
from app.skills.registry import get_skill


class RunSkillInput(TypedDict):
    skill_id: str
    # Which plan step this is. Carried onto the result so the state reducer can
    # merge by step, which is what lets a failing step be re-run on its own.
    step_index: int
    task: str
    business_date: str | None
    # What the steps this one depends on already established. None for an
    # independent step, which is almost all of them.
    prior_findings: str | None


async def run_skill(payload: RunSkillInput) -> dict:
    outer = get_runtime(Deps).context
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None
    # A fresh copy per Send-spawned task: several run_skill invocations can be
    # in flight at once and must not race on one mutable Deps.
    deps = dataclasses.replace(outer, stream_writer=writer, business_date=payload.get("business_date"))

    skill_id = payload["skill_id"]
    step_index = payload.get("step_index", 0)
    skill = get_skill(skill_id)

    started = datetime.now(timezone.utc)
    out = await skill.subgraph.ainvoke(
        {
            "skill_id": skill_id,
            "task": payload["task"],
            "business_date": payload.get("business_date"),
            "prior_findings": payload.get("prior_findings"),
            "messages": [],
            "tool_steps": 0,
            "execution": None,
        },
        context=deps,
    )
    execution: SkillExecutionResult | None = out.get("execution")
    if execution is None:
        execution = SkillExecutionResult(
            skill_id=skill_id,
            step_index=step_index,
            result=SkillResult(skill_id=skill_id, status="error", narrative="skill produced no result"),
            started_at=started,
        )
    else:
        execution.started_at = started
        execution.step_index = step_index

    return {"skill_results": [execution]}
