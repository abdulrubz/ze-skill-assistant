"""Each planned step runs exactly once, through the REAL graph wiring.

Regression: the next-wave decision was a conditional edge off run_skill, which
LangGraph evaluates per task against that task's own writes. With two
independent steps in flight each task saw the other as not done and dispatched
it again -- every two-skill plan ran each skill twice. Unit tests of the
routing function could not see it; only running the graph does.
"""

from __future__ import annotations

from collections import Counter

import pytest

import app.graph.build as build
from app.agents.manifest import load_all_agent_manifests
from app.config import get_settings
from app.deps import Caller, Deps
from app.graph.state import Plan, PlanStep
from app.skills.envelope import SkillExecutionResult, SkillResult


@pytest.fixture
def runs(monkeypatch):
    counter: Counter = Counter()

    async def admission(state):
        return {"admission": {"allowed": True, "reason": None}}

    async def resolve(state):
        return {"resolved": {"business_date": "2026-10-01"}}

    def make_plan(steps):
        async def plan(state):
            return {"plan": Plan(steps=steps, output_skill_id="prose_synthesis", unanswerable=False,
                                 unanswerable_reason=None), "skill_results": None}
        return plan

    async def run_skill(payload):
        counter[payload["step_index"]] += 1
        return {"skill_results": [SkillExecutionResult(
            skill_id=payload["skill_id"], step_index=payload["step_index"],
            result=SkillResult(skill_id=payload["skill_id"], status="ok", narrative="n"))]}

    async def passthrough(state):
        return {}

    async def validate(state):
        return {"last_validation_problems": [], "failed_step_indexes": []}

    for name, fn in {"admission": admission, "resolve": resolve, "run_skill": run_skill,
                     "join": passthrough, "validate": validate, "respond": passthrough}.items():
        monkeypatch.setattr(build, name, fn)

    async def go(steps):
        monkeypatch.setattr(build, "plan", make_plan(steps))
        graph = build.build_graph()
        agent = load_all_agent_manifests()["general"]
        state = build.initial_state(request_id="r", thread_id="t", question="q", agent=agent,
                                    caller=Caller(agent_id="general"))
        deps = Deps(settings=get_settings(), caller=Caller(agent_id="general"))
        await graph.ainvoke(state, context=deps, config={"recursion_limit": 40})
        return counter

    return go


def _step(skill_id, depends_on=()):
    return PlanStep(skill_id=skill_id, inputs={}, rationale="", depends_on=list(depends_on))


async def test_independent_steps_each_run_once(runs):
    counts = await runs([_step("customer_profile"), _step("order_analysis")])
    assert counts == {0: 1, 1: 1}


async def test_a_dependent_step_runs_once_after_its_dependency(runs):
    counts = await runs([_step("customer_profile"), _step("order_analysis", depends_on=[0])])
    assert counts == {0: 1, 1: 1}


async def test_a_step_waiting_on_two_parallel_steps_still_runs(runs):
    counts = await runs([_step("customer_profile"), _step("order_analysis"),
                         _step("order_analysis", depends_on=[0, 1])])
    assert counts == {0: 1, 1: 1, 2: 1}
