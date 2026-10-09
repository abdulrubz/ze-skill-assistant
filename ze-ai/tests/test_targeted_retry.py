"""A validation failure must re-run only the step that failed.

Measured before this: 4 of 5 multi-skill runs replanned, against almost no
single-skill run, because `validate` pooled problems across all skills and
`plan` then discarded and re-ran every one of them. In one run
revenue_analytics was correct on the first attempt and was recomputed twice
because subscription_lifecycle failed. Cost scaled with the number of skills
selected rather than with the number that were wrong.
"""

from __future__ import annotations

import pytest
from langgraph.types import Send

from app.agents.manifest import load_all_agent_manifests
from app.graph.nodes.validate import validate
from app.graph.routing import route_after_plan, route_after_validate
from app.skills.envelope import Fact, Provenance, SkillExecutionResult, SkillResult


def _plan(*skill_ids: str) -> dict:
    return {
        "steps": [
            {"skill_id": s, "inputs": {}, "rationale": f"because {s}"} for s in skill_ids
        ],
        "output_skill_id": "prose_synthesis",
        "unanswerable": False,
        "unanswerable_reason": None,
    }


def _execution(skill_id: str, step: int, *, grounded: bool) -> SkillExecutionResult:
    """grounded=False produces an uncited number, which fails reference closure."""
    narrative = "the total is [f_01]" if grounded else "the total is 42"
    return SkillExecutionResult(
        skill_id=skill_id, step_index=step,
        result=SkillResult(
            skill_id=skill_id, status="ok", narrative=narrative,
            facts=[Fact(
                ref="f_01", name="total", value_str="42", value_kind="int",
                provenance=Provenance(tool_call_id="t1", tool="sql_query",
                                      source="api", locator="row#1.total"),
            )],
        ),
    )


def _state(**over) -> dict:
    base = {
        "question": "how much?",
        "agent": load_all_agent_manifests()["general"],
        "plan": _plan("revenue_analytics", "subscription_lifecycle"),
        "resolved": {"business_date": "2026-11-04"},
        "skill_results": [],
        "retry_count": 0,
        "last_validation_problems": [],
        "failed_step_indexes": [],
    }
    return {**base, **over}


@pytest.mark.asyncio
async def test_validate_attributes_failures_to_the_step_that_caused_them():
    state = _state(skill_results=[
        _execution("revenue_analytics", 0, grounded=True),
        _execution("subscription_lifecycle", 1, grounded=False),
    ])
    out = await validate(state)
    assert out["failed_step_indexes"] == [1], "only the ungrounded step should be blamed"
    assert out["last_validation_problems"]


@pytest.mark.asyncio
async def test_a_clean_run_blames_nobody():
    state = _state(skill_results=[
        _execution("revenue_analytics", 0, grounded=True),
        _execution("subscription_lifecycle", 1, grounded=True),
    ])
    out = await validate(state)
    assert out["failed_step_indexes"] == []
    assert out["last_validation_problems"] == []


def test_retry_reruns_only_the_failing_step_and_skips_the_planner():
    decision = route_after_validate(_state(
        last_validation_problems=["subscription_lifecycle: narrative states number(s) ['42']"],
        failed_step_indexes=[1],
        retry_count=1,
    ))
    assert isinstance(decision, list), "should fan out, not route back to plan"
    assert len(decision) == 1, "only the failing step should re-run"
    assert decision[0].arg["skill_id"] == "subscription_lifecycle"
    assert decision[0].arg["step_index"] == 1


def test_the_rerun_step_is_told_what_to_fix():
    decision = route_after_validate(_state(
        last_validation_problems=["subscription_lifecycle: uncited number 42"],
        failed_step_indexes=[1],
        retry_count=1,
    ))
    assert "uncited number 42" in decision[0].arg["task"]


def test_a_rerun_keeps_the_original_context():
    """Retry must be byte-identical to the first attempt apart from the
    feedback, or the retry path can drift from the one that was validated."""
    first = {s.arg["step_index"]: s.arg["task"] for s in route_after_plan(_state())}
    retry = route_after_validate(_state(
        last_validation_problems=["x"], failed_step_indexes=[1], retry_count=1,
    ))[0]
    assert retry.arg["task"].startswith(first[1])
    assert retry.arg["business_date"] == "2026-11-04"


def test_an_unattributable_failure_falls_back_to_a_full_replan():
    """A cross-skill invariant is nobody's individual fault; there is no single
    step to re-run, so the planner must get another go."""
    assert route_after_validate(_state(
        last_validation_problems=["never blend currencies"],
        failed_step_indexes=[],
        retry_count=1,
    )) == "plan"


def test_exhausted_retries_still_respond():
    agent = load_all_agent_manifests()["general"]
    assert route_after_validate(_state(
        last_validation_problems=["x"],
        failed_step_indexes=[1],
        retry_count=agent.max_retries,
    )) == "respond"


def test_a_passing_run_responds():
    assert route_after_validate(_state()) == "respond"
