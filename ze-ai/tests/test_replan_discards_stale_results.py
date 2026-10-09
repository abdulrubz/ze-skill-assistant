"""A replan must not leave the failed attempt's results in state.

Observed: a question failed validation, replanned once, and the answer came back
as six skill narratives -- the same three skills twice over, the broken version
and the corrected one concatenated into a single reply, with the stale facts
still cited. `skill_results` used a plain `add` reducer, so the retry appended
instead of replacing.
"""

from __future__ import annotations

from app.graph.state import replace_or_append
from app.skills.envelope import Fact, Provenance, SkillExecutionResult, SkillResult


def _result(skill_id: str, value: str, step: int = 0) -> SkillExecutionResult:
    return SkillExecutionResult(
        skill_id=skill_id,
        step_index=step,
        result=SkillResult(
            skill_id=skill_id,
            status="ok",
            narrative=f"value is [f_01] ({value})",
            facts=[Fact(
                ref="f_01", name="n", value_str=value, value_kind="int",
                provenance=Provenance(tool_call_id="t", tool="sql_query",
                                      source="api", locator="row#1.n"),
            )],
        ),
    )


def test_fan_out_results_from_different_steps_accumulate():
    merged = replace_or_append([_result("a", "1", step=0)], [_result("b", "2", step=1)])
    assert [r.skill_id for r in merged] == ["a", "b"]


def test_a_rerun_step_replaces_only_itself():
    """The point of per-step merging: re-running the step that failed must not
    disturb the steps that passed. Previously one skill's bad fact discarded and
    re-ran every skill -- in one run a correct result was recomputed twice
    because a DIFFERENT skill failed."""
    state = [_result("a", "1", step=0), _result("b", "2", step=1)]
    state = replace_or_append(state, [_result("b", "CORRECTED", step=1)])

    assert len(state) == 2
    by_step = {r.step_index: r.result.facts[0].value_str for r in state}
    assert by_step[0] == "1", "the passing step's work was destroyed"
    assert by_step[1] == "CORRECTED"


def test_the_same_skill_may_occupy_two_steps():
    """Keyed on step, not skill id, so a plan may legally run one skill twice
    with different inputs."""
    merged = replace_or_append(
        [_result("a", "acct1", step=0)], [_result("a", "acct2", step=1)]
    )
    assert len(merged) == 2


def test_none_clears_the_previous_attempt():
    assert replace_or_append([_result("a", "1", step=0), _result("b", "2", step=1)], None) == []


def test_a_cleared_state_accepts_the_retrys_results():
    """The full replan sequence: two results, cleared by Plan, refilled."""
    state = [_result("a", "1", step=0), _result("b", "2", step=1)]
    state = replace_or_append(state, None)
    state = replace_or_append(state, [_result("a", "99", step=0)])
    assert len(state) == 1
    assert state[0].result.facts[0].value_str == "99"


def test_clearing_an_empty_state_is_safe():
    assert replace_or_append(None, None) == []
    assert replace_or_append(None, [_result("a", "1")])[0].skill_id == "a"
