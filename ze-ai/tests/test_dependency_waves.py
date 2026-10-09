"""Plan steps that declare `depends_on` run in waves, and get the upstream
findings.

Why this exists: the planner picked customer_profile AND order_analysis
for "for accounts in the most recent signup cohort, what's their retention and
plan mix", they fanned out in one wave with no channel between them, and both
independently called describe_schema(signup_cohorts), both ran a "most recent
signup cohort" query, and both joined accounts->bundles->subscriptions. Measured
across the ledger, 2-skill plans cost 3.2x a 1-skill plan rather than 2x.

The invariants here are the ones that keep that fix from breaking other things:
a plan with no declared dependency must behave exactly as before, a cycle must
be unexpressible, the retry path's task string must not move, and upstream
findings must never become a substitute for the downstream skill's own citations.
"""

from __future__ import annotations

from app.agents.manifest import load_all_agent_manifests
from app.graph.nodes.plan import PlanStepOut, keep_domain_steps
from app.graph.routing import (
    _prior_findings,
    ready_step_indexes,
    route_after_plan,
    route_after_run_skill,
    route_after_validate,
)
from app.graph.state import Plan, PlanStep
from app.skills.envelope import Fact, Provenance, SkillExecutionResult, SkillResult


def _step(skill_id: str, *, depends_on=(), rationale="r") -> PlanStep:
    return PlanStep(
        skill_id=skill_id, inputs={}, rationale=rationale, depends_on=list(depends_on)
    )


def _result(skill_id: str, step_index: int, *, facts=(), status="ok", narrative="n") -> SkillExecutionResult:
    return SkillExecutionResult(
        skill_id=skill_id, step_index=step_index,
        result=SkillResult(
            skill_id=skill_id, status=status, narrative=narrative, facts=list(facts)
        ),
    )


def _state(steps, results=None, **over) -> dict:
    base = {
        "question": "q",
        "agent": load_all_agent_manifests()["general"],
        "plan": Plan(steps=steps, output_skill_id="prose_synthesis",
                     unanswerable=False, unanswerable_reason=None),
        "resolved": {"business_date": "2026-11-04"},
        "skill_results": list(results or []),
        "retry_count": 0,
        "failed_step_indexes": [],
        "last_validation_problems": [],
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# the no-dependency case must be untouched
# --------------------------------------------------------------------------

def test_independent_steps_still_fan_out_in_one_wave():
    sends = route_after_plan(_state([_step("a"), _step("b"), _step("c")]))
    assert [s.arg["step_index"] for s in sends] == [0, 1, 2]
    assert all(s.arg["prior_findings"] is None for s in sends)


def test_a_completed_single_wave_falls_through_to_join():
    state = _state(
        [_step("a"), _step("b")],
        [_result("a", 0), _result("b", 1)],
    )
    assert route_after_run_skill(state) == "join"


# --------------------------------------------------------------------------
# wave scheduling
# --------------------------------------------------------------------------

def test_a_dependent_step_is_held_back_from_the_first_wave():
    steps = [_step("customer_profile"), _step("order_analysis", depends_on=[0])]
    sends = route_after_plan(_state(steps))
    assert [s.arg["step_index"] for s in sends] == [0], "step 1 must wait for step 0"


def test_the_next_wave_dispatches_once_its_dependency_lands():
    steps = [_step("customer_profile"), _step("order_analysis", depends_on=[0])]
    state = _state(steps, [_result("customer_profile", 0)])
    sends = route_after_run_skill(state)
    assert [s.arg["step_index"] for s in sends] == [1]


def test_independent_steps_in_a_mixed_plan_still_run_together():
    steps = [_step("a"), _step("b"), _step("c", depends_on=[0])]
    sends = route_after_plan(_state(steps))
    assert [s.arg["step_index"] for s in sends] == [0, 1], "b must not wait on anything"


def test_a_step_whose_dependency_never_produced_a_result_does_not_hang():
    """Belt and braces: if step 0 somehow produced nothing, step 1 can never
    become ready. Falling through to join reports a missing step; spinning
    would hang the request."""
    steps = [_step("a"), _step("b", depends_on=[0])]
    state = _state(steps, [])  # nothing done, but pretend step 0 will never land
    # simulate the impossible state directly: step 0 absent, and not ready
    state["plan"]["steps"][0]["depends_on"] = [0]  # self-dep, which plan.py forbids
    assert ready_step_indexes(state) == []
    assert route_after_run_skill(state) == "join"


# --------------------------------------------------------------------------
# cycles are unexpressible -- enforced in plan.py, not hoped for
# --------------------------------------------------------------------------

def _sanitise(raw_steps, valid_ids):
    """The real shipped function, not a reimplementation of it."""
    return keep_domain_steps(raw_steps, lambda sid: sid in valid_ids)


def test_a_forward_or_self_reference_is_dropped():
    raw = [
        PlanStepOut(skill_id="a", rationale="r", depends_on=[1]),   # forward
        PlanStepOut(skill_id="b", rationale="r", depends_on=[1]),   # self
        PlanStepOut(skill_id="c", rationale="r", depends_on=[0, 9]),  # one valid, one OOR
    ]
    steps = _sanitise(raw, {"a", "b", "c"})
    assert [s["depends_on"] for s in steps] == [[], [], [0]]


def test_dependencies_are_remapped_when_a_step_is_filtered_out():
    """`depends_on` indexes the planner's own list. If a step is dropped for
    being unknown or an output skill, every later index shifts -- a raw index
    would then point at the wrong step."""
    raw = [
        PlanStepOut(skill_id="dropped", rationale="r"),
        PlanStepOut(skill_id="a", rationale="r"),
        PlanStepOut(skill_id="b", rationale="r", depends_on=[1]),  # means "a"
    ]
    steps = _sanitise(raw, {"a", "b"})
    assert [s["skill_id"] for s in steps] == ["a", "b"]
    assert steps[1]["depends_on"] == [0], "should still point at 'a' after the shift"


def test_a_dependency_on_a_dropped_step_is_discarded_not_misaimed():
    raw = [
        PlanStepOut(skill_id="dropped", rationale="r"),
        PlanStepOut(skill_id="a", rationale="r", depends_on=[0]),
    ]
    steps = _sanitise(raw, {"a"})
    assert steps[0]["depends_on"] == []


# --------------------------------------------------------------------------
# what gets threaded, and what must NOT
# --------------------------------------------------------------------------

def _fact(ref, name, value, **kw):
    return Fact(
        ref=ref, name=name, value_str=value, value_kind=kw.pop("kind", "str"),
        provenance=Provenance(
            tool_call_id="call_1", tool="sql_query", source="api", locator="row 1"
        ),
        **kw,
    )


def test_prior_findings_carry_the_upstream_narrative_and_values():
    steps = [_step("customer_profile"), _step("order_analysis", depends_on=[0])]
    upstream = _result(
        "customer_profile", 0,
        narrative="The most recent signup cohort is 2026-08 with 22 accounts.",
        facts=[
            _fact("f_01", "cohort_month", "2026-08", kind="date"),
            _fact("f_02", "cohort_accounts", "22", kind="int"),
        ],
    )
    text = _prior_findings(_state(steps, [upstream]), 1)
    assert "customer_profile" in text
    assert "2026-08" in text and "22" in text
    assert "cohort_month" in text


def test_an_independent_step_gets_no_prior_findings():
    assert _prior_findings(_state([_step("a"), _step("b")]), 1) is None


def test_a_failed_upstream_step_contributes_nothing():
    """Steering a skill with a failed step's narrative would propagate the
    failure into a step that could still succeed on its own."""
    steps = [_step("a"), _step("b", depends_on=[0])]
    for status in ("error", "denied"):
        bad = _result("a", 0, status=status, narrative="tools all failed")
        assert _prior_findings(_state(steps, [bad]), 1) is None


def test_prior_findings_ride_on_their_own_key_not_the_task():
    """The retry path asserts a re-run's task is a PREFIX of the first
    attempt's. Upstream findings exist at retry time whether or not they
    existed at first dispatch, so folding them into `task` would make every
    retry of a dependent step diverge."""
    steps = [_step("customer_profile"), _step("order_analysis", depends_on=[0])]
    upstream = _result("customer_profile", 0, narrative="cohort is 2026-08",
                       facts=[_fact("f_01", "cohort_month", "2026-08")])

    first = route_after_run_skill(_state(steps, [upstream]))[0]
    retry = route_after_validate(_state(
        steps, [upstream, _result("order_analysis", 1)],
        last_validation_problems=["x"], failed_step_indexes=[1], retry_count=1,
    ))[0]

    assert retry.arg["task"].startswith(first.arg["task"])
    assert "2026-08" not in first.arg["task"], "findings must not be in the task string"
    assert "2026-08" in first.arg["prior_findings"]


def test_the_downstream_prompt_forbids_restating_upstream_numbers():
    """A fact is kept only if its provenance points at a tool call in THIS
    skill's transcript (app/skills/factory.py). A downstream skill that
    restated an upstream number would have it silently dropped and its status
    downgraded, so the prompt has to say so explicitly."""
    from app.skills.factory import render_skill_system_prompt
    from app.skills.registry import all_skills, load_and_validate

    load_and_validate()
    manifest = all_skills()["order_analysis"].manifest
    rendered = render_skill_system_prompt(manifest, "2026-11-04", "cohort is 2026-08")
    assert "cohort is 2026-08" in rendered
    assert "Already established by an earlier step" in rendered
    assert "does NOT substitute for retrieval" in rendered


def test_prior_findings_sit_after_the_cacheable_prefix():
    from app.skills.factory import render_skill_system_prompt, render_static_skill_prompt
    from app.skills.registry import all_skills, load_and_validate

    load_and_validate()
    manifest = all_skills()["order_analysis"].manifest
    static = render_static_skill_prompt(manifest)
    assert render_skill_system_prompt(
        manifest, "2026-11-04", "anything at all"
    ).startswith(static)


# --------------------------------------------------------------------------
# regression: citation markers must not cross a skill boundary
# --------------------------------------------------------------------------

def test_upstream_citation_markers_are_resolved_not_leaked():
    """Observed live: customer_profile's narrative "retention stands at [f_11]
    accounts" was injected verbatim into order_analysis, which copied
    [f_11] and [f_12] into its OWN narrative. Fact refs are only unique within
    one skill result, so groundedness rejected it -- "narrative cites unknown
    fact ref(s) ['f_11', 'f_12']" -- and the retry re-ran the whole skill,
    roughly doubling the run. The markers must never reach the next skill.
    """
    steps = [_step("customer_profile"), _step("order_analysis", depends_on=[0])]
    upstream = _result(
        "customer_profile", 0,
        narrative="The cohort is [f_01] with [f_02] accounts; retention is [f_03].",
        facts=[
            _fact("f_01", "cohort_month", "2026-08", kind="date"),
            _fact("f_02", "cohort_accounts", "22", kind="int"),
            _fact("f_03", "retained_accounts", "13", kind="int"),
        ],
    )
    text = _prior_findings(_state(steps, [upstream]), 1)

    assert "[f_" not in text, f"citation marker leaked across skills: {text}"
    assert "2026-08" in text and "22" in text and "13" in text, (
        "markers must be replaced by their VALUES, not just stripped -- "
        "'the cohort is  with  accounts' steers nothing"
    )


def test_a_dangling_upstream_marker_is_dropped_not_passed_on():
    steps = [_step("a"), _step("b", depends_on=[0])]
    upstream = _result("a", 0, narrative="value is [f_09] here", facts=[])
    text = _prior_findings(_state(steps, [upstream]), 1)
    assert "[f_09]" not in text


def test_a_money_value_keeps_its_currency_when_resolved():
    steps = [_step("a"), _step("b", depends_on=[0])]
    upstream = _result(
        "a", 0, narrative="balance is [f_01]",
        facts=[_fact("f_01", "balance", "990.00", kind="decimal", currency="USD")],
    )
    assert "990.00 USD" in _prior_findings(_state(steps, [upstream]), 1)


def test_the_citation_rule_is_restated_after_the_volatile_tail():
    """Splitting the prompt for cacheability moved the citation rule off the
    end, and what replaced it is dense with numbers the model might repeat."""
    from app.skills.factory import render_skill_system_prompt
    from app.skills.registry import all_skills, load_and_validate

    load_and_validate()
    manifest = all_skills()["order_analysis"].manifest
    rendered = render_skill_system_prompt(manifest, "2026-11-04", "cohort is 2026-08")
    tail = rendered[rendered.index("Already established by an earlier step"):]
    assert "EVERY number you write must be one YOU retrieved" in tail, (
        "the citation rule must come AFTER the numbers it is protecting against"
    )
