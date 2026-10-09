"""The boot self-test, as a test. If these fail the app refuses to start --
which is the point: a mis-wired skill should never reach a user."""

from __future__ import annotations

import pytest

from app.agents.manifest import load_all_agent_manifests
from app.skills.registry import load_and_validate, project_for_planner
from app.tools.registry import all_specs, declare_tools


@pytest.fixture(scope="module")
def skills():
    return load_and_validate()


def test_all_skills_load_and_validate(skills):
    assert len(skills) >= 2
    assert any(s.manifest.kind == "output" for s in skills.values())
    assert any(s.manifest.kind == "domain" for s in skills.values())


def test_every_domain_skill_has_a_rich_answers_surface(skills):
    """`answers` is the planner's ONLY retrieval surface -- a thin one silently
    narrows what the whole system can be asked."""
    for sid, s in skills.items():
        if s.manifest.kind != "domain":
            continue
        assert len(s.manifest.answers) >= 5, f"{sid} has only {len(s.manifest.answers)} example questions"


def test_every_domain_skill_declares_limits(skills):
    for sid, s in skills.items():
        if s.manifest.kind == "domain":
            assert s.manifest.cannot, f"{sid} declares no `cannot` -- the planner cannot decline on its behalf"


def test_planner_projection_leaks_no_implementation_detail(skills):
    """The planner must reason over declarations only -- never tool names,
    prompts or grants."""
    projection = project_for_planner(list(skills))
    for entry in projection:
        assert "tool_grants" not in entry
        assert "prompt" not in entry
        assert set(entry) <= {
            # `entities`/`operations` are declarations too -- the structural
            # retrieval surface. They say WHAT a skill is about and WHICH
            # question shapes it serves, never how it does either.
            "id", "kind", "summary", "entities", "operations", "answers", "intents",
            "requires", "provides", "cannot", "prefer_table",
        }


def test_write_capabilities_are_never_granted_to_a_skill(skills):
    """The tool layer is read-only: nothing may hold a write tool."""
    declare_tools()
    specs = all_specs()
    for sid, s in skills.items():
        for tool in s.manifest.tool_grants:
            assert specs[tool].access == "read", f"{sid} grants write tool {tool}"


def test_agents_reference_real_skills(skills):
    for aid, agent in load_all_agent_manifests().items():
        unknown = [s for s in agent.skills if s not in skills]
        assert not unknown, f"agent {aid} references unknown skills {unknown}"


def test_every_agent_has_an_output_skill(skills):
    """Without one, respond() has nothing to render and every answer would
    fall back to the default."""
    for aid, agent in load_all_agent_manifests().items():
        outputs = [s for s in agent.skills if skills[s].manifest.kind == "output"]
        assert outputs, f"agent {aid} grants no output skill"


def test_shortlist_preserves_the_expected_skill_for_the_whole_question_bank(skills):
    """The prefilter exists for latency, so it must not cost coverage: for
    every question in the bank, at least one acceptable skill must survive it.
    A miss here means the planner never even sees the right skill."""
    import yaml

    from app.paths import EVALS_DIR
    from app.skills.registry import shortlist_skills

    bank = yaml.safe_load((EVALS_DIR / "routing_bank.yaml").read_text())
    all_ids = list(skills)
    misses = []
    for case in bank:
        expected = [e for e in case["expect"] if e != "UNANSWERABLE"]
        if not expected:
            continue
        survivors = set(shortlist_skills(case["q"], all_ids))
        if not any(e in survivors for e in expected):
            misses.append(f"{case['q']!r}: none of {expected} survived")
    assert not misses, "prefilter dropped the right skill:\n" + "\n".join(misses)


def test_shortlist_always_keeps_an_output_skill(skills):
    from app.skills.registry import shortlist_skills

    survivors = shortlist_skills("what is our revenue", list(skills))
    assert any(skills[s].manifest.kind == "output" for s in survivors)


def test_shortlist_does_not_narrow_when_there_is_no_lexical_signal(skills):
    from app.skills.registry import shortlist_skills

    assert set(shortlist_skills("???", list(skills))) == set(skills)


async def test_a_refusal_is_not_reported_as_a_degraded_answer(skills):
    """`degraded` means "claims were made that could not be verified". A clean
    decline makes no claims, so flagging it degraded misrepresents it."""
    from app.agents.manifest import load_all_agent_manifests
    from app.deps import Caller
    from app.graph.nodes.respond import respond

    agent = load_all_agent_manifests()["general"]
    base = {
        "question": "q", "agent": agent,
        "caller": Caller(agent_id="general", ),
        "skill_results": [], "retry_count": 0, "last_validation_problems": [],
    }

    refused = await respond({**base, "admission": {"allowed": False, "reason": "injection"}, "plan": None})
    assert refused["answer"].degraded is False

    unanswerable = await respond({
        **base, "admission": {"allowed": True, "reason": None},
        "plan": {"steps": [], "output_skill_id": "prose_synthesis",
                 "unanswerable": True, "unanswerable_reason": "no skill covers this"},
    })
    assert unanswerable["answer"].degraded is False
    assert "no skill covers this" in unanswerable["answer"].blocks[0].text
