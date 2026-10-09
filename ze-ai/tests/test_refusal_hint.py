"""A refusal must say whether it is a scope limit or a real capability gap.

Asking about orders as the Customer Directory agent declines -- correctly,
since that agent is not granted order_analysis. But a flat "I can't answer
that" is indistinguishable from a broken system, so the refusal names the
broader agent.
"""

from __future__ import annotations

import pytest

from app.agents.manifest import load_all_agent_manifests
from app.graph.nodes.respond import _agent_that_could_help, respond


def _state(agent_id: str, **over) -> dict:
    agents = load_all_agent_manifests()
    base = {
        "agent": agents[agent_id],
        "question": "how much has Globex spent?",
        "admission": {"allowed": True, "reason": ""},
        "plan": {"steps": [], "output_skill_id": "prose_synthesis",
                 "unanswerable": True, "unanswerable_reason": "no skill covers that"},
        "skill_results": [],
        "retry_count": 0,
        "last_validation_problems": [],
    }
    return {**base, **over}


def test_a_scoped_agent_is_told_which_agent_is_broader():
    hint = _agent_that_could_help(_state("customer_directory"))
    assert "General" in hint
    assert "scope limit" in hint


def test_the_broadest_agent_gets_no_misleading_hint():
    """`general` holds every skill, so a refusal from it is a genuine
    capability gap and suggesting somewhere else would be a lie."""
    assert _agent_that_could_help(_state("general")) == ""


@pytest.mark.asyncio
async def test_refusal_carries_the_hint_and_is_not_degraded():
    out = await respond(_state("customer_directory"))
    text = out["answer"].blocks[0].text
    assert "I can't answer that" in text
    assert "General" in text
    # A decline makes no claims, so it is a clean answer, never "unverified".
    assert out["degraded"] is False
    assert out["answer"].degraded is False


@pytest.mark.asyncio
async def test_an_admission_refusal_is_unaffected():
    state = _state("general", admission={"allowed": False, "reason": "blocked"})
    out = await respond(state)
    assert "I can't help with that: blocked" in out["answer"].blocks[0].text
