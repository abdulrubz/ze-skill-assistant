"""Respond: runs plan.output_skill_id. EVERY path reaches this node --
admission refusals and unanswerable plans included -- so the stream always
ends with an answer block rather than dead-ending."""

from __future__ import annotations

from functools import lru_cache

from app.agents.manifest import AgentManifest, load_all_agent_manifests
from app.graph.state import PipelineState
from app.skills.answer import Answer, TextSpan
from app.skills.output import render_answer
from app.skills.registry import all_skills


@lru_cache
def _agent_manifests() -> dict[str, AgentManifest]:
    """Manifests are read from disk at boot and never change at runtime; the
    refusal path should not re-parse six YAML files to build a hint."""
    return load_all_agent_manifests()


async def respond(state: PipelineState) -> dict:
    # A refusal is NOT a degraded answer. `degraded` means "claims were made
    # that validation could not fully verify"; a decline makes no claims at
    # all, so flagging it as unverified misrepresents a clean, correct refusal.
    admission = state.get("admission")
    if admission and not admission["allowed"]:
        return {"answer": Answer(
            blocks=[TextSpan(text=f"I can't help with that: {admission['reason']}")],
            skills_used=[], degraded=False), "degraded": False}

    plan = state.get("plan")
    if plan is None or plan["unanswerable"] or not plan["steps"]:
        reason = (plan and plan["unanswerable_reason"]) or "none of my skills cover that question"
        text = f"I can't answer that: {reason}"
        hint = _agent_that_could_help(state)
        if hint:
            text += f"\n\n{hint}"
        return {"answer": Answer(
            blocks=[TextSpan(text=text)],
            skills_used=[], degraded=False), "degraded": False}

    manifest = all_skills()[plan["output_skill_id"]].manifest
    answer = await render_answer(
        state["question"], state["skill_results"], allow_table=manifest.prefer_table
    )

    degraded = bool(state.get("last_validation_problems")) and \
        state["retry_count"] >= state["agent"].max_retries

    if degraded:
        answer.degraded = True
        answer.caveats.append(
            "Some claims could not be fully verified after repeated attempts; treat this as best-effort."
        )
    return {"answer": answer, "degraded": degraded}


def _agent_that_could_help(state: PipelineState) -> str:
    """A refusal that only says "no" is indistinguishable from a broken system.

    Most refusals in practice are SCOPE refusals: the skill exists, this agent
    just isn't granted it. Asking "how many subscriptions are in trial" as the
    Collections Analyst declines correctly -- it has no subscription_lifecycle
    skill -- but reads like a failure unless the reply says which agent does.

    Returns "" for a genuine capability gap, where no agent can help and
    suggesting one would be a lie.
    """
    agent = state["agent"]
    try:
        registry = all_skills()
        mine = set(agent.skills)

        def gains(other) -> list[str]:
            return [s for s in other.skills if s not in mine]

        others = {
            a.display_name: gains(a)
            for a in _agent_manifests().values()
            if a.id != agent.id
        }
    except Exception:  # noqa: BLE001 -- a hint must never break a refusal
        return ""

    # Prefer the broadest agent that adds something; `general` holds every
    # read-only skill, so it is the right suggestion whenever it adds any.
    ranked = sorted(
        ((name, extra) for name, extra in others.items() if extra),
        key=lambda kv: len(kv[1]), reverse=True,
    )
    if not ranked:
        return ""
    name, extra = ranked[0]
    return (
        f"This is a scope limit, not a capability gap: **{name}** has "
        f"{len(extra)} skill{'s' if len(extra) != 1 else ''} that "
        f"{agent.display_name} does not. Try switching agent and asking again."
    )
