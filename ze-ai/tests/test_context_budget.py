"""What a skill's tool loop is allowed to put in its context, and keep re-sending.

A skill re-sends its entire system prompt, its tool schemas and every prior tool
result on EVERY step of its loop. Measured on run b3206a76 that made the
byte-identical prefix alone 42% of the run's input tokens, resent 18 times with
zero cache reads. These tests hold the property that fix bought: the static
prefix really is static (nothing per-run leaks into it), so a provider that
caches prompt prefixes can serve it.
"""

from __future__ import annotations

from app.skills.factory import render_static_skill_prompt, render_skill_system_prompt
from app.skills.registry import all_skills, load_and_validate
from app.tools.registry import declare_tools

declare_tools()
load_and_validate()


# --------------------------------------------------------------------------
# 1. prefix stability -- the property every caching win depends on
# --------------------------------------------------------------------------

def test_the_static_prompt_carries_nothing_run_specific():
    """The cacheable prefix must not vary with the business date.

    This is the whole point of splitting the prompt. The business-date block
    used to sit second, ahead of `cannot`, the caveats and the schema card,
    so everything after it changed daily and could never be served from a
    prompt cache.
    """
    for skill_id, registered in all_skills().items():
        manifest = registered.manifest
        static = render_static_skill_prompt(manifest)
        # Stability under a changing business date is the actual invariant.
        # Grepping for date-shaped text would be wrong: a playbook may
        # legitimately contain a date as a worked example.
        assert "TODAY IS" not in static, (
            f"{skill_id}: the injected business date leaked into the cacheable prefix"
        )
        # rendering twice must be byte-identical, or nothing can cache it
        assert static == render_static_skill_prompt(manifest)
        for date in ("2026-11-04", "2027-06-30"):
            assert render_skill_system_prompt(manifest, date).startswith(static), (
                f"{skill_id}: business date {date} perturbs the prefix"
            )


def test_the_business_date_is_appended_after_the_static_prefix():
    manifest = all_skills()["order_analysis"].manifest
    static = render_static_skill_prompt(manifest)
    for date in ("2026-11-04", "2027-01-01"):
        full = render_skill_system_prompt(manifest, date)
        assert full.startswith(static), (
            "the volatile tail must come AFTER the static prefix, not inside it"
        )
        assert date in full


def test_the_prompt_tells_the_model_not_to_look_the_date_up():
    """The date is resolved before the skill runs; the skill prompts otherwise
    say 'retrieve real values, never recall', which would send it looking."""
    manifest = all_skills()["order_analysis"].manifest
    rendered = render_skill_system_prompt(manifest, "2026-11-04")
    assert "do not spend a tool step looking it up" in rendered
