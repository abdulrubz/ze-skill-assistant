"""The shared vocabulary the planner reasons over. Skills declare `requires`
and `provides` in these terms, never in terms of each other's names -- that is
what lets Plan compute selection from declarations rather than prose, and what
the boot self-test validates against. The domain's slots are listed in
domain/vocabulary.yaml."""

from __future__ import annotations

from app.vocabulary import vocabulary

# Always produced by the resolve node, whatever the domain.
BUILTIN_RESOLVED_SLOTS = {"business_date", "time_window"}


def resolve_produced_slots() -> set[str]:
    """Produced by the deterministic `resolve` node before planning."""
    return BUILTIN_RESOLVED_SLOTS | set(vocabulary().resolve)


def skill_producible_slots() -> set[str]:
    """Producible by one or more skills."""
    return set(vocabulary().slots)
