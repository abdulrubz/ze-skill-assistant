"""Always-on domain invariants, independent of any skill's own logic. These
encode things that are true of THIS system regardless of what was asked."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.skills.envelope import SkillExecutionResult


@dataclass
class InvariantReport:
    ok: bool
    problems: list[str] = field(default_factory=list)


def check_invariants(execution: SkillExecutionResult) -> InvariantReport:
    problems: list[str] = []
    sid = execution.skill_id
    by_ref = {f.ref: f for f in execution.result.facts}

    for f in execution.result.facts:
        # No FX table exists anywhere in this system, so combining amounts in
        # different currencies is always wrong, never a rounding nicety.
        if f.derivation and f.derivation.op in {"sum", "sub"}:
            currencies = {
                by_ref[r].currency for r in f.derivation.input_refs if r in by_ref
            }
            if len({c for c in currencies if c}) > 1:
                problems.append(
                    f"{sid}: fact {f.ref} combines amounts across currencies "
                    f"{sorted(c for c in currencies if c)} -- no exchange rates exist in this system"
                )

    # Any skill that reports on overdue state must carry the lag caveat,
    # because the stored state is known to trail reality in this dataset.
    if sid in {"dunning_collections", "data_quality_audit"}:
        narrative = execution.result.narrative.lower()
        mentions = any(k in narrative for k in ("overdue", "od1", "od2", "od3", "dunning"))
        has_caveat = any(
            k in c.lower() for c in execution.result.caveats for k in ("lag", "stale", "trail")
        )
        if mentions and not has_caveat:
            problems.append(
                f"{sid}: discusses overdue state without the standing lag caveat"
            )

    return InvariantReport(ok=not problems, problems=problems)
