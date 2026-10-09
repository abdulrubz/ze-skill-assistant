"""Join: deterministic. skill_results are already merged by the state reducer
as each parallel task finishes; this node's job is conflict DETECTION between
skills, which is mechanical. Only conflict NARRATION would need a model, and
the output skill already does that."""

from __future__ import annotations

from collections import defaultdict

from app.graph.state import PipelineState


async def join(state: PipelineState) -> dict:
    seen: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for execution in state["skill_results"]:
        for fact in execution.result.facts:
            seen[(fact.name, tuple(sorted(fact.entity.items())))].append(
                (execution.skill_id, fact.value_str)
            )

    notes = []
    for (name, entity), contributions in seen.items():
        values = {v for _, v in contributions}
        if len(values) > 1:
            skills = ", ".join(sorted({s for s, _ in contributions}))
            notes.append(
                f"conflict: '{name}' for {dict(entity)} differs across skills ({skills}): {sorted(values)}"
            )
    return {"validation_notes": notes} if notes else {}
