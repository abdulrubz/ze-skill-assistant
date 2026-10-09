"""Validate: deterministic checks only. The "LLM used to repair" happens by
feeding these problems back into the next Plan call, not by asking a model to
judge its own grounding. Bounded by retry_count in state -- NOT by LangGraph's
recursion_limit, which is a separate super-step safety net."""

from __future__ import annotations

from app.graph.state import PipelineState
from app.validation.groundedness import check_skill_result
from app.validation.invariants import check_invariants


async def validate(state: PipelineState) -> dict:
    problems: list[str] = []
    # Attribute each problem to the STEP that produced it, so a retry can re-run
    # only that step. Recorded from the execution itself rather than parsed back
    # out of the message text, which would break the moment a message is reworded.
    failed: list[int] = []

    for execution in state["skill_results"]:
        if execution.result.status == "denied":
            continue  # nothing to ground; already reported as denied
        mine = (
            check_skill_result(execution).problems
            + check_invariants(execution).problems
        )
        if mine:
            failed.append(execution.step_index)
            problems += mine

    update: dict = {
        "last_validation_problems": problems,
        "validation_notes": problems,
        "failed_step_indexes": failed,
    }
    if problems:
        update["retry_count"] = state["retry_count"] + 1
    return update
