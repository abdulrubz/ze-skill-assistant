"""PipelineState. TypedDict is the recommended state shape in LangGraph 1.x;
reducers are used only where parallel tasks append rather than replace."""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from app.agents.manifest import AgentManifest
from app.deps import Caller
from app.skills.answer import Answer
from app.skills.envelope import SkillExecutionResult


def replace_or_append(
    left: list[SkillExecutionResult] | None,
    right: list[SkillExecutionResult] | None,
) -> list[SkillExecutionResult]:
    """Merge fan-out results by `step_index`; a later result REPLACES its own
    step. `None` clears everything (used when Plan produces a genuinely new
    plan, whose step indices mean something different).

    Plain `add` concatenated a failed attempt onto its retry, so the user saw
    every finding twice -- the wrong version and the corrected one stitched
    into one answer, stale facts still cited.

    Merging per step is what makes a targeted retry possible: re-running only
    the step that failed validation replaces that step's result and leaves
    every other skill's correct work untouched. Previously one skill's bad
    fact discarded and re-ran all of them.
    """
    if right is None:
        return []
    merged = {ex.step_index: ex for ex in (left or [])}
    for ex in right:
        merged[ex.step_index] = ex
    return [merged[k] for k in sorted(merged)]


class AdmissionVerdict(TypedDict):
    allowed: bool
    reason: str | None


# What the deterministic resolve node found in the question: `business_date`,
# `time_window`, plus one key per pattern in domain/vocabulary.yaml `resolve`.
# A plain dict because its keys are the domain's, not the engine's.
ResolvedContext = dict


class PlanStep(TypedDict):
    skill_id: str
    inputs: dict[str, str]
    rationale: str
    # Indices of EARLIER steps whose findings this step needs. Empty for the
    # overwhelming majority of plans, which keeps the single parallel wave the
    # pipeline has always had. Only ever refers backwards -- app/graph/nodes/
    # plan.py drops anything else -- so a dependency cycle cannot be expressed
    # and the wave scheduler cannot deadlock.
    depends_on: list[int]


class Plan(TypedDict):
    steps: list[PlanStep]
    output_skill_id: str
    unanswerable: bool
    unanswerable_reason: str | None


class PipelineState(TypedDict):
    request_id: str
    thread_id: str
    question: str
    agent: AgentManifest
    caller: Caller

    admission: AdmissionVerdict | None
    resolved: ResolvedContext | None
    plan: Plan | None

    skill_results: Annotated[list[SkillExecutionResult], replace_or_append]
    validation_notes: Annotated[list[str], add]   # accumulated across retries, for the trace
    last_validation_problems: list[str]           # plain replace -- this attempt only, drives routing
    failed_step_indexes: list[int]                # plan steps to re-run; plain replace
    retry_count: int

    answer: Answer | None
    degraded: bool
