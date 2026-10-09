"""Plan: one of only two LLM calls in the pipeline (Validate is deterministic;
Join and Terminate were deliberately made deterministic too -- see the plan's
"deliberate simplifications"). Selects skills and the output skill, reasoning
over DECLARATIONS ONLY."""

from __future__ import annotations

import json
import time

from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.state import PipelineState, Plan, PlanStep
from app import prompt_store
from app.llm import named, planner_llm
from app.obs.emit import emit_model_usage
from app.obs.ledger import PLANNER
from app.skills.registry import all_skills, project_for_planner, shortlist_skills


class PlanStepOut(BaseModel):
    skill_id: str = Field(description="Must be one of the candidate DOMAIN skill ids")
    inputs: dict[str, str] = Field(
        default_factory=dict, description="slot -> value carried from the resolved context"
    )
    rationale: str = Field(description="One sentence: why this skill for this question")
    depends_on: list[int] = Field(
        default_factory=list,
        description="0-based indexes of EARLIER steps in this list whose findings this "
        "step needs before it can start. Leave empty unless the dependency is real -- "
        "independent steps run in parallel and are faster. Only ever refer to a LOWER "
        "index than this step's own.",
    )


class PlanOutput(BaseModel):
    # `steps` is intentionally REQUIRED with a strong description: given a
    # default it gets omitted by the grammar-constrained decoder and every
    # question comes back unanswerable.
    steps: list[PlanStepOut] = Field(
        description="REQUIRED and non-empty for any answerable question. One entry per "
        "DOMAIN-kind skill (kind='domain') you are selecting."
    )
    output_skill_id: str = Field(
        description="Exactly one OUTPUT-kind skill id (kind='output') to render the answer. "
        "NEVER a domain skill, and never also listed in steps."
    )
    unanswerable: bool = Field(default=False)
    unanswerable_reason: str | None = None


# The planner's system prompt is editable in Langfuse as `pipeline/planner`;
# the repo copy (and fallback) is app/prompts/planner.md. See app/prompt_store.py.


def keep_domain_steps(
    raw_steps: list[PlanStepOut], is_valid_domain_skill
) -> list[PlanStep]:
    """Filter the planner's steps and make their `depends_on` safe.

    Two corrections, both of which matter:

    - `depends_on` indexes the planner's OWN list, so dropping a step shifts
      every later position. A raw index carried through would then point at a
      different skill than the planner meant. Remap through the survivors.
    - Clamp what is left to STRICTLY BACKWARD references. That makes a cycle
      unexpressible rather than merely unlikely, which is what lets the wave
      scheduler in app/graph/routing.py assume it can always make progress.

    Pure and separate from `plan()` so it is testable without a model call --
    the deadlock it prevents would otherwise only show up in production.
    """
    remap: dict[int, int] = {}
    kept: list[PlanStepOut] = []
    for original_index, s in enumerate(raw_steps):
        if is_valid_domain_skill(s.skill_id):
            remap[original_index] = len(kept)
            kept.append(s)

    steps: list[PlanStep] = []
    for i, s in enumerate(kept):
        deps = {remap[d] for d in s.depends_on if d in remap}
        steps.append(PlanStep(
            skill_id=s.skill_id,
            inputs=s.inputs,
            rationale=s.rationale,
            depends_on=sorted(d for d in deps if d < i),
        ))
    return steps


async def plan(state: PipelineState) -> dict:
    agent = state["agent"]
    registry = all_skills()
    candidate_ids = [sid for sid in agent.skills if sid in registry]
    # Prefilter to the lexically-plausible skills before projecting -- see
    # shortlist_skills() for why this matters to end-to-end latency.
    candidates = project_for_planner(shortlist_skills(state["question"], candidate_ids))

    resolved = state.get("resolved") or {}
    lines = [f"Question: {state['question']}", ""]
    known = {k: v for k, v in resolved.items() if v}
    if known:
        lines.append("Resolved context:")
        lines += [f"- {k}: {v}" for k, v in known.items()]
        lines.append("")
    lines.append("Candidate skills (JSON):")
    lines.append(json.dumps(candidates, indent=2))

    if state.get("retry_count", 0) > 0 and state.get("last_validation_problems"):
        lines += [
            "",
            f"This is retry {state['retry_count']}. The previous attempt failed validation. "
            "Adjust the plan to address these -- pick a better-suited skill, or decline if "
            "nothing can substantiate the claim:",
        ]
        lines += [f"- {p}" for p in state["last_validation_problems"]]

    system = prompt_store.pipeline(prompt_store.PLANNER)
    started = time.monotonic()
    with prompt_store.linked(system):
        raw = await named(planner_llm(), "planner").with_structured_output(
            PlanOutput, include_raw=True
        ).ainvoke(
            [{"role": "system", "content": system.text},
             {"role": "user", "content": "\n".join(lines)}]
        )
    # Planning is not free and runs again on every replan, so it has to be in
    # the cost ledger -- it is bucketed as __planner__, not against a skill.
    try:
        emit_model_usage(
            get_stream_writer(), raw.get("raw"), role=PLANNER,
            model_hint=get_settings().endpoint_for("planner"),
            latency_ms=round((time.monotonic() - started) * 1000),
        )
    except Exception:  # noqa: BLE001 -- no writer outside a streamed run
        pass
    parsed: PlanOutput | None = raw["parsed"]

    # On a replan, discard the attempt that just failed validation. The results
    # are about to be regenerated, and keeping both would render each finding
    # twice -- once stale, once corrected -- in a single answer.
    fresh: dict = {"skill_results": None} if state.get("retry_count", 0) else {}

    if parsed is None:
        return {**fresh, "plan": Plan(
            steps=[], output_skill_id=_default_output(candidate_ids, registry),
            unanswerable=True,
            unanswerable_reason=f"the planner could not produce a usable plan ({raw.get('parsing_error')})",
        )}

    valid = set(candidate_ids)
    steps = keep_domain_steps(
        parsed.steps,
        lambda sid: sid in valid and registry[sid].manifest.kind == "domain",
    )

    output_id = parsed.output_skill_id
    if output_id not in valid or registry[output_id].manifest.kind != "output":
        output_id = _default_output(candidate_ids, registry)

    return {**fresh, "plan": Plan(
        steps=steps,
        output_skill_id=output_id,
        unanswerable=parsed.unanswerable or not steps,
        unanswerable_reason=parsed.unanswerable_reason,
    )}


def _default_output(candidate_ids: list[str], registry) -> str:
    for sid in candidate_ids:
        if registry[sid].manifest.kind == "output":
            return sid
    return "prose_synthesis"
