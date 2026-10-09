"""SkillManifest -- what makes "a new skill is a manifest plus its tool grants"
literally true. Adding a skill is adding one YAML file to domain/skills/ plus
a prompt playbook in domain/skills/prompts/; the planner, the pipeline and every existing skill are untouched.

The manifest is also the ONLY thing the planner sees about a skill: it reasons
over the declarations, never over the skill's name or its implementation.

Coverage rests on TWO retrieval surfaces, and a skill needs both:

- `entities` + `operations` -- the structural surface. What the skill is about
  and which question shapes it serves, drawn from closed vocabularies in
  domain/vocabulary.yaml. This is what makes a skill findable by a phrasing
  nobody anticipated: "top 3 customers by spend" is (customer x rank)
  whatever words it uses.
- `answers` -- the lexical surface. Example questions carry nuance the axes
  cannot, and win outright when a user phrases something the skill explicitly
  anticipated. Write many, the way a user would actually say it.

Declaring only examples is how a perfectly capable skill becomes invisible:
in the system this was extracted from, a skill that could rank had ranking
questions refused purely because all twelve of its examples were about single
records. The failure surfaces as a polite refusal, the hardest kind to notice."""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from app.paths import SKILL_PROMPTS_DIR as PROMPTS_DIR, SKILLS_DIR as MANIFESTS_DIR
from app.skills.capability import Entity, Operation

CostClass = Literal["low", "medium", "high"]


class SlotRequirement(BaseModel):
    slot: str
    optional: bool = False


class SkillManifest(BaseModel):
    id: str
    kind: Literal["domain", "output"] = "domain"
    summary: str = Field(default="", description="One line, shown in the UI's skill catalog")
    answers: list[str] = Field(
        default_factory=list,
        description="Example questions, phrased as a user would. The planner's retrieval surface.",
    )
    # The STRUCTURAL retrieval surface, complementing `answers`.
    #
    # Examples can never be exhaustive, and a phrasing no example anticipated
    # made a perfectly capable skill invisible to the planner -- a failure that
    # surfaces as a polite refusal rather than an error. These two axes let a
    # question route on its shape instead: "top 3 customers by spend" is
    # (customer x rank) whatever words it uses.
    #
    # Closed vocabularies (domain/vocabulary.yaml), validated at boot by the
    # registry's self-test. Free-form intents drift instantly (`rank_orders`
    # vs `rank_customers` -- one operation, two names, matchable by neither).
    entities: list[Entity] = Field(
        default_factory=list,
        description="Domain nouns this skill is authoritative about (domain/vocabulary.yaml entities).",
    )
    operations: list[Operation] = Field(
        default_factory=list,
        description="Question shapes this skill can perform (domain/vocabulary.yaml operations).",
    )
    intents: list[str] = Field(
        default_factory=list,
        description="Free-form, human-facing labels. Kept for the UI and for lexical "
        "matching; routing decisions belong to entities/operations.",
    )
    requires: list[SlotRequirement] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    cannot: list[str] = Field(
        default_factory=list,
        description="Negative capability -- lets the planner decline instead of guessing",
    )
    caveats: list[str] = Field(
        default_factory=list, description="Copied into every SkillResult this skill produces"
    )
    tool_grants: list[str] = Field(default_factory=list)
    cost_class: CostClass = "low"
    max_tool_steps: int = 6
    prompt: str = Field(description="Filename under domain/skills/prompts/")
    prefer_table: bool = Field(
        default=False, description="Output skills only: render a table when facts look tabular"
    )

    @field_validator("entities", "operations")
    @classmethod
    def _no_duplicates(cls, v: list) -> list:
        if len(set(v)) != len(v):
            dupes = sorted({x for x in v if v.count(x) > 1})
            raise ValueError(f"duplicate capability terms: {dupes}")
        return v

    def prompt_text(self) -> str:
        path = PROMPTS_DIR / self.prompt
        if not path.exists():
            raise FileNotFoundError(f"skill '{self.id}': prompt file not found: {path}")
        return path.read_text()


def load_all_manifests() -> dict[str, SkillManifest]:
    manifests: dict[str, SkillManifest] = {}
    for path in sorted(MANIFESTS_DIR.glob("*.yaml")):
        manifest = SkillManifest.model_validate(yaml.safe_load(path.read_text()))
        if manifest.id != path.stem:
            raise ValueError(f"skill id '{manifest.id}' must match filename '{path.stem}.yaml'")
        if manifest.id in manifests:
            raise ValueError(f"duplicate skill id '{manifest.id}'")
        manifests[manifest.id] = manifest
    return manifests
