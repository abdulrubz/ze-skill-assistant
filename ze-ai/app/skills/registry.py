"""Loads every skill manifest, runs the boot self-test, compiles each into a
subgraph. Fails at STARTUP, not request time, if a manifest is wired wrong --
that is what makes "a new skill is just a manifest" a safe claim."""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.skills.capability import analyze_question, entities, operations
from app.skills.factory import build_skill_subgraph
from app.skills.manifest import SkillManifest, load_all_manifests
from app.skills.slots import resolve_produced_slots, skill_producible_slots
from app.tools.registry import all_specs, declare_tools, get_spec


class SkillRegistrationError(RuntimeError):
    pass


@dataclass
class RegisteredSkill:
    manifest: SkillManifest
    subgraph: object | None  # compiled graph for domain skills; None for output skills


_SKILLS: dict[str, RegisteredSkill] = {}


def _self_test(manifests: dict[str, SkillManifest]) -> list[str]:
    errors: list[str] = []
    RESOLVE_PRODUCED_SLOTS, SKILL_PRODUCIBLE_SLOTS = resolve_produced_slots(), skill_producible_slots()
    ENTITIES, OPERATIONS = entities(), operations()
    all_provided = set(RESOLVE_PRODUCED_SLOTS)
    for m in manifests.values():
        all_provided |= set(m.provides)

    for sid, m in manifests.items():
        if m.kind == "output" and m.tool_grants:
            errors.append(f"skill '{sid}': output skills must not grant tools (got {m.tool_grants})")

        access: dict[str, str] = {}
        for tool_name in m.tool_grants:
            try:
                access[tool_name] = get_spec(tool_name).access
            except KeyError:
                errors.append(f"skill '{sid}': tool_grant '{tool_name}' is not in domain/tools.yaml "
                              "(every tool a skill can use must be configured there)")

        for req in m.requires:
            if req.slot not in (RESOLVE_PRODUCED_SLOTS | SKILL_PRODUCIBLE_SLOTS):
                errors.append(f"skill '{sid}': requires unknown slot '{req.slot}' (add it to domain/vocabulary.yaml slots)")
            elif not req.optional and req.slot not in all_provided:
                errors.append(f"skill '{sid}': requires '{req.slot}' which nothing produces")

        for p in m.provides:
            if p not in SKILL_PRODUCIBLE_SLOTS:
                errors.append(f"skill '{sid}': provides undeclared slot '{p}' (add it to domain/vocabulary.yaml slots)")

        for e in m.entities:
            if e not in ENTITIES:
                errors.append(f"skill '{sid}': unknown entity '{e}' (domain/vocabulary.yaml entities: {list(ENTITIES)})")
        for o in m.operations:
            if o not in OPERATIONS:
                errors.append(f"skill '{sid}': unknown operation '{o}' (domain/vocabulary.yaml operations: {list(OPERATIONS)})")

        if m.kind == "domain" and not m.answers:
            errors.append(f"skill '{sid}': domain skills need `answers` -- examples are half the retrieval surface")
        if m.kind == "domain" and not (m.entities and m.operations):
            errors.append(
                f"skill '{sid}': domain skills need `entities` AND `operations` -- without them "
                f"the skill is invisible to any question its examples do not happen to match"
            )
        if m.kind == "output" and (m.entities or m.operations):
            errors.append(f"skill '{sid}': output skills render answers; they own no entity or operation")

        try:
            m.prompt_text()
        except FileNotFoundError as exc:
            errors.append(str(exc))

    # Every registered tool must be reachable. A tool no skill grants is dead
    # weight the model can never call -- and the self-test only ever checked the
    # other direction (grants resolve to tools).
    granted = {t for m in manifests.values() for t in m.tool_grants}
    for name, spec in all_specs().items():
        if spec.access == "read" and name not in granted:
            errors.append(
                f"tool '{name}' is registered but granted by no skill -- grant it or delete it"
            )

    # Whole-catalog coverage. A vocabulary term no skill claims is a hole the
    # planner can only answer with a refusal. Failing at boot makes the gap
    # loud instead of silent: claim it in a skill or drop it from the vocabulary.
    domain = [m for m in manifests.values() if m.kind == "domain"]
    if domain:
        for entity in ENTITIES:
            if not any(entity in m.entities for m in domain):
                errors.append(f"capability gap: no skill owns entity '{entity}'")
        for operation in OPERATIONS:
            if not any(operation in m.operations for m in domain):
                errors.append(f"capability gap: no skill performs operation '{operation}'")

    return errors


def load_and_validate() -> dict[str, RegisteredSkill]:
    declare_tools()
    manifests = load_all_manifests()
    errors = _self_test(manifests)
    if errors:
        raise SkillRegistrationError(
            f"{len(errors)} skill manifest problem(s):\n" + "\n".join(f"  - {e}" for e in errors)
        )
    for sid, m in manifests.items():
        _SKILLS[sid] = RegisteredSkill(
            manifest=m, subgraph=build_skill_subgraph(m) if m.kind == "domain" else None
        )
    return _SKILLS


def get_skill(skill_id: str) -> RegisteredSkill:
    return _SKILLS[skill_id]


def all_skills() -> dict[str, RegisteredSkill]:
    return dict(_SKILLS)


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "of", "to", "for",
    "in", "on", "at", "by", "and", "or", "our", "we", "us", "this", "that", "these",
    "those", "what", "which", "who", "whom", "how", "when", "where", "why", "show", "me",
    "give", "list", "tell", "get", "have", "has", "had", "can", "could", "would", "should",
    "with", "from", "it", "its", "be", "been", "as", "if", "any", "all", "much", "many",
}


def _tokens(text: str) -> set[str]:
    return {
        t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split()
        if len(t) > 2 and t not in _STOPWORDS
    }


def shortlist_skills(question: str, skill_ids: list[str], *, k: int = 8) -> list[str]:
    """Lexical prefilter over each skill's declared `answers` before planning.

    The rich `answers` surface is what makes broad coverage possible, but
    sending every skill's full declarations on every request costs ~4.9k prompt
    tokens and dominates end-to-end latency on local inference. Scoring the
    question against those same declarations and passing only the top-k to the
    planner keeps the coverage benefit without paying for all of it each time.

    Deliberately recall-oriented and deterministic: output skills are ALWAYS
    kept (the planner must pick one), k is generous relative to the catalog,
    and if the question has no lexical signal at all every skill is passed
    through rather than guessing.
    """
    q = _tokens(question)
    shape = analyze_question(question)
    domain, output = [], []
    for sid in skill_ids:
        skill = _SKILLS.get(sid)
        if skill is None:
            continue
        (output if skill.manifest.kind == "output" else domain).append(sid)

    if len(domain) <= k or (not q and shape.empty):
        return skill_ids

    scored: list[tuple[float, str]] = []
    for sid in domain:
        m = _SKILLS[sid].manifest

        # STRUCTURAL score first. This is what lets an unanticipated phrasing
        # still route: "top 3 customers by spend" is (customer x rank)
        # regardless of wording, and any skill declaring that pair scores here
        # with zero matching examples.
        #
        # Entity and operation are multiplied, not added: a skill that owns the
        # right noun but cannot perform the operation is the WRONG skill, and
        # summing would let a strong entity match drag it in anyway. Ranking
        # orders does not belong to a skill that can only look them up.
        entity_hit = len(shape.entities & set(m.entities))
        op_hit = len(shape.operations & set(m.operations))
        if shape.entities and shape.operations:
            structural = 3.0 * (entity_hit > 0) * (op_hit > 0) + 0.4 * entity_hit + 0.3 * op_hit
        else:
            # Only one axis was detected; fall back to additive so a half-shaped
            # question still gets a signal rather than zero everywhere.
            structural = 0.6 * entity_hit + 0.5 * op_hit

        # LEXICAL score second: the `answers` examples carry nuance the axes
        # cannot express, so they still break ties and still win on a phrasing
        # a skill explicitly anticipated.
        best_answer = max(
            (len(q & _tokens(a)) / (len(_tokens(a)) ** 0.5 or 1) for a in m.answers), default=0.0
        )
        breadth = len(q & _tokens(" ".join(m.answers + m.intents + [m.summary]))) * 0.15
        scored.append((structural + best_answer + breadth, sid))

    scored.sort(key=lambda s: (-s[0], s[1]))
    if scored[0][0] == 0.0:  # no signal on either surface -- don't narrow blindly
        return skill_ids

    # A skill matching on BOTH axes is never cut by the top-k. The prefilter's
    # only real failure mode is hiding a capable skill from the planner, and
    # every routing bug found so far has been exactly that -- the planner
    # declining a question it was never shown the answer to. Lexical rank is a
    # guess; a declared (entity x operation) match is the skill saying "this is
    # mine", so it outranks the cut entirely.
    kept = [sid for _, sid in scored[:k]]
    for _, sid in scored:
        if sid in kept:
            continue
        m = _SKILLS[sid].manifest
        if (shape.entities & set(m.entities)) and (shape.operations & set(m.operations)):
            kept.append(sid)
    return kept + output


def project_for_planner(skill_ids: list[str]) -> list[dict]:
    """The catalog the Plan node sees: DECLARATIONS ONLY. No tool names, no
    implementation detail, no hint of how a skill works internally -- this is
    the mechanism behind "the planner reasons over declarations, never over
    domain names"."""
    out = []
    for sid in skill_ids:
        skill = _SKILLS.get(sid)
        if skill is None:
            continue
        m = skill.manifest
        out.append({
            "id": m.id,
            "kind": m.kind,
            "summary": m.summary,
            # The structural axes, shown BEFORE the examples so the planner
            # matches on question shape first and falls back to phrasing.
            "entities": m.entities,
            "operations": m.operations,
            # `intents` is deliberately omitted: it is largely redundant with
            # `answers` for selection and costs prompt tokens on every call.
            # It is still used by shortlist_skills() for prefiltering.
            # `answers` is 39% of this prompt. Measured with the examples off:
            #   with examples    61/61 default + 20/20 unanticipated, ~3200 tok
            #   without examples 60/61 default + 20/20 unanticipated, ~1900 tok
            # (measured on the 15-skill catalog this engine came from). So
            # examples are NOT what makes routing work -- entities/operations
            # carry that. They earn their place on the nuance the axes cannot
            # express, when several skills declare the same (entity x operation).
            # Flip ZE_PLANNER_ANSWERS=0 to re-measure as the catalog grows.
            **({} if os.getenv("ZE_PLANNER_ANSWERS") == "0" else {"answers": m.answers}),
            "requires": [{"slot": r.slot, "optional": r.optional} for r in m.requires],
            "provides": m.provides,
            "cannot": m.cannot,
            "prefer_table": m.prefer_table if m.kind == "output" else None,
        })
    return out
