"""Output skills consume the already-gathered, already-validated SkillResults
and render the final Answer.

Nothing model-written carries a number. Blocks are built deterministically
from real Facts, each skill's cited narrative is kept verbatim, and the only
LLM involvement is a short orienting lead-in that is discarded if it contains
a digit or a citation marker."""

from __future__ import annotations

import re
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer

from app.config import get_settings
from app import prompt_store
from app.llm import message_text, named, prose_llm
from app.obs.emit import emit_model_usage
from app.obs.ledger import PROSE
from app.skills.answer import Answer, MetricBlock, TableBlock, TableColumn, TextSpan
from app.skills.envelope import Fact, SkillExecutionResult

_REF = re.compile(r"\[([a-zA-Z0-9_]+)\]")

# The lead-in is deliberately citation-FREE.
#
# An earlier version asked the model to weave several skills' narratives into
# one cited paragraph. It reliably reused valid [f_xx] markers but attached
# them to the wrong words -- producing "charged through USD", "billing period
# 990.00 USD" and "active since 0.00 USD". Every marker existed, so reference
# closure passed, yet the prose was nonsense: a citation that sits beside the
# wrong words is worse than no citation, because it looks authoritative.
#
# So the model now only writes a short orienting summary with NO numbers and
# NO markers, and each skill's own already-correct narrative follows verbatim.
# Readability where it is safe; determinism where it matters.
# The lead-in prompt is editable in Langfuse as `pipeline/lead-in`; the repo
# copy (and fallback) is app/prompts/lead_in.md. See app/prompt_store.py.


async def _lead_in(question: str, usable: list[SkillExecutionResult]) -> str:
    """A short, citation-free orientation sentence. Returns "" rather than
    risking a bad one: if the model slips a citation marker or a digit in, it
    is discarded, because the whole point of this shape is that nothing
    model-written carries a number."""
    transcript = "\n\n".join(f"[{r.skill_id}] {r.result.narrative}" for r in usable)
    system = prompt_store.pipeline(prompt_store.LEAD_IN)
    try:
        with prompt_store.linked(system):
            response = await named(prose_llm(), "prose").ainvoke([
                SystemMessage(content=system.text),
                HumanMessage(content=f"Question: {question}\n\nFindings:\n{transcript}"),
            ])
    except Exception:
        return ""

    try:
        emit_model_usage(get_stream_writer(), response, role=PROSE,
                         model_hint=get_settings().prose_model)
    except Exception:  # noqa: BLE001 -- no writer outside a streamed run
        pass
    text = message_text(response.content).strip()
    if not text or _REF.search(text) or any(ch.isdigit() for ch in text):
        return ""
    return text


def _namespaced(results: list[SkillExecutionResult]) -> list[tuple[str, str, list[Fact]]]:
    """Returns (skill_id, narrative, facts) with refs made unique across skills.

    Each skill numbers its own facts from f_01, which is fine in isolation and
    silently corrupting once several skills are merged: three skills all define
    f_01, the merge keeps whichever came last, and citations in the first two
    narratives then resolve to another skill's unrelated number. In one
    three-skill run f_01 meant 0 failed payments, 1,480 transactions and 300
    accounts at the same time.

    So each result gets a prefix, applied to its facts AND to the markers in its
    own narrative, keeping the two in step. A single-skill answer is left alone
    -- there is nothing to collide with, and bare f_01 reads better.
    """
    if len(results) <= 1:
        return [(r.skill_id, r.result.narrative, list(r.result.facts)) for r in results]

    out: list[tuple[str, str, list[Fact]]] = []
    for i, r in enumerate(results, start=1):
        mapping = {f.ref: f"s{i}_{f.ref}" for f in r.result.facts}
        narrative = _REF.sub(
            lambda m: f"[{mapping.get(m.group(1), m.group(1))}]", r.result.narrative
        )
        facts = [
            f.model_copy(update={"ref": mapping[f.ref]}) for f in r.result.facts
        ]
        out.append((r.skill_id, narrative, facts))
    return out


def _display(fact: Fact) -> str:
    if fact.currency:
        return f"{fact.value_str} {fact.currency}"
    if fact.unit:
        return f"{fact.value_str} {fact.unit}"
    return fact.value_str


async def render_answer(
    question: str, results: list[SkillExecutionResult], *, allow_table: bool = False
) -> Answer:
    caveats = list(dict.fromkeys(c for r in results for c in r.result.caveats))
    usable = [r for r in results if r.result.status != "denied"]
    prepared = _namespaced(usable)
    by_ref = {f.ref: f for _, _, facts in prepared for f in facts}

    if not usable:
        narrative = "I could not gather anything I can stand behind for that question."
        if results:
            narrative = results[0].result.narrative or narrative
    elif len(prepared) == 1:
        narrative = prepared[0][1]
    else:
        # Each skill's own narrative is kept VERBATIM (bar the ref prefixing) --
        # its citations were placed by the skill that retrieved the values and
        # are correct. Only the uncited lead-in is model-written.
        body = "\n\n".join(n for _, n, _ in prepared)
        lead_in = await _lead_in(question, usable)
        narrative = f"{lead_in}\n\n{body}" if lead_in else body

    cited = [ref for ref in dict.fromkeys(_REF.findall(narrative)) if ref in by_ref]
    blocks: list = [TextSpan(text=narrative)]
    for ref in cited:
        f = by_ref[ref]
        blocks.append(MetricBlock(label=f.name.replace("_", " "), ref=ref, display=_display(f)))
    used = [by_ref[r] for r in cited]

    if allow_table:
        table, table_facts = _maybe_table(by_ref)
        if table is not None:
            blocks.append(table)
            used += [f for f in table_facts if f.ref not in cited]

    return Answer(blocks=blocks, skills_used=[r.skill_id for r in results],
                  facts=used, caveats=caveats)


def _maybe_table(by_ref: dict[str, Fact]) -> tuple[TableBlock | None, list[Fact]]:
    """Facts sharing a name but differing by entity are naturally a table --
    e.g. `invoice_balance` across several accounts."""
    grouped: dict[str, list[Fact]] = defaultdict(list)
    for f in by_ref.values():
        if f.entity:
            grouped[f.name].append(f)

    for name, facts in grouped.items():
        if len(facts) < 3:
            continue
        keys = sorted({k for f in facts for k in f.entity})
        if not keys:
            continue
        columns = [TableColumn(key=k, label=k.replace("_", " ")) for k in keys]
        columns.append(TableColumn(key="value", label=name.replace("_", " ")))
        rows = [
            {**{k: f.entity.get(k, "") for k in keys}, "value": _display(f)} for f in facts
        ]
        return TableBlock(title=name.replace("_", " ").title(), columns=columns, rows=rows), facts
    return None, []
