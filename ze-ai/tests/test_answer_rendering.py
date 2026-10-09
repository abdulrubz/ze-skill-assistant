"""Rendering rules that protect the one property the whole system rests on:
a number in the answer means what it sits next to."""

from __future__ import annotations

import pytest

from app.llm import message_text
from app.skills.envelope import Fact, Provenance, SkillExecutionResult, SkillResult
from app.skills.output import render_answer


def _fact(ref, name, value, currency=None, kind="decimal"):
    return Fact(ref=ref, name=name, value_str=value, value_kind=kind, currency=currency,
                provenance=Provenance(tool_call_id="t1", tool="get_account",
                                      source="api", locator="x"))


def _result(skill_id, narrative, facts):
    return SkillExecutionResult(
        skill_id=skill_id,
        result=SkillResult(skill_id=skill_id, status="ok", narrative=narrative, facts=facts),
    )


# --- message_text: Gemini returns content as typed blocks, not a string ------

def test_message_text_handles_a_plain_string():
    assert message_text("hello") == "hello"


def test_message_text_flattens_gemini_content_blocks():
    """str() on this leaks the Python repr AND the base64 thought signature
    straight into the answer -- observed in a real run."""
    content = [{"type": "text", "text": "Account is on growth-annual.",
                "extras": {"signature": "ErodCrcdARFNMg++98ho"}}]
    out = message_text(content)
    assert out == "Account is on growth-annual."
    assert "signature" not in out and "extras" not in out


def test_message_text_drops_non_text_blocks():
    content = [{"type": "thinking", "thinking": "secret"}, {"type": "text", "text": "visible"}]
    assert message_text(content) == "visible"


def test_message_text_is_safe_on_none_and_odd_shapes():
    assert message_text(None) == ""
    assert message_text([]) == ""


# --- multi-skill rendering ---------------------------------------------------

async def test_single_skill_narrative_is_passed_through_verbatim():
    facts = [_fact("f_01", "account_balance", "990.00", "USD")]
    answer = await render_answer("q", [_result("a", "They owe [f_01].", facts)])
    assert answer.blocks[0].text == "They owe [f_01]."


async def test_multi_skill_keeps_each_narrative_verbatim(monkeypatch):
    """The per-skill wording must survive the merge unchanged -- no model
    rewrites it. Only the citation markers are mechanically prefixed, and each
    still points at the fact its own skill placed there."""
    monkeypatch.setattr("app.skills.output._lead_in", _no_lead_in)
    a = _result("a", "Plan is growth-annual [f_01].", [_fact("f_01", "plan", "growth-annual", kind="str")])
    b = _result("b", "Next invoice is [f_02].", [_fact("f_02", "next_invoice", "990.00", "USD")])
    answer = await render_answer("q", [a, b])
    text = answer.blocks[0].text
    assert "Plan is growth-annual [s1_f_01]." in text
    assert "Next invoice is [s2_f_02]." in text
    resolved = {f.ref: f.value_str for f in answer.facts}
    assert resolved["s1_f_01"] == "growth-annual"
    assert resolved["s2_f_02"] == "990.00"


async def test_colliding_refs_across_skills_stay_distinct(monkeypatch):
    """THE bug this prefixing exists for. Skills number facts independently, so
    several define f_01. Merged naively, the last one won and every earlier
    narrative's [f_01] silently rendered another skill's unrelated number."""
    monkeypatch.setattr("app.skills.output._lead_in", _no_lead_in)
    a = _result("a", "There were [f_01] failed payments.",
                [_fact("f_01", "failed_payments", "0", kind="int")])
    b = _result("b", "Across [f_01] accounts.",
                [_fact("f_01", "total_accounts", "300", kind="int")])
    answer = await render_answer("q", [a, b])

    resolved = {f.ref: f.value_str for f in answer.facts}
    assert resolved["s1_f_01"] == "0"
    assert resolved["s2_f_01"] == "300"

    text = answer.blocks[0].text
    assert "There were [s1_f_01] failed payments." in text
    assert "Across [s2_f_01] accounts." in text
    # every marker in the prose must resolve to exactly one fact
    import re
    for ref in re.findall(r"\[([a-zA-Z0-9_]+)\]", text):
        assert ref in resolved


async def test_single_skill_refs_are_left_bare(monkeypatch):
    """Nothing to collide with, and bare f_01 reads better."""
    monkeypatch.setattr("app.skills.output._lead_in", _no_lead_in)
    a = _result("a", "They owe [f_01].", [_fact("f_01", "balance", "990.00", "USD")])
    answer = await render_answer("q", [a])
    assert answer.blocks[0].text == "They owe [f_01]."
    assert [f.ref for f in answer.facts] == ["f_01"]


async def test_metric_blocks_are_built_from_facts_not_prose(monkeypatch):
    monkeypatch.setattr("app.skills.output._lead_in", _no_lead_in)
    facts = [_fact("f_01", "account_balance", "990.00", "USD")]
    answer = await render_answer("q", [_result("a", "Owed: [f_01].", facts)])
    metrics = [b for b in answer.blocks if b.kind == "metric"]
    assert len(metrics) == 1
    assert metrics[0].display == "990.00 USD"   # currency from the field, never doubled
    assert metrics[0].ref == "f_01"


async def test_uncited_facts_do_not_become_metric_blocks(monkeypatch):
    monkeypatch.setattr("app.skills.output._lead_in", _no_lead_in)
    facts = [_fact("f_01", "balance", "990.00", "USD"), _fact("f_99", "unused", "1.00", "USD")]
    answer = await render_answer("q", [_result("a", "Only [f_01] matters.", facts)])
    assert {b.ref for b in answer.blocks if b.kind == "metric"} == {"f_01"}


async def _no_lead_in(question, usable):
    return ""
