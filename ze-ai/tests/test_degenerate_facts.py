"""A "fact" whose entire value is a currency code carries no information and
actively corrupts the rendered answer: the renderer substitutes a citation
marker with the fact's value, so a standalone "USD" fact turns
"charged through [f_04]" into "charged through USD", and renders amounts as
"990.00 USD USD". Observed in a real cross-domain run before this filter."""

from __future__ import annotations

import pytest

from app.skills.envelope import Fact, Provenance
from app.skills.factory import _is_degenerate, _strip_refs


def _fact(ref: str, name: str, value: str, kind: str = "str", currency: str | None = None) -> Fact:
    return Fact(
        ref=ref, name=name, value_str=value, value_kind=kind, currency=currency,
        provenance=Provenance(tool_call_id="tc1", tool="get_account", source="api", locator="x"),
    )


@pytest.mark.parametrize("value", ["USD", "EUR", "GBP", "usd", " eur "])
def test_bare_currency_code_is_degenerate(value):
    assert _is_degenerate(_fact("f_04", "currency", value))


@pytest.mark.parametrize("name", ["currency", "currency_code", "unit", "unit_type"])
def test_facts_named_as_a_unit_are_degenerate(name):
    assert _is_degenerate(_fact("f_04", name, "api-calls"))


def test_a_real_amount_is_never_degenerate():
    assert not _is_degenerate(_fact("f_03", "account_balance", "990.00", "decimal", "USD"))


def test_a_meaningful_string_fact_is_kept():
    assert not _is_degenerate(_fact("f_01", "plan_name", "growth-annual"))
    assert not _is_degenerate(_fact("f_02", "overdue_state", "OD3"))
    assert not _is_degenerate(_fact("f_16", "overdue_state", "(none)"))


def test_a_count_is_never_degenerate():
    assert not _is_degenerate(_fact("f_05", "unpaid_invoices", "61", "int"))


def test_stripping_a_dropped_ref_leaves_clean_prose():
    narrative = "They owe 990.00 USD [f_03] USD [f_04] with 0.00 [f_05] in credit ."
    out = _strip_refs(narrative, {"f_04"})
    assert "[f_04]" not in out
    assert "[f_03]" in out and "[f_05]" in out
    assert "  " not in out, f"double space left behind: {out!r}"
    assert " ." not in out, f"space before punctuation left behind: {out!r}"


def test_stripping_nothing_is_a_no_op():
    narrative = "Balance is 990.00 [f_03]."
    assert _strip_refs(narrative, set()) == narrative
