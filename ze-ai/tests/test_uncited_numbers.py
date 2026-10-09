"""Reference closure must run in BOTH directions.

Only one half was implemented: every [ref] had to resolve to a real fact, but
nothing stopped the model stating a number with no ref at all. A real answer
asserted "only 2 became delinquent after the change date, while the other 2
were already delinquent" and validated completely clean. The true split was 3
and 1 -- and it was the single causal claim in the whole answer.

Uncited numbers are the dangerous ones: they are the model's own arithmetic,
nothing stands behind them, and they read exactly as authoritative as a fact
that came from a tool call.
"""

from __future__ import annotations

from app.validation.groundedness import _check_no_uncited_numbers as check


def test_an_uncited_number_is_flagged():
    problems = check("s", "Only 2 became delinquent after the change.")
    assert problems and "2" in problems[0]


def test_a_fully_cited_narrative_passes():
    assert check("s", "Of [f_01] accounts, [f_02] ([f_03]%) are overdue.") == []


def test_iso_dates_are_context_not_claims():
    assert check("s", "As of 2026-11-04, [f_01] accounts are overdue.") == []


def test_overdue_state_names_are_not_numbers():
    """OD1-OD3 is a state enum. Flagging it would make every collections answer
    fail and trigger pointless replans."""
    assert check("s", "In an active overdue state (OD1-OD3): [f_01] accounts.") == []
    assert check("s", "States OD1, OD2 and OD3 all count as overdue.") == []


def test_plan_names_containing_digits_are_left_alone():
    assert check("s", "[f_01] accounts are on growth-annual-v2 today.") == []


def test_decimals_and_thousands_separators_are_caught():
    problems = check("s", "They owe 1,240.50 in total.")
    assert problems and "1,240.50" in problems[0]


def test_the_message_names_every_offending_number():
    problems = check("s", "Of 61 accounts, 3 were overdue.")
    assert "61" in problems[0] and "3" in problems[0]


def test_duplicate_numbers_are_reported_once():
    problems = check("s", "2 before and 2 after.")
    assert problems[0].count("'2'") == 1


# --- analysis parameters are not claims -------------------------------------
#
# The first version of this check had no such exemption and flagged the 90 in
# "within the last 90 days". That is the window the skill chose to look at, not
# a retrieved value, so there is no fact to cite and no retry can fix it: one
# run burned $0.21 and two replans, then shipped degraded anyway.


def test_a_time_window_is_not_a_claim():
    assert check("s", "Plan changes within the last 90 days ([f_01] accounts).") == []
    assert check("s", "Customers who changed plan in the last 90 days: [f_01].") == []
    assert check("s", "Over the past 3 months, [f_01] churned.") == []


def test_top_n_is_a_parameter_not_a_claim():
    assert check("s", "Top 5 accounts by balance: [f_01].") == []


def test_an_age_in_days_is_still_a_claim():
    """The discriminating case. 'last 90 days' selects a window; '90 days
    overdue' asserts a measured age and must be grounded."""
    assert check("s", "The invoice is 90 days overdue.")


def test_hyphenated_durations_are_left_alone():
    assert check("s", "Measured over a 90-day window, [f_01] churned.") == []


# --- dates written in prose --------------------------------------------------
#
# Only ISO dates were stripped, so "October 2026" and a bare "2026" were flagged
# as uncited claims. No retry can fix that: the model rewrites the sentence,
# writes "October 2026" again, and fails identically. One adversarial question
# burned two replans and $0.29 and shipped degraded for this reason alone.


def test_a_year_in_prose_is_a_date_not_a_claim():
    assert check("s", "MRR grew in October 2026 to [f_01].") == []
    assert check("s", "As of 2026 the total was [f_01].") == []


def test_a_quarter_label_is_not_a_claim():
    assert check("s", "Revenue in Q4 was [f_01].") == []


def test_stripping_years_does_not_hide_real_numbers():
    """The exemption must not swallow measurements that merely sit near a year."""
    problems = check("s", "In October 2026, 14 customers churned.")
    assert problems and "14" in problems[0]


# --- derivation tolerance follows the claim's own precision -------------------

def test_tolerance_scales_with_stated_precision():
    """A flat 0.02 rejected a percentage written as "13.7" whose exact quotient
    was 13.6734 -- correct, correctly rounded, and it cost a replan. Stating
    "13.7" asserts the value only to +/-0.05; stating "13.67" asserts +/-0.005
    and is held to it."""
    from decimal import Decimal

    from app.validation.groundedness import _tolerance_for

    assert abs(Decimal("13.6734821") - Decimal("13.7")) <= _tolerance_for("13.7")
    assert abs(Decimal("13.6734821") - Decimal("13.67")) <= _tolerance_for("13.67")
    # ...but a genuinely wrong restatement still fails at every precision
    assert abs(Decimal("1240") - Decimal("1204")) > _tolerance_for("1204")
    assert abs(Decimal("990.04") - Decimal("990.00")) > _tolerance_for("990.00")


# --- ordered-list markers ----------------------------------------------------

def test_numbered_list_markers_are_typography_not_claims():
    """A ranked answer is naturally a numbered list. Flagging '1.' '2.' '3.'
    cost two replans and a degraded answer on a collections question, and no
    retry could fix it -- the model renumbers and fails identically."""
    assert check("s", "Top debtors:\n1. [f_01] owes [f_02]\n2. [f_03] owes [f_04]") == []


def test_a_list_marker_does_not_hide_the_claim_beside_it():
    """Only the marker is stripped, never the sentence."""
    problems = check("s", "1. The invoice is 90 days overdue.")
    assert problems and "90" in problems[0]


def test_inline_enumeration_is_still_flagged_deliberately():
    """Markers are matched at line start only. Broadening to mid-line would
    start swallowing real measurements, and generated narratives use newline
    lists -- that is the case that actually failed."""
    assert check("s", "1) [f_01]  2) [f_02]")
