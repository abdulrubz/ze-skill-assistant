"""Deterministic groundedness checks -- zero LLM. Run against every
SkillResult before it may reach the output skill. This is what makes "every
number traces back to a real tool call" checkable rather than hoped for."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

from app.skills.envelope import Fact, SkillExecutionResult

_REF = re.compile(r"\[([a-zA-Z0-9_]+)\]")
_ALLOWED_OPS = {"sum", "sub", "ratio", "count", "pct"}
_TOLERANCE = Decimal("0.02")

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Dates in prose, not just ISO form. "October 2026" and a bare "2026" were being
# flagged as uncited claims, which no retry can fix -- the model rewrites the
# sentence, says "October 2026" again, and fails identically. One question burned
# two replans and $0.29 before shipping degraded for this reason alone. A
# four-digit year in this range is a date reference, never a measurement.
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_QUARTER = re.compile(r"\bq[1-4]\b", re.IGNORECASE)
# Analysis PARAMETERS, not claims. "within the last 90 days" states the window
# the skill chose to look at; there is no fact to cite because nothing was
# retrieved. Flagging these was costing a full replan -- one run burned $0.21
# and two retries, then shipped degraded anyway, over the 90 in "last 90 days".
# "top 5 accounts" is the same shape: a parameter of the question.
_WINDOW = re.compile(
    r"\b(?:last|past|previous|trailing|next|within|over)\s+(?:the\s+)?\d[\d,]*\s*"
    r"(?:day|week|month|quarter|year)s?\b",
    re.IGNORECASE,
)
_TOP_N = re.compile(r"\b(?:top|first|bottom)\s+\d+\b", re.IGNORECASE)
# Ordered-list markers: "1." / "2)" at the start of a line. Ranked answers are
# naturally numbered lists, and the marker is typography, not a measurement.
# Flagging them cost two replans and a degraded answer on a collections
# question, and no retry could fix it -- the model renumbers the list and fails
# identically. Anchored to line start so "owes 22 dollars" is untouched.
_LIST_MARKER = re.compile(r"^\s*\d+\s*[.)]\s+", re.MULTILINE)
# A digit run that is NOT glued to a letter, underscore, hyphen or dot. The
# boundaries are what keep OD1, OD1-OD3, f_01 and v_invoice_balance out of it.
_BARE_NUMBER = re.compile(r"(?<![A-Za-z0-9_.\-])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9_\-])")


@dataclass
class GroundednessReport:
    ok: bool
    problems: list[str] = field(default_factory=list)


def check_skill_result(execution: SkillExecutionResult) -> GroundednessReport:
    problems: list[str] = []
    sid = execution.skill_id
    by_ref = {f.ref: f for f in execution.result.facts}
    real_tool_calls = {t.tool_call_id for t in execution.tool_trace}

    # 1. Reference closure, BOTH directions.
    #    (a) every [ref] in the narrative must be a real fact, and
    unknown = set(_REF.findall(execution.result.narrative)) - by_ref.keys()
    if unknown:
        problems.append(f"{sid}: narrative cites unknown fact ref(s) {sorted(unknown)}")

    #    (b) every number in the narrative must BE a [ref].
    #
    #    Only (a) existed for a long time, which let an uncited number sail
    #    through untouched: an answer asserting "only 2 became delinquent after
    #    the change, the other 2 were already delinquent" validated clean, and
    #    the real split was 3 and 1. Uncited numbers are the dangerous ones --
    #    they are the model's own arithmetic or recollection, with nothing
    #    behind them, and they read exactly as authoritative as a real fact.
    problems += _check_no_uncited_numbers(sid, execution.result.narrative)

    for f in execution.result.facts:
        # 2. Value/kind agreement -- catches the model labelling a currency
        #    code or a status as a decimal, which would make any downstream
        #    arithmetic nonsense.
        problems += _check_value(sid, f)

        # 3. Provenance must point at a tool call that actually happened.
        if not f.provenance.tool_call_id:
            problems.append(f"{sid}: fact {f.ref} has no provenance.tool_call_id")
        elif (
            real_tool_calls
            and f.provenance.source != "derived"
            and f.provenance.tool_call_id not in real_tool_calls
        ):
            problems.append(
                f"{sid}: fact {f.ref} cites tool_call_id '{f.provenance.tool_call_id}' "
                f"which is not in this skill's tool trace"
            )

        # 4. Derivation replay -- recompute and compare.
        if f.derivation is not None:
            problems += _check_derivation(sid, f, by_ref)

    return GroundednessReport(ok=not problems, problems=problems)


def _check_value(sid: str, f: Fact) -> list[str]:
    if f.value_kind == "decimal":
        try:
            Decimal(f.value_str)
        except InvalidOperation:
            return [f"{sid}: fact {f.ref} value_kind=decimal but value_str='{f.value_str}' is not a number"]
    elif f.value_kind == "int":
        if not f.value_str.lstrip("-").isdigit():
            return [f"{sid}: fact {f.ref} value_kind=int but value_str='{f.value_str}' is not an integer"]
    elif f.value_kind == "date":
        try:
            date.fromisoformat(f.value_str)
        except ValueError:
            return [f"{sid}: fact {f.ref} value_kind=date but value_str='{f.value_str}' is not an ISO date"]
    elif f.value_kind == "bool":
        if f.value_str.lower() not in {"true", "false"}:
            return [f"{sid}: fact {f.ref} value_kind=bool but value_str='{f.value_str}'"]
    return []


def _check_no_uncited_numbers(sid: str, narrative: str) -> list[str]:
    """Flag numerals the narrative states without citing a fact.

    Citation markers and ISO dates are removed first: a marker renders as the
    number for the reader, and a business date is context rather than a claim.
    Digits attached to a word are left alone, so OD1, OD1-OD3 and plan names
    like growth-annual-v2 do not trip it.
    """
    stripped = _LIST_MARKER.sub(" ", narrative)
    stripped = _ISO_DATE.sub(" ", _REF.sub(" ", stripped))
    stripped = _YEAR.sub(" ", _QUARTER.sub(" ", stripped))
    stripped = _TOP_N.sub(" ", _WINDOW.sub(" ", stripped))
    bare = list(dict.fromkeys(_BARE_NUMBER.findall(stripped)))
    if not bare:
        return []
    return [
        f"{sid}: narrative states number(s) {bare} without citing a fact -- "
        f"every number must be a [f_xx] reference to a fact you retrieved, or "
        f"be removed"
    ]


def _tolerance_for(value_str: str) -> Decimal:
    """How far a replayed derivation may sit from the stated value.

    The CLAIM'S OWN PRECISION sets the bar. A flat 0.02 rejected a percentage
    written as "13.7" when the exact quotient was 13.6734 -- a correct value,
    correctly rounded to one decimal place, failed validation and cost a replan.
    Stating "13.7" only asserts the value to within +/-0.05; stating "13.67"
    asserts it to +/-0.005 and is held to that.

    The floor stays at 0.02 so integers and whole currency amounts keep the old
    strictness -- a restated 1204 for 1240 must still fail.
    """
    _, _, fraction = value_str.strip().partition(".")
    digits = len(fraction) if fraction.isdigit() else 0
    return max(_TOLERANCE, Decimal("0.5") * (Decimal(10) ** -digits))


def _check_derivation(sid: str, f: Fact, by_ref: dict[str, Fact]) -> list[str]:
    d = f.derivation
    if d.op not in _ALLOWED_OPS:
        return [f"{sid}: fact {f.ref} uses disallowed derivation op '{d.op}'"]
    missing = [r for r in d.input_refs if r not in by_ref]
    if missing:
        return [f"{sid}: fact {f.ref} derivation references missing input(s) {missing}"]

    try:
        target = Decimal(f.value_str)
        values = [Decimal(by_ref[r].value_str) for r in d.input_refs]
    except InvalidOperation:
        return []  # already reported by _check_value

    if d.op == "sum":
        computed = sum(values, Decimal(0))
    elif d.op == "sub":
        if len(values) != 2:
            return [f"{sid}: fact {f.ref} op 'sub' needs exactly 2 inputs"]
        computed = values[0] - values[1]
    elif d.op == "ratio":
        if len(values) != 2 or values[1] == 0:
            return [f"{sid}: fact {f.ref} op 'ratio' needs 2 inputs with a non-zero denominator"]
        computed = values[0] / values[1]
    elif d.op == "pct":
        if len(values) != 2 or values[1] == 0:
            return [f"{sid}: fact {f.ref} op 'pct' needs 2 inputs with a non-zero denominator"]
        computed = values[0] / values[1] * Decimal(100)
    else:  # count
        computed = Decimal(len(values))

    if abs(computed - target) > _tolerance_for(f.value_str):
        problem = (
            f"{sid}: fact {f.ref} claims {target} but replaying '{d.op}' over "
            f"{[str(v) for v in values]} gives {computed}"
        )
        # A failed check that does not say how to fix it just burns the retry
        # budget -- the model re-made the identical ratio/pct mistake twice
        # before this hint existed. Name the likely correction.
        if d.op == "ratio" and abs(computed * 100 - target) <= _TOLERANCE:
            problem += " -- this value is a percentage, so op should be 'pct', not 'ratio'"
        elif d.op == "pct" and abs(computed / 100 - target) <= _TOLERANCE:
            problem += " -- this value is a bare fraction, so op should be 'ratio', not 'pct'"
        return [problem]
    return []
