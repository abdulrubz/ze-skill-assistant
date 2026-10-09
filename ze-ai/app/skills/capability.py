"""A capability grammar for routing: what a question is ABOUT x what it DOES.

Routing on lexical similarity to each skill's `answers` examples alone works
until a user phrases something no example anticipated -- and then a perfectly
capable skill is never offered to the planner. The failure looks like a polite
refusal, the hardest kind to notice. (In the system this engine came from, "the
companies with the top 3 invoices by dollar amount" was declined by a skill
that could answer it, because all its examples said "this invoice".)

Examples cannot be made exhaustive, so skills also declare their capability
STRUCTURALLY, on two axes drawn from closed vocabularies:

    entities   -- the domain's nouns (domain/vocabulary.yaml): customer, order...
    operations -- the shapes a question takes: rank, explain, trend, audit...

A question is decomposed onto the same two axes by `analyze_question`,
deterministically and with no model. "Top 3 customers by spend" is
(customer x rank) whatever words it uses, and any skill declaring that pair is
a candidate even with zero matching examples.

The two surfaces are complementary: examples carry nuance the axes cannot,
the axes carry generality examples cannot. Both feed the planner.

Vocabularies are CLOSED on purpose: free-form intents drift immediately
(`rank_orders` and `rank_customers` are one operation under two names), and a
typo in a closed vocabulary fails at boot instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.vocabulary import vocabulary

# Entities are the DOMAIN's nouns and come from domain/vocabulary.yaml.
# Operations are question shapes -- generic English, so their cues live here;
# the vocabulary picks which of them this domain supports.
Entity = str
Operation = str

_OPERATION_CUES: dict[str, tuple[str, ...]] = {
    "lookup": ("what is", "show me", "tell me about", "give me", "details", "profile", "status of"),
    "explain": (
        "why", "how come", "reason", "reasons", "explain", "cause", "caused",
        "because", "what happened", "root cause",
    ),
    # "who is"/"who has" sit in BOTH list and resolve. "Who is still in trial"
    # enumerates; "who is the Dubois account" disambiguates a name. Cheap to
    # claim both -- scoring is overlap-based, and a spurious candidate costs a
    # few planner tokens while a missing one costs a refused question.
    "list": (
        "which", "list", "show all", "who are", "who is", "who has", "who have",
        "what are", "find all", "enumerate", "anyone", "everyone",
    ),
    "aggregate": (
        "how many", "how much", "count", "total", "sum", "average", "mean",
        "percentage", "percent", "rate", "share", "proportion", "number of",
        "breakdown", "by currency", "per currency", "distribution",
    ),
    "rank": (
        "top", "largest", "biggest", "highest", "smallest", "lowest", "most",
        "least", "rank", "ranked", "ranking", "best", "worst", "leading",
        "bottom", "first few", "maximum", "minimum", "greatest",
    ),
    "compare": (
        "compare", "compared", "comparison", "versus", "vs", "difference between",
        "differ", "more than", "less than", "against", "relative to", "than those",
        "better or worse", "correlate", "correlated",
        # Comparative phrasing without the word "compare". "Did customers who
        # changed plan have MORE failures afterward" is a cohort comparison and
        # was extracting no operation at all before these.
        "stack up", "side by side", "against the ones", "have more", "had more",
        "have fewer", "had fewer", "more likely",
        "less likely", "higher", "lower", "increase", "increased", "decrease",
        "decreased", "afterward", "afterwards", "before and after",
    ),
    "trend": (
        "over time", "trend", "trending", "trajectory", "month over month",
        "by month", "monthly", "each month", "per month", "growing", "shrinking",
        "history", "historical", "last year", "since",
    ),
    "forecast": (
        "will", "upcoming", "projected", "projection", "forecast",
        "expect", "expected", "going to", "about to", "due next", "predict",
    ),
    "audit": (
        "inconsistent", "inconsistency", "mismatch", "mismatched", "missing",
        "wrong", "incorrect", "reconcile", "reconciliation", "audit", "anomaly",
        "anomalies", "stuck", "orphan*", "should not", "shouldn t",
        "data quality", "discrepanc*",
        # Idiomatic "these two things disagree". Negated forms only -- a bare
        # "add up" is an aggregate ("add up the totals"), not an audit.
        "line up", "lines up", "not add up", "add up", "out of sync", "disagree",
        "don t match", "does not match", "doesn t match", "look off", "looks off",
    ),
    "simulate": (
        "what if", "if we", "would happen", "simulate", "simulation", "hypothetical",
        "suppose", "were to", "impact of changing", "if the",
    ),
    "resolve": (
        "who is", "which one is", "find the", "called", "named",
        "more than one", "ambiguous", "same name",
    ),
}


@dataclass(frozen=True)
class QuestionShape:
    entities: frozenset[str] = field(default_factory=frozenset)
    operations: frozenset[str] = field(default_factory=frozenset)

    @property
    def empty(self) -> bool:
        return not self.entities and not self.operations


def _normalise(text: str) -> str:
    """Lowercase, punctuation to spaces, single-spaced and space-padded, so a
    phrase cue can be matched with plain containment and still respect word
    boundaries (' vs ' must not fire inside 'versus')."""
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def _matches(haystack: str, cues: tuple[str, ...]) -> bool:
    """A cue ending in '*' matches any word starting with it.

    Without this the cue lists have to enumerate every inflection, and they
    never will: "consuming" missed a list containing "consumption" and
    "consumed", which silently dropped usage_metering from the candidates.
    A stem costs one entry and covers the family.
    """
    for cue in cues:
        cue = cue.strip()
        if cue.endswith("*"):
            if re.search(rf"\b{re.escape(cue[:-1])}", haystack):
                return True
        elif f" {cue} " in haystack:
            return True
    return False


def analyze_question(question: str) -> QuestionShape:
    """Decompose a question onto (entities, operations). Deterministic, no model.

    Both axes may be empty -- an unrecognised question narrows nothing, which is
    the safe direction: callers fall back to passing every skill to the planner
    rather than guessing.
    """
    text = _normalise(question)
    vocab = vocabulary()
    entities = {e for e, cues in vocab.entities.items() if _matches(text, tuple(cues))}
    operations = {o for o in vocab.operations if _matches(text, _OPERATION_CUES[o])}

    # A bare "top 3 customers" is a ranking question even though no cue says
    # "rank"; likewise "the 5 biggest". Digits next to a ranking word are a
    # strong enough signal to add the operation when nothing else claimed it.
    if not operations and "rank" in vocab.operations and re.search(r"\b(top|first|bottom)\s+\d+\b", text):
        operations.add("rank")

    return QuestionShape(frozenset(entities), frozenset(operations))


ALL_OPERATIONS: tuple[str, ...] = tuple(_OPERATION_CUES)


def entities() -> tuple[str, ...]:
    return tuple(vocabulary().entities)


def operations() -> tuple[str, ...]:
    """The operations this domain supports, validated against the engine's."""
    ops = tuple(vocabulary().operations)
    unknown = [o for o in ops if o not in _OPERATION_CUES]
    if unknown:
        raise ValueError(f"domain/vocabulary.yaml: unknown operation(s) {unknown}; "
                         f"choose from {list(_OPERATION_CUES)}")
    return ops


def describe_vocabulary() -> dict[str, list[str]]:
    """For the planner prompt and the /skills endpoint."""
    return {"entities": list(entities()), "operations": list(operations())}
