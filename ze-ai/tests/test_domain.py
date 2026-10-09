"""Guards for THIS domain's declarations (domain/). Replace them with your own
when you replace the dummy skills: they exist because a missing declaration
fails silently, as a polite refusal, rather than as an error.

The pattern worth keeping: a ranking question must reach a skill that DECLARES
it can rank. The planner reasons over declarations alone, so a capable skill
whose `answers` are all single-record examples reads as unable to rank.
"""

from __future__ import annotations

import pytest

from app.agents.manifest import load_all_agent_manifests
from app.graph.nodes.resolve import extract
from app.skills.registry import all_skills, load_and_validate, shortlist_skills

RANKING_QUESTIONS = [
    "who are our top 3 customers by spend",
    "which companies bring in the most money",
    "rank customers by revenue this year",
    "biggest customers in EUR",
]


@pytest.fixture(scope="module", autouse=True)
def _registry():
    load_and_validate()


@pytest.mark.parametrize("question", RANKING_QUESTIONS)
def test_a_ranking_question_shortlists_the_skill_that_can_rank(question):
    shortlisted = set(shortlist_skills(question, load_all_agent_manifests()["general"].skills))
    assert "order_analysis" in shortlisted, f"order_analysis not shortlisted for: {question}"


def test_order_analysis_declares_ranking_and_forbids_blending_currencies():
    m = all_skills()["order_analysis"].manifest
    assert "rank" in m.operations
    assert "rank_customers_by_spend" in m.tool_grants
    assert "currenc" in " ".join(m.cannot).lower()
    for slot in m.requires:
        assert slot.optional, f"{slot.slot} became required; tenant-wide questions would refuse"


def test_order_analysis_can_put_names_on_ids_from_the_other_service():
    """orders knows customers by id only; naming them needs the customers service."""
    assert "get_customer" in all_skills()["order_analysis"].manifest.tool_grants


def test_resolve_picks_up_this_domains_ids():
    r = extract("what did cust-1004 buy in order ORD-10005, in eur?")
    assert (r["customer_id"], r["order_id"], r["currency"]) == ("CUST-1004", "ORD-10005", "EUR")


def test_resolve_leaves_absent_ids_empty():
    r = extract("who are our top customers")
    assert r["customer_id"] is None and r["order_id"] is None
    assert r["business_date"]
