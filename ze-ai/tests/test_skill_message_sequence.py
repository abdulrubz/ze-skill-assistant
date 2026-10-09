"""The skill loop's stored message history must stay well-formed.

Gemini enforces that a turn containing function calls comes immediately after
a user turn or a function-response turn. An earlier version built the opening
system+task turns locally without persisting them, so from step two the stored
history began with an assistant/function-call turn and Gemini rejected the
request with a 400. Ollama had tolerated it, so the bug was invisible locally.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.skills.factory import build_skill_subgraph
from app.skills.manifest import SkillManifest
from app.tools.registry import declare_tools

declare_tools()


def _manifest() -> SkillManifest:
    return SkillManifest(
        id="customer_profile",  # any real manifest; we only build the graph
        kind="domain",
        answers=["x"],
        tool_grants=["get_customer"],
        prompt="customer_profile.md",
    )


async def test_opening_turns_are_persisted_to_state(monkeypatch):
    """After the first agent step, state must contain System + Human + AI --
    not just the AI turn."""
    captured: dict = {}

    class FakeModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            captured["sent"] = list(messages)
            return AIMessage(content="done")

    monkeypatch.setattr("app.skills.factory.skill_llm", lambda: FakeModel())
    monkeypatch.setattr(
        "app.skills.factory.planner_llm",
        lambda: _FakeExtractor(),
    )

    graph = build_skill_subgraph(_manifest())
    out = await graph.ainvoke(
        {"skill_id": "s", "task": "what is the balance?", "business_date": "2026-11-04",
         "messages": [], "tool_steps": 0, "execution": None},
        context=None,
    )

    kinds = [type(m).__name__ for m in out["messages"]]
    assert kinds[0] == "SystemMessage", f"history starts with {kinds[0]}, not a system turn"
    assert kinds[1] == "HumanMessage", f"no user turn before the assistant turn: {kinds}"
    assert "AIMessage" in kinds


def test_a_function_call_turn_always_follows_a_user_or_tool_turn():
    """The invariant Gemini enforces, asserted directly against a realistic
    accumulated history."""
    history = [
        SystemMessage(content="sys"),
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "get_customer", "args": {}, "id": "1", "type": "tool_call"}]),
        ToolMessage(content="{}", tool_call_id="1"),
        AIMessage(content="", tool_calls=[{"name": "get_customer", "args": {}, "id": "2", "type": "tool_call"}]),
        ToolMessage(content="{}", tool_call_id="2"),
        AIMessage(content="final"),
    ]
    for i, msg in enumerate(history):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            assert i > 0, "a function-call turn cannot be first"
            prev = history[i - 1]
            assert isinstance(prev, (HumanMessage, ToolMessage)), (
                f"function-call turn at {i} follows {type(prev).__name__}"
            )


class _FakeExtractor:
    def with_structured_output(self, schema, include_raw=False):
        return self

    async def ainvoke(self, messages):
        from app.skills.envelope import SkillResult

        return {
            "parsed": SkillResult(skill_id="customer_profile", status="ok", narrative="ok"),
            "parsing_error": None,
            "raw": None,
        }
