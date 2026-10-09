"""Compiles a SkillManifest into a runnable subgraph: a bounded ReAct loop
(model bound to ONLY this skill's granted tools <-> ToolNode carrying the
authz/trace wrapper), then a deterministic extraction step that turns the raw
tool transcript into a SkillResult of typed Facts.

Every skill goes through this same function -- that is what the plan's "new
skill = manifest + tool grants" claim rests on."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import get_runtime
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import get_settings
from app.deps import Deps
from app import prompt_store
from app.llm import describe_model_error, message_text, named, planner_llm, skill_llm
from app.obs.emit import emit_model_usage
from app.skills.envelope import Fact, SkillExecutionResult, SkillResult, ToolTrace
from app.skills.manifest import SkillManifest
from app.tools.authz import authorize_and_trace
from app.tools.registry import tools_for


class SkillState(TypedDict):
    skill_id: str
    task: str
    business_date: str | None
    prior_findings: str | None
    messages: Annotated[list, add_messages]
    tool_steps: int
    execution: SkillExecutionResult | None


def render_static_skill_prompt(manifest: SkillManifest) -> str:
    """Everything in a skill's system prompt that does not vary between runs.

    Kept as a stable PREFIX, with everything per-run after it: this text plus
    the tool schemas is re-sent on every step of a skill's tool loop (measured
    at ~42% of a run's input tokens), and providers that cache prompt prefixes
    can only serve it from cache if it never varies.
    """
    # The playbook prose is the editable part (Langfuse `skills/<id>`); the
    # manifest-derived sections below are not, so no edit can drop them.
    parts = [prompt_store.skill(manifest).text, ""]

    if manifest.cannot:
        parts.append("You CANNOT do the following -- say so plainly rather than guessing:")
        parts += [f"- {c}" for c in manifest.cannot]
        parts.append("")

    if manifest.caveats:
        parts.append("Standing caveats that must travel with your answer:")
        parts += [f"- {c}" for c in manifest.caveats]
        parts.append("")

    parts.append(
        f"You have at most {manifest.max_tool_steps} tool-calling steps -- use as few as you "
        "need. Retrieve real values; do not estimate or recall from memory. If a tool "
        "returns an error, read it and correct your call rather than giving up. When you "
        "have enough, stop calling tools and write a plain-language answer citing the "
        "specific values you retrieved; a separate step extracts the cited facts."
    )
    return "\n".join(parts)


def render_volatile_skill_context(
    manifest: SkillManifest,
    business_date: str | None,
    prior_findings: str | None = None,
) -> str:
    """The half of a skill's instructions that changes from run to run, kept
    after the static prefix (see render_static_skill_prompt)."""
    parts: list[str] = []

    if business_date:
        parts.append(
            f"TODAY IS {business_date}. Anchor every relative date ('this month', "
            "'last 90 days') on this date; do not spend a tool step looking it up."
        )

    if prior_findings:
        parts += [
            "## Already established by an earlier step",
            prior_findings,
            "Use this to SKIP work: it tells you which accounts, cohort, window or plans "
            "the question turned out to be about, so you do not spend your tool budget "
            "re-deriving it. It does NOT substitute for retrieval -- every value YOU "
            "report must come from a tool call YOU make in this step and be cited from "
            "it. Do not restate a number from above as if you had retrieved it; if you "
            "need it in your own answer, query for it.",
        ]

    # Restated LAST, deliberately. Splitting the prompt for cacheability pushed
    # the citation rule out of the final position -- and the text that displaced
    # it is dense with concrete numbers (the business date, an upstream step's
    # findings), which is exactly the material a model repeats uncited. Uncited
    # numbers fail groundedness, and a failure costs a retry that re-runs the
    # whole skill, so ~30 tokens here buy back far more.
    if parts:
        parts.append(
            "Reminder, and it is the rule most often broken: EVERY number you write must "
            "be one YOU retrieved by a tool call in THIS step. Do not carry a number over "
            "from the context above, and do not write a [f_xx] marker yourself -- a "
            "separate step adds those."
        )

    return "\n\n".join(parts)


def render_skill_system_prompt(
    manifest: SkillManifest,
    business_date: str | None,
    prior_findings: str | None = None,
) -> str:
    """Static prefix followed by the volatile context."""
    volatile = render_volatile_skill_context(manifest, business_date, prior_findings)
    static = render_static_skill_prompt(manifest)
    return f"{static}\n\n{volatile}" if volatile else static


# The extraction instructions are editable in Langfuse as `pipeline/extraction`;
# the repo copy (and fallback) is app/prompts/extraction.md. See app/prompt_store.py.


# Per tool result, in the extraction transcript. Longer results are cut, and
# the cut is announced: a silently truncated result reads as complete, so the
# extractor reports its first rows as the whole answer (seen with a per-currency
# ranking whose last currency fell past the limit).
TRANSCRIPT_RESULT_CHARS = 2500


def _clip_result(content: str) -> str:
    if len(content) <= TRANSCRIPT_RESULT_CHARS:
        return content
    cut = len(content) - TRANSCRIPT_RESULT_CHARS
    return (f"{content[:TRANSCRIPT_RESULT_CHARS]} ...[TRUNCATED: {cut} more characters not shown. Do not "
            "present this result as complete; say which part is missing]")


def _transcript(messages: list) -> tuple[str, set[str]]:
    lines: list[str] = []
    real_ids: set[str] = set()
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"TASK: {m.content}")
        elif isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                real_ids.add(tc["id"])
                lines.append(f"CALLED {tc['name']}(tool_call_id={tc['id']}) args={tc['args']}")
            text = message_text(m.content)
            if text:
                lines.append(f"ASSISTANT: {text}")
        elif isinstance(m, ToolMessage):
            lines.append(f"RESULT for tool_call_id={m.tool_call_id}: {_clip_result(str(m.content))}")
    return "\n".join(lines), real_ids



# Common ISO currency codes. A "fact" whose entire value is one of these carries
# no information: the currency already belongs on the amount it describes.
_CURRENCY_CODES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "SEK", "NOK", "DKK", "INR", "SGD", "AED"}


def _is_degenerate(fact: Fact) -> bool:
    """A fact whose whole value is a currency code or unit label.

    These are redundant by construction, and actively harmful: the renderer
    substitutes a citation marker with the fact's value, so a standalone "USD"
    fact turns "charged through [f_04]" into "charged through USD" and renders
    amounts as "990.00 USD USD". Dropped deterministically rather than trusted
    to the prompt.
    """
    value = fact.value_str.strip().upper()
    if fact.value_kind in {"decimal", "int"}:
        return False
    if value in _CURRENCY_CODES:
        return True
    return fact.name.strip().lower() in {"currency", "currency_code", "unit", "unit_type"}


def _strip_refs(narrative: str, refs: set[str]) -> str:
    """Remove citation markers for facts that were dropped, so the narrative
    still satisfies reference closure instead of failing validation for a
    problem we created by cleaning up."""
    for ref in refs:
        narrative = narrative.replace(f"[{ref}]", "")
    # tidy the double spaces / space-before-punctuation the removal leaves
    narrative = re.sub(r"[ \t]{2,}", " ", narrative)
    narrative = re.sub(r"\s+([,.;:)])", r"\1", narrative)
    return narrative.strip()


def _writer(state):
    """Resolve the trace writer the same way app/tools/authz.py does: prefer
    the one threaded through Deps, since a skill subgraph runs as its own
    Pregel run where get_stream_writer() cannot see the parent's queue."""
    try:
        threaded = getattr(get_runtime(Deps).context, "stream_writer", None)
    except Exception:
        threaded = None
    if threaded is not None:
        return threaded
    try:
        return get_stream_writer()
    except Exception:
        return None


def _model_for(skill_id: str, bound: list):
    model = named(skill_llm(), f"skill:{skill_id}")
    return model.bind_tools(bound) if bound else model


def build_skill_subgraph(manifest: SkillManifest):
    tools = tools_for(manifest.tool_grants)
    tool_node = ToolNode(tools, awrap_tool_call=authorize_and_trace) if tools else None

    async def agent_node(state: SkillState) -> dict:
        # The opening turns must be PERSISTED to state, not just used locally
        # for the first call. Otherwise the stored history starts at an
        # assistant turn containing function calls with no user turn before
        # it, which several providers reject outright.
        opening: list = []
        if not state["messages"]:
            opening = [
                SystemMessage(content=render_skill_system_prompt(
                    manifest, state.get("business_date"), state.get("prior_findings"),
                )),
                HumanMessage(content=state["task"]),
            ]
        messages = list(state["messages"]) or list(opening)

        out_of_budget = state["tool_steps"] >= manifest.max_tool_steps
        nudge = (
            [HumanMessage(
                content=f"You have used all {manifest.max_tool_steps} tool-calling steps. "
                "Stop calling tools and give your best answer from what you already have."
            )]
            if out_of_budget
            else []
        )

        # Models are resolved per call, not at build time: compiling a skill
        # must not require a configured model plane (manifest validation and
        # the boot self-test run without one). The underlying client is cached
        # in app/llm.py, so this only re-applies the tool binding.
        model = _model_for(manifest.id, tools)

        # Trace the model call itself. Without this the UI goes silent between
        # a tool returning and the next tool starting -- which is exactly when
        # the model is deciding what to do next, often the slowest part of a
        # step. A silent gap is indistinguishable from a hang.
        writer = _writer(state)
        step = state["tool_steps"] + 1
        if writer:
            writer({"type": "model_call", "skill_id": manifest.id, "step": step,
                    "phase": "deciding next step" if state["messages"] else "planning approach"})
        started = time.monotonic()
        playbook = prompt_store.skill(manifest)
        try:
            with prompt_store.linked(playbook):
                response = await model.ainvoke(messages + nudge)
        except Exception as exc:  # noqa: BLE001
            if writer:
                writer({"type": "model_result", "skill_id": manifest.id, "step": step,
                        "ok": False, "latency_ms": round((time.monotonic() - started) * 1000),
                        # Unwrapped: the gateway buries the provider's real
                        # complaint inside "All N fallback attempts failed".
                        "error": describe_model_error(exc)[:300]})
            raise
        latency_ms = round((time.monotonic() - started) * 1000)
        if writer:
            writer({"type": "model_result", "skill_id": manifest.id, "step": step, "ok": True,
                    "latency_ms": latency_ms,
                    "tool_calls": [tc["name"] for tc in (getattr(response, "tool_calls", None) or [])]})
        emit_model_usage(writer, response, role="skill", skill_id=manifest.id,
                         model_hint=get_settings().endpoint_for("skill"), latency_ms=latency_ms)
        if out_of_budget and isinstance(response, AIMessage) and response.tool_calls:
            response = AIMessage(content=response.content or "Ran out of tool-calling budget.")
        return {"messages": opening + nudge + [response]}

    async def tools_node(state: SkillState) -> dict:
        assert tool_node is not None
        result = await tool_node.ainvoke(state)
        return {"messages": result["messages"], "tool_steps": state["tool_steps"] + 1}

    def route(state: SkillState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and tool_node is not None:
            return "tools"
        return "extract"

    async def extract_node(state: SkillState) -> dict:
        transcript, real_ids = _transcript(state["messages"])
        # Named after the skill, not the planner, even though it runs on the
        # planner model: the ledger books extraction against the skill too.
        extractor = named(planner_llm(), f"skill:{manifest.id}").with_structured_output(
            SkillResult, include_raw=True
        )
        # Resolved per call, not at compile time, so a Langfuse edit reaches
        # already-built skill subgraphs.
        extraction = prompt_store.pipeline(prompt_store.EXTRACTION)
        system_text = extraction.text
        with prompt_store.linked(extraction):
            raw = await extractor.ainvoke(
                [SystemMessage(content=system_text), HumanMessage(content=transcript)]
            )
        # include_raw=True keeps the underlying AIMessage, which is the only
        # place the token counts for a structured-output call survive.
        emit_model_usage(_writer(state), raw.get("raw"), role="extract",
                         skill_id=manifest.id, model_hint=get_settings().endpoint_for("planner"))
        parsed: SkillResult | None = raw["parsed"]

        if parsed is None:
            result = SkillResult(
                skill_id=manifest.id,
                status="error",
                narrative=f"Could not extract a structured result: {raw.get('parsing_error')}",
                caveats=list(manifest.caveats),
            )
        else:
            kept: list[Fact] = []
            dropped = 0
            degenerate: set[str] = set()
            for f in parsed.facts:
                if _is_degenerate(f):
                    degenerate.add(f.ref)
                    continue
                # Deterministic groundedness gate: a fact citing a tool call
                # that never happened is discarded, not trusted.
                if f.provenance.tool_call_id in real_ids or f.provenance.source == "derived":
                    kept.append(f)
                else:
                    dropped += 1
            caveats = list(dict.fromkeys(parsed.caveats + manifest.caveats))
            status = parsed.status
            if dropped:
                status = "partial" if status == "ok" else status
                caveats.append(f"{dropped} claimed fact(s) cited no real tool call and were dropped")
            result = SkillResult(
                skill_id=manifest.id,
                status=status,
                narrative=_strip_refs(parsed.narrative, degenerate),
                facts=kept,
                caveats=caveats,
            )

        trace = [
            ToolTrace(tool_call_id=tc["id"], tool=tc["name"], args=tc["args"])
            for m in state["messages"]
            if isinstance(m, AIMessage) and m.tool_calls
            for tc in m.tool_calls
        ]
        return {
            "execution": SkillExecutionResult(
                skill_id=manifest.id,
                result=result,
                tool_trace=trace,
                steps_used=state["tool_steps"],
                finished_at=datetime.now(timezone.utc),
            )
        }

    g = StateGraph(SkillState, context_schema=Deps)
    g.add_node("agent", agent_node)
    g.add_node("extract", extract_node)
    g.add_edge(START, "agent")
    if tool_node is not None:
        g.add_node("tools", tools_node)
        g.add_edge("tools", "agent")
        g.add_conditional_edges("agent", route, {"tools": "tools", "extract": "extract"})
    else:
        g.add_edge("agent", "extract")
    g.add_edge("extract", END)
    return g.compile()
