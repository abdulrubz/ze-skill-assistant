"""Wires the pipeline:

  START -> admission -> resolve -> plan -> [Send fan-out] run_skill -> schedule -> join
                                                              ^____ next wave ______|
        -> validate -> [ok or retries exhausted: respond | else: retry] -> END

"Request"/"Response" from the reference architecture are START/END plus the
caller's input/output shape rather than nodes of their own.
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.deps import Caller, Deps
from app.graph.nodes.admission import admission
from app.graph.nodes.join import join
from app.graph.nodes.plan import plan
from app.graph.nodes.respond import respond
from app.graph.nodes.resolve import resolve
from app.graph.nodes.run_skill import RunSkillInput, run_skill
from app.graph.nodes.validate import validate
from app.graph.routing import (
    route_after_admission,
    route_after_plan,
    route_after_run_skill,
    route_after_validate,
)
from app.graph.state import PipelineState


def _traced(name: str, fn: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
    """Emits node_start/node_end trace events around every pipeline node, so
    the UI sees the whole pipeline -- not just the nodes that call tools. Kept
    as a wrapper so the node functions stay framework-agnostic."""

    async def wrapped(payload):
        try:
            writer = get_stream_writer()
        except Exception:
            writer = None
        if writer:
            writer({"type": "node_start", "node": name})
        started = time.monotonic()
        try:
            result = await fn(payload)
        except Exception as exc:  # noqa: BLE001
            if writer:
                writer({"type": "node_end", "node": name, "ok": False, "error": str(exc)})
            raise
        if writer:
            writer({
                "type": "node_end", "node": name, "ok": True,
                "latency_ms": round((time.monotonic() - started) * 1000),
            })
        return result

    wrapped.__name__ = f"traced_{name}"
    return wrapped


async def schedule(_state: PipelineState) -> dict:
    """A no-op whose only job is WHEN it runs: once, after every run_skill task
    in the wave has finished and its result is merged.

    The next-wave decision used to be a conditional edge straight off
    run_skill. LangGraph evaluates such an edge once PER TASK, against that
    task's own writes only -- so with two independent steps in flight, each
    task saw the other step as not done and dispatched it again. Every
    two-skill plan ran each skill twice (measured: 4 skill runs for 2 steps,
    validation clean, no retry). Deciding here, on the merged state, fixes it.
    """
    return {}


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    g = StateGraph(PipelineState, context_schema=Deps)

    g.add_node("admission", _traced("admission", admission))
    g.add_node("resolve", _traced("resolve", resolve))
    g.add_node("plan", _traced("plan", plan))
    g.add_node("run_skill", _traced("run_skill", run_skill), input_schema=RunSkillInput)
    g.add_node("schedule", schedule)
    g.add_node("join", _traced("join", join))
    g.add_node("validate", _traced("validate", validate))
    g.add_node("respond", _traced("respond", respond))

    g.add_edge(START, "admission")
    g.add_conditional_edges("admission", route_after_admission, ["resolve", "respond"])
    g.add_edge("resolve", "plan")
    g.add_conditional_edges("plan", route_after_plan, ["run_skill", "respond"])
    # Conditional, not a plain edge: a plan whose later step declared
    # `depends_on` runs in waves, and each completed wave re-enters the
    # fan-out with the next one. With no dependencies declared -- almost
    # every plan -- this fires once and goes straight to join.
    g.add_edge("run_skill", "schedule")
    g.add_conditional_edges("schedule", route_after_run_skill, ["run_skill", "join"])
    g.add_edge("join", "validate")
    # "run_skill" is a target here too: a validation failure re-runs ONLY the
    # steps that failed, straight back into the fan-out without a replan.
    g.add_conditional_edges("validate", route_after_validate, ["plan", "run_skill", "respond"])
    g.add_edge("respond", END)

    return g.compile(checkpointer=checkpointer)


def initial_state(*, request_id: str, thread_id: str, question: str, agent, caller: Caller) -> PipelineState:
    return PipelineState(
        request_id=request_id,
        thread_id=thread_id,
        question=question,
        agent=agent,
        caller=caller,
        admission=None,
        resolved=None,
        plan=None,
        skill_results=[],
        failed_step_indexes=[],
        validation_notes=[],
        last_validation_problems=[],
        retry_count=0,
        answer=None,
        degraded=False,
    )
