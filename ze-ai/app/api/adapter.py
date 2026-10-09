"""The ONLY file coupled to LangGraph's stream shapes. Everything downstream
speaks UIEvent. Built on the STABLE astream(stream_mode=[...]) API rather than
the beta v3 stream_events protocol.

Our own nodes and the authz tool wrapper emit rich events on the "custom"
channel; "updates" is used only to surface a few specific state transitions
worth their own named event."""

from __future__ import annotations

from collections.abc import AsyncIterator

from langgraph.types import Command

from app.api.events import UIEvent
from app.graph.state import PipelineState
from app.obs.ledger import RunLedger
from app.obs.tracing import trace_scope


async def stream_ui_events(
    graph, run_input: PipelineState | Command, *, context, config, ledger_store=None
) -> AsyncIterator[UIEvent]:
    """Shared by POST /ask (a PipelineState) and POST /ask/{id}/resume (a
    Command) -- one emission path for both halves of the two-phase contract.

    Every event also feeds the run's cost ledger on its way out. Metering here
    rather than inside the graph means there is exactly one place that knows
    about a run's totals, and the ledger stays framework-agnostic.
    """
    seq = 0
    ledger = RunLedger()

    def emit(event_type: str, data: dict) -> UIEvent:
        nonlocal seq
        seq += 1
        ledger.observe(event_type, data)
        return UIEvent(type=event_type, seq=seq, data=data)

    try:
        with trace_scope(config):
            async for event in _run(graph, run_input, config=config, context=context, emit=emit):
                yield event
        yield emit("run_cost", ledger.summary())
        yield emit("done", {})
    finally:
        # Also runs when the client disconnects mid-stream. LangGraph is
        # durable, so the work already done was real and its cost was really
        # incurred -- dropping the ledger there would quietly under-report.
        if ledger_store is not None:
            try:
                await ledger_store.save(ledger)
            except Exception:  # noqa: BLE001 -- never fail a run over telemetry
                pass


async def _run(graph, run_input, *, config, context, emit) -> AsyncIterator[UIEvent]:
    if isinstance(run_input, dict):
        yield emit("run_start", {
            "request_id": run_input["request_id"],
            "thread_id": run_input["thread_id"],
            "question": run_input["question"],
            "agent_id": run_input["agent"].id,
        })
    else:
        yield emit("run_resumed", {})

    async for mode, chunk in graph.astream(
        run_input, context=context, config=config, stream_mode=["custom", "updates"]
    ):
        if mode == "custom":
            event_type = chunk.get("type", "custom")
            yield emit(event_type, {k: v for k, v in chunk.items() if k != "type"})
            continue

        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue

            if node == "resolve" and update.get("resolved"):
                yield emit("resolved", dict(update["resolved"]))

            if node == "plan" and update.get("plan") is not None:
                p = update["plan"]
                yield emit("plan_ready", {
                    "steps": [dict(s) for s in p["steps"]],
                    "output_skill_id": p["output_skill_id"],
                    "unanswerable": p["unanswerable"],
                    "unanswerable_reason": p.get("unanswerable_reason"),
                })

            if node == "run_skill" and update.get("skill_results"):
                for ex in update["skill_results"]:
                    yield emit("skill_done", {
                        "skill_id": ex.skill_id,
                        "status": ex.result.status,
                        "narrative": ex.result.narrative,
                        "facts": [f.model_dump(mode="json") for f in ex.result.facts],
                        "caveats": ex.result.caveats,
                        "steps_used": ex.steps_used,
                    })

            if node == "join" and update.get("validation_notes"):
                for note in update["validation_notes"]:
                    yield emit("join_conflict", {"note": note})

            if node == "validate":
                problems = update.get("last_validation_problems") or []
                yield emit("validate_result", {
                    "ok": not problems,
                    "problems": problems,
                    "retry_count": update.get("retry_count", 0),
                })

            if node == "respond" and update.get("answer") is not None:
                a = update["answer"]
                yield emit("answer_ready", {
                    "blocks": [b.model_dump(mode="json") for b in a.blocks],
                    "skills_used": a.skills_used,
                    "facts": [f.model_dump(mode="json") for f in a.facts],
                    "caveats": a.caveats,
                    "degraded": a.degraded,
                })

    # `done` is emitted by the caller, AFTER run_cost -- clients stop reading at
    # `done`, so anything sent after it is never seen.
