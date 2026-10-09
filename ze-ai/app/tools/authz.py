"""The single chokepoint every tool call passes through: deterministic
authorization (never model-decided), the end user's token, and the trace
events the client's live view is built from.

Authorization here is DEFENSE IN DEPTH. The primary defense is that a skill
subgraph binds the model to ONLY its own tool_grants, so it cannot even name a
forbidden tool. This wrapper re-checks unconditionally and short-circuits
BEFORE execute(), because binding is advisory (a factory bug, a manifest typo)
while this runs on every call.

It is also where the caller's bearer token is put in scope (app/tools/auth.py)
for the duration of the call, so every service request carries it while
nothing the model, the trace or the checkpointer sees ever does.
"""

from __future__ import annotations

import json
import time
from typing import Awaitable, Callable

from langchain_core.messages import ToolMessage
from langgraph.config import get_stream_writer
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from app.tools.auth import acting_as
from app.tools.registry import get_spec

Execute = Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]


def _skill_id(state: object) -> str | None:
    if isinstance(state, dict):
        return state.get("skill_id")
    return getattr(state, "skill_id", None)


def _emit(request: ToolCallRequest, event: dict) -> None:
    # Prefer the writer threaded through Deps: this call may be running inside
    # a skill subgraph invoked via a manual .ainvoke(), where
    # get_stream_writer()'s contextvar does not see the parent astream()'s
    # queue. Fall back for direct/unit-test invocations.
    writer = getattr(getattr(request.runtime, "context", None), "stream_writer", None)
    if writer is None:
        try:
            writer = get_stream_writer()
        except Exception:
            return
    writer(event)


async def authorize_and_trace(request: ToolCallRequest, execute: Execute) -> ToolMessage | Command:
    name = request.tool_call["name"]
    call_id = request.tool_call.get("id")
    args = request.tool_call.get("args", {})
    skill = _skill_id(request.state)
    caller = request.runtime.context.caller

    try:
        spec = get_spec(name)
    except KeyError:
        return _deny(request, f"unknown tool '{name}'", skill, args)

    if name in caller.tool_deny:
        return _deny(request, f"agent '{caller.agent_id}' has '{name}' on its deny list", skill, args)

    _emit(request, {
        "type": "tool_call", "skill_id": skill, "tool": name, "tool_call_id": call_id,
        "group": spec.group, "service": spec.service, "access": spec.access, "args": args,
    })

    started = time.monotonic()
    try:
        with acting_as(getattr(request.runtime.context, "user_token", None)):
            result = await execute(request)
    except Exception as exc:  # noqa: BLE001 -- a failing tool must not kill the run
        # Hand the failure BACK to the model rather than raising. A tool that
        # blows up is normal: the model wrote a bad column name, the API 500'd,
        # a query timed out. Re-raising turned every one of those into a dead
        # SSE stream with no answer -- the user saw a run that simply stopped.
        #
        # Returned as an error ToolMessage, it becomes the self-correction loop
        # the tool descriptions already promise. The skill's max_tool_steps
        # bounds the retrying, so this cannot spin.
        #
        # asyncio.CancelledError is a BaseException and is deliberately NOT
        # caught here -- client disconnect must still cancel the run.
        _emit(request, {
            "type": "tool_result", "skill_id": skill, "tool": name, "tool_call_id": call_id,
            "ok": False, "latency_ms": round((time.monotonic() - started) * 1000), "error": str(exc),
        })
        # JSON, not prose: some gateway routes parse tool content as JSON, and
        # a plain-text error there broke the self-correction loop.
        return ToolMessage(
            content=json.dumps({
                "error": f"{type(exc).__name__}: {exc}",
                "instruction": "This call failed. Read the error, fix the cause and try a corrected "
                               "call. If it cannot be fixed, say so in your findings rather than "
                               "guessing at the answer.",
            }),
            tool_call_id=call_id,
            status="error",
        )

    latency = round((time.monotonic() - started) * 1000)
    rows, summary = _summarize(result)
    _emit(request, {
        "type": "tool_result", "skill_id": skill, "tool": name, "tool_call_id": call_id,
        "ok": True, "latency_ms": latency, "row_count": rows, "summary": summary,
    })
    return result


def _deny(request: ToolCallRequest, reason: str, skill: str | None, args: dict) -> ToolMessage:
    call_id = request.tool_call.get("id")
    name = request.tool_call["name"]
    _emit(request, {
        "type": "tool_call", "skill_id": skill, "tool": name,
        "tool_call_id": call_id, "access": "unknown", "args": args,
    })
    _emit(request, {
        "type": "tool_result", "skill_id": skill, "tool": name, "tool_call_id": call_id,
        "ok": False, "denied": True, "error": reason,
    })
    return ToolMessage(content=json.dumps({"error": f"DENIED: {reason}"}), tool_call_id=call_id, status="error")


def _summarize(result: ToolMessage | Command) -> tuple[int | None, str]:
    if isinstance(result, Command):
        return None, "state update"
    content = result.content
    text = content if isinstance(content, str) else str(content)
    rows = None
    if isinstance(content, str) and ('"row_count"' in content or '"count"' in content):
        try:
            parsed = json.loads(content)
            rows = parsed.get("row_count", parsed.get("count"))
        except Exception:
            pass
    return rows, text[:400]
