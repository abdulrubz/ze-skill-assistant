"""A tool that raises must not kill the run.

This was a real failure: the model wrote `s.account_id` against a table with no
such column, pymysql raised OperationalError, authz re-raised it, and the whole
SSE stream died 6 seconds in with no answer and no explanation. The user saw a
run that simply stopped.

Tool failures are ordinary -- a wrong column, a 500 from Kill Bill, a timeout.
Each one must come back to the model as a readable error so it can correct
itself, which is exactly what the tool descriptions already promise. The skill's
max_tool_steps bounds the retrying.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from app.deps import Caller
from app.tools.authz import authorize_and_trace
from app.tools.registry import declare_tools


@dataclass
class _Ctx:
    caller: Caller


@dataclass
class _Runtime:
    context: _Ctx


@dataclass
class _Request:
    tool_call: dict[str, Any]
    runtime: _Runtime
    state: dict[str, Any]


def _request(tool: str = "list_orders") -> _Request:
    declare_tools()
    return _Request(
        tool_call={"name": tool, "id": "call_1", "args": {"sql": "SELECT 1", "purpose": "x"}},
        runtime=_Runtime(_Ctx(Caller(agent_id="general", tool_deny=()))),
        state={"skill_id": "subscription_lifecycle"},
    )


@pytest.mark.asyncio
async def test_a_raising_tool_returns_an_error_message_instead_of_propagating():
    async def boom(_request):
        raise RuntimeError("(1054, \"Unknown column 's.account_id' in 'SELECT'\")")

    result = await authorize_and_trace(_request(), boom)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "call_1"
    # The model needs the actual cause to fix it, not a generic failure.
    assert "Unknown column" in result.content
    assert "RuntimeError" in result.content


@pytest.mark.asyncio
async def test_the_error_tells_the_model_what_to_do_next():
    async def boom(_request):
        raise ValueError("bad")

    result = await authorize_and_trace(_request(), boom)
    assert "corrected" in result.content or "fix" in result.content


@pytest.mark.asyncio
async def test_cancellation_still_propagates():
    """Client disconnect must cancel the run. CancelledError is a BaseException,
    so the broad `except Exception` must not swallow it -- if it ever did, a
    disconnected browser would leave the graph running."""
    async def cancelled(_request):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await authorize_and_trace(_request(), cancelled)


@pytest.mark.asyncio
async def test_a_successful_tool_is_unaffected():
    async def fine(_request):
        return ToolMessage(content="ok", tool_call_id="call_1")

    result = await authorize_and_trace(_request(), fine)
    assert result.content == "ok"
    assert result.status != "error"
