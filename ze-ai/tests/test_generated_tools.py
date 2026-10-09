"""Specs -> generated tools -> a real call, across more than one service.

Uses the committed spec snapshot (specs/) and an httpx MockTransport standing in
for the services, so it needs no network. What it holds:

  - operations and schemas are namespaced per service, so two teams' specs can
    never collide, and every $ref still resolves;
  - each tool calls ITS service's base URL;
  - every call carries the end user's own token, and nothing leaves without one;
  - the token never appears in a trace event or in graph state.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack

import httpx
import pytest
from langgraph.prebuilt.tool_node import ToolCallRequest

from app.config import get_settings
from app.deps import Caller, Deps
from app.tools.auth import MissingUserToken, acting_as, bearer_from_header
from app.tools.authz import authorize_and_trace
from app.tools.openapi.config import load_tools, resolve
from app.tools.openapi.generate import load_generated, stale_services
from app.tools.openapi.server import install
from app.tools.registry import get_tool

URLS = {"customers": "http://customers.test", "orders": "http://orders.test"}
CUSTOMER = {"customer_id": "CUST-1002", "name": "Globex Ltd", "segment": "smb", "region": "EU", "currency": "EUR",
            "email": "billing@globex.example", "signup_date": "2024-12-27", "credit_limit": 3000, "active": True}


def test_the_committed_generated_files_match_the_specs():
    assert stale_services() == [], "run scripts/tool_spec.py generate and commit the result"


def test_operations_and_schemas_are_namespaced_by_service():
    g = load_generated()
    assert {"customers.getCustomer", "orders.getOrder"} <= set(g["operations"])
    assert all("." in name for name in g["schemas"])
    # Both specs define HTTPValidationError; namespacing keeps both.
    assert {"customers.HTTPValidationError", "orders.HTTPValidationError"} <= set(g["schemas"])
    out = g["operations"]["orders.getOrder"]["output_schema"]
    assert out["$ref"] == "#/schemas/orders.Order"
    assert resolve(g, out)["properties"]["lines"]


def test_every_configured_tool_names_a_service_with_a_spec():
    services = set(load_generated()["services"])
    for name, tool in load_tools().tools.items():
        assert tool.service in services, f"{name}: no spec for service '{tool.service}'"


@pytest.fixture
async def installed():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "customers.test" and request.url.path == "/customers/CUST-1002":
            return httpx.Response(200, json=CUSTOMER)
        if request.url.host == "orders.test" and request.url.path == "/reports/top-customers":
            return httpx.Response(200, json=[{"rank": 1, "customer_id": "CUST-1002", "currency": "EUR",
                                              "total": 3853.0, "order_count": 3}])
        return httpx.Response(404, json={"detail": "not found"})

    async with AsyncExitStack() as stack:
        await install(stack, base_urls=URLS, transport=httpx.MockTransport(handler))
        yield seen


def _request(tool: str, args: dict, token: str | None, events: list):
    deps = Deps(settings=get_settings(), caller=Caller(agent_id="general"), user_token=token,
                stream_writer=events.append)

    class Runtime:
        context = deps

    call = {"name": tool, "args": args, "id": "call_1", "type": "tool_call"}
    return ToolCallRequest(tool_call=call, tool=get_tool(tool), state={"skill_id": "s"}, runtime=Runtime())


async def _execute(request):
    from langchain_core.messages import ToolMessage

    out = await request.tool.ainvoke(request.tool_call["args"])
    return ToolMessage(content=out, tool_call_id=request.tool_call["id"])


async def test_each_tool_calls_its_own_service_with_the_users_token(installed):
    events: list = []
    msg = await authorize_and_trace(_request("get_customer", {"customer_id": "CUST-1002"}, "tok-alice", events),
                                    _execute)
    assert json.loads(msg.content)["name"] == "Globex Ltd"
    await authorize_and_trace(_request("rank_customers_by_spend", {"currency": "EUR"}, "tok-alice", events), _execute)

    hosts = [r.url.host for r in installed]
    assert hosts == ["customers.test", "orders.test"]
    assert all(r.headers["authorization"] == "Bearer tok-alice" for r in installed)
    # The token is plumbing: it is in no trace event the client or Langfuse sees.
    assert "tok-alice" not in json.dumps(events, default=str)
    assert any(e.get("service") == "orders" for e in events if e["type"] == "tool_call")


async def test_two_users_never_see_each_others_token(installed):
    import asyncio

    await asyncio.gather(
        authorize_and_trace(_request("get_customer", {"customer_id": "CUST-1002"}, "tok-alice", []), _execute),
        authorize_and_trace(_request("get_customer", {"customer_id": "CUST-1002"}, "tok-bob", []), _execute),
    )
    assert sorted(r.headers["authorization"] for r in installed) == ["Bearer tok-alice", "Bearer tok-bob"]


async def test_no_call_leaves_without_a_user_token(installed):
    msg = await authorize_and_trace(_request("get_customer", {"customer_id": "CUST-1002"}, None, []), _execute)
    assert msg.status == "error"
    assert "MissingUserToken" in json.loads(msg.content)["error"]
    assert installed == [], "the request must not be sent at all"


async def test_a_service_error_goes_back_to_the_model_as_json(installed):
    msg = await authorize_and_trace(_request("get_customer", {"customer_id": "CUST-9999"}, "t", []), _execute)
    body = json.loads(msg.content)
    assert msg.status == "error" and "404" in body["error"] and body["instruction"]


def test_the_token_is_kept_out_of_deps_repr():
    deps = Deps(settings=get_settings(), caller=Caller(agent_id="a"), user_token="secret-token")
    assert "secret-token" not in repr(deps)


def test_bearer_header_parsing():
    assert bearer_from_header("Bearer abc") == "abc"
    assert bearer_from_header("bearer  abc ") == "abc"
    assert bearer_from_header("Basic abc") is None
    assert bearer_from_header(None) is None


async def test_the_hook_refuses_without_a_token():
    from app.tools.auth import attach_user_token

    with acting_as(None), pytest.raises(MissingUserToken):
        await attach_user_token(httpx.Request("GET", "http://x.test/"))
