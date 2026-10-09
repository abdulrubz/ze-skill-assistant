"""Specs + tools.yaml -> the read tools in the registry.

    specs/<service>.openapi.json --FastMCP.from_openapi--> one raw tool per operation, per service
    domain/tools.yaml            --allow-list + Tool.from_tool--> the configured reads, renamed,
                                   documented and projected
                                 --> registry.install_tool --> skills

Only the READS tools.yaml configures become tools; nothing else in any spec is
callable. Each service gets its own HTTP client (base URL SERVICE_<NAME>_URL)
whose request hook adds the end user's token (app/tools/auth.py).

The generated tools are called IN-PROCESS (`Tool.run`), not through an MCP
client session. That is deliberate: an MCP session's server tasks start when
the session opens and run in that context, so a per-request context variable
-- the user's token -- would never reach the HTTP call. Called directly, the
tool runs in the request's own context. The FastMCP server object is still
built (it is the catalogue, and can be served as a real MCP server later).
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import get_settings, service_env_var, service_url
from app.tools.auth import attach_user_token
from app.tools.openapi.config import ToolConfig, ToolsFile, load_tools, model_description, project
from app.tools.openapi.drift import check, errors
from app.tools.openapi.generate import load_generated, spec_path, stale_services

logger = logging.getLogger(__name__)


class ToolDriftError(RuntimeError):
    pass


class ServiceNotConfigured(RuntimeError):
    pass


def _load_spec(service: str) -> dict:
    return json.loads(spec_path(service).read_text())


def _transformed(base, name: str, tool: ToolConfig, generated: dict):
    from fastmcp.tools import Tool
    from fastmcp.tools.tool_transform import ArgTransform, forward

    agent = generated["operations"][tool.operation]["agent"]
    exposed = {e["key"]: e for e in agent["inputs"]}
    transforms: dict[str, ArgTransform] = {}
    for param in base.parameters.get("properties", {}):
        if param in exposed:
            e = exposed[param]
            desc = e["schema"].get("description")
            transforms[param] = ArgTransform(name=e["name"], **({"description": desc} if desc else {}))
        elif param in agent["fixed"]:
            transforms[param] = ArgTransform(hide=True, default=agent["fixed"][param])
        else:
            # Everything else stays invisible: plumbing the spec marks hidden,
            # headers, and inputs an `inputs` hint leaves out.
            transforms[param] = ArgTransform(hide=True)

    output = tool.output

    async def shape(**kwargs: Any):
        result = await forward(**kwargs)
        data = result.structured_content
        if data is None:
            text = "".join(getattr(c, "text", "") for c in result.content)
            if not text.strip():
                return {"empty": True, "note": "the API returned no content (e.g. HTTP 204)"}
            try:
                data = json.loads(text)
            except ValueError:
                return text
        # FastMCP wraps non-object (array) results as {"result": [...]}.
        if isinstance(data, dict) and set(data) == {"result"}:
            data = data["result"]
        return project(data, output)

    return Tool.from_tool(base, name=name, description=model_description(generated, tool),
                          transform_args=transforms, transform_fn=shape)


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    kind: str        # "configured" (in tools.yaml) | "available" (in a spec only)
    access: str      # "read" | "write"
    callable: bool   # configured reads only
    service: str
    operation_id: str
    method: str
    path: str
    tags: tuple[str, ...]
    description: str
    parameters: dict


@dataclass
class Built:
    server: Any
    tools: dict[str, Any] = field(default_factory=dict)       # name -> fastmcp Tool (configured reads)
    catalog: list[CatalogEntry] = field(default_factory=list)


async def build(generated: dict, tools: ToolsFile, clients: dict[str, httpx.AsyncClient]) -> Built:
    """The configured reads, and a catalogue of every operation in every spec:
    the menu to choose from when adding a tool to tools.yaml."""
    from fastmcp import FastMCP

    configured = {t.operation: name for name, t in tools.tools.items()}
    server = FastMCP("ze-ai-tools")
    built = Built(server)
    raw: dict[str, Any] = {}

    for opid, op in generated["operations"].items():
        service = op["service"]
        name = configured.get(opid)
        if name is not None and op["kind"] == "read":
            if service not in raw:
                raw[service] = FastMCP.from_openapi(_load_spec(service), client=clients[service],
                                                    name=f"{service}-generated", validate_output=False)
            base = await raw[service].get_tool(op["operation_id"])
            if base is None:
                raise ToolDriftError(f"operation '{opid}' was not generated")
            tool = _transformed(base, name, tools.tools[name], generated)
            server.add_tool(tool)
            built.tools[name] = tool
            entry = dict(name=name, kind="configured", callable=True, description=tool.description,
                         parameters=tool.parameters)
        else:
            entry = dict(name=name or opid, kind="configured" if name else "available", callable=False,
                         description=" ".join(x for x in (op["summary"], op["description"]) if x),
                         parameters=op["input_schema"])
        built.catalog.append(CatalogEntry(
            access=op["kind"], service=service, operation_id=op["operation_id"], method=op["method"],
            path=op["path"], tags=tuple(op["tags"]), **entry,
        ))

    missing = set(configured) - set(generated["operations"])
    if missing:
        raise ToolDriftError(f"configured operations not in any spec: {sorted(missing)}")
    return built


def _validate() -> tuple[dict, ToolsFile]:
    stale = stale_services()
    if stale:
        raise ToolDriftError(f"generated tool files are out of date for {stale}: run scripts/tool_spec.py generate")
    generated, tools = load_generated(), load_tools()
    problems = errors(check(generated, tools))
    if problems:
        raise ToolDriftError("tools.yaml no longer fits the specs:\n" + "\n".join(f"  {p}" for p in problems))
    return generated, tools


def _as_langchain(tool):
    """A LangChain tool that runs the generated tool in the caller's context and
    returns a string (what the extraction transcript and ledger expect).

    Failures raise rather than return text, so app/tools/authz.py sees them:
    it counts them and hands the model its standard "this call failed, correct
    it" message."""
    from langchain_core.tools import StructuredTool

    async def call(**kwargs):
        result = await tool.run(kwargs)
        if result.structured_content is not None:
            data = result.structured_content
            if isinstance(data, dict) and set(data) == {"result"}:
                data = data["result"]
            return data if isinstance(data, str) else json.dumps(data, default=str)
        return "".join(getattr(c, "text", "") for c in result.content)

    return StructuredTool(name=tool.name, description=tool.description, args_schema=tool.parameters,
                          coroutine=call)


def used_services(tools: ToolsFile | None = None) -> list[str]:
    return sorted({t.service for t in (tools or load_tools()).tools.values()})


def service_base_urls(tools: ToolsFile | None = None) -> dict[str, str]:
    """Base URL for every service tools.yaml uses; fails naming every missing one."""
    urls, missing = {}, []
    for service in used_services(tools):
        url = service_url(service)
        if url:
            urls[service] = url.rstrip("/")
        else:
            missing.append(service_env_var(service))
    if missing:
        raise ServiceNotConfigured(f"tools.yaml uses services with no base URL: set {', '.join(missing)}")
    return urls


async def install(stack: AsyncExitStack, *, base_urls: dict[str, str] | None = None,
                  transport: httpx.AsyncBaseTransport | None = None) -> list[str]:
    """Generate the configured reads and put them in the registry; return their names.

    Must run BEFORE the skill registry compiles subgraphs, which bind tool
    objects at build time. Fails fast on drift or a missing service URL: a
    half-working tool set is worse than a refusal to start.
    """
    from app.tools import registry

    registry.declare_tools()
    generated, tools = _validate()
    urls = base_urls if base_urls is not None else service_base_urls(tools)
    timeout = get_settings().service_timeout_s
    clients: dict[str, httpx.AsyncClient] = {}
    for service in sorted({op["service"] for op in generated["operations"].values()}):
        clients[service] = await stack.enter_async_context(httpx.AsyncClient(
            base_url=urls.get(service, "http://not-configured.invalid"),
            headers={"Accept": "application/json"},
            event_hooks={"request": [attach_user_token]},
            transport=transport, timeout=timeout,
        ))

    global _BUILT
    _BUILT = await build(generated, tools, clients)
    for name, tool in _BUILT.tools.items():
        registry.install_tool(name, _as_langchain(tool))
    names = sorted(_BUILT.tools)
    logger.info("tools: %d read tool(s) generated from %d service spec(s)", len(names), len(used_services(tools)))
    return names


_BUILT: Built | None = None
_INSPECTION: list[CatalogEntry] | None = None


async def catalog() -> tuple[list[CatalogEntry], bool]:
    """(entries, live). The running catalogue once installed; otherwise built
    on demand for inspection (listing needs no network)."""
    global _INSPECTION
    if _BUILT is not None:
        return _BUILT.catalog, True
    if _INSPECTION is None:
        generated = load_generated()
        async with AsyncExitStack() as stack:
            clients = {}
            for service in {op["service"] for op in generated["operations"].values()}:
                clients[service] = await stack.enter_async_context(
                    httpx.AsyncClient(base_url="http://inspection-only.invalid"))
            _INSPECTION = (await build(generated, load_tools(), clients)).catalog
    return _INSPECTION, False


async def install_from_settings(stack: AsyncExitStack) -> list[str]:
    """The one entry point the app and scripts use."""
    return await install(stack)
