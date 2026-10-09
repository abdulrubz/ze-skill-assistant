"""HTTP contract: POST /ask streams a run's progress and answer as server-sent
events. The caller's own bearer token is required and is forwarded to every
service the run calls (app/tools/auth.py)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.api.schemas import (
    SSE_EVENTS,
    SSE_RESPONSE,
    TAG,
    AgentOut,
    Costs,
    Health,
    Problem,
    Run,
    SkillOut,
    ToolCatalog,
    ToolOut,
)

from app.api.adapter import stream_ui_events
from app.api.sse import SSE_HEADERS, to_sse_bytes
from app.graph.build import initial_state
from app.obs.tracing import callbacks_for, tracing_enabled

router = APIRouter(tags=[TAG])

# auto_error=False so a missing token is our 401 with a clear message, and so
# Swagger UI shows an Authorize button for the bearer token.
_bearer = HTTPBearer(auto_error=False, description="The end user's token; forwarded to every service call")


ASK_DESCRIPTION = (
    "Runs the question through the pipeline (resolve, plan, skills, validate, respond) and streams its progress.\n\n"
    "**Returns:** " + SSE_EVENTS + "\n"
    "**Does not return:** JSON. The final answer is the `answer_ready` event."
)


class AskRequest(BaseModel):
    question: str = Field(description="The question, in plain language")
    agent_id: str = Field(default="general", description="Agent to answer as (see listAgents)")
    thread_id: str | None = Field(default=None, description="Continue an existing conversation; omit for a new one")


@router.get("/health", response_model=Health, operation_id="getHealth", summary="Liveness and model plane",
            response_description="The server is up, with what it loaded")
async def health(request: Request):
    """**Returns:** skill, agent and tool counts, and which MLflow endpoints answer.

    **Does not return:** the health of the services or the gateway (see scripts/doctor.py).
    """
    from app.llm import describe_model_plane

    app = request.app
    return {
        "ok": True,
        "skills": len(app.state.skills),
        "agents": len(app.state.agents),
        "model_plane": describe_model_plane(),
    }


@router.get("/agents", response_model=list[AgentOut], operation_id="listAgents", summary="Agents that can answer",
            response_description="Every agent")
async def list_agents(request: Request):
    """**Returns:** each agent's id (the `agent_id` of `askQuestion`), purpose and skills.

    **Does not return:** the agents' prompts.
    """
    return [
        {
            "id": a.id,
            "display_name": a.display_name,
            "summary": a.summary,
            "skills": a.skills,
        }
        for a in request.app.state.agents.values()
    ]


@router.get("/skills", response_model=list[SkillOut], operation_id="listSkills", summary="Skills the planner can run",
            response_description="Every skill manifest, summarised")
async def list_skills(request: Request):
    """**Returns:** each skill's scope (entities, operations, example questions, what it cannot do) and its tool grants.

    **Does not return:** skill prompts or playbooks.
    """
    return [
        {
            "id": s.manifest.id,
            "kind": s.manifest.kind,
            "entities": s.manifest.entities,
            "operations": s.manifest.operations,
            "summary": s.manifest.summary,
            "answers": s.manifest.answers,
            "provides": s.manifest.provides,
            "cannot": s.manifest.cannot,
            "tool_grants": s.manifest.tool_grants,
            "cost_class": s.manifest.cost_class,
        }
        for s in request.app.state.skills.values()
    ]


@router.get("/tools", response_model=list[ToolOut], operation_id="listTools", summary="Tools registered for skills",
            response_description="Every registered tool")
async def list_tools(request: Request):
    """**Returns:** each tool's name, group, service and operation.

    **Does not return:** tool schemas or descriptions (see `listToolCatalog` for spec-generated tools).
    """
    from app.tools.registry import all_specs

    return [
        {"name": s.name, "group": s.group, "service": s.service, "access": s.access,
         "operation_id": s.operation_id, "cost_class": s.cost_class}
        for s in all_specs().values()
    ]


@router.get("/tools/mcp", response_model=ToolCatalog, response_model_exclude_none=True,
            operation_id="listToolCatalog", summary="Every service operation and which are configured as tools",
            response_description="The spec-generated tool catalogue")
async def list_mcp_tools(
    request: Request,
    kind: str | None = Query(None, description="configured or available"),
    access: str | None = Query(None, description="read or write"),
    service: str | None = Query(None, description="Only this service's operations"),
    callable: bool | None = Query(None, description="Only callable (true) or non-callable (false) operations"),  # noqa: A002
    tag: str | None = Query(None, description="Spec tag"),
    q: str | None = Query(None, description="Substring of the name, path or description"),
    schemas: bool = Query(True, description="Include each tool's argument schema"),
):
    """Every operation in every service spec. Only the reads configured in
    domain/tools.yaml can be called (`callable: true`); every other operation is
    listed as `<service>.<operationId>` (`kind: available`) as the menu for
    adding one. `granted_by` shows which skills put a tool in front of the model.

    **Returns:** per operation: name, kind, access, callable, service, operationId, method, path, tags, granting
    skills, description and (unless `schemas=false`) its argument schema.

    **Does not return:** each tool's output projection; see tools.yaml.
    """
    from app.tools.openapi.server import catalog

    entries, live = await catalog()

    grants: dict[str, list[str]] = {}
    for sid, skill in request.app.state.skills.items():
        for t in skill.manifest.tool_grants:
            grants.setdefault(t, []).append(sid)

    def keep(e) -> bool:
        if kind and e.kind != kind:
            return False
        if access and e.access != access:
            return False
        if service and e.service != service:
            return False
        if callable is not None and e.callable != callable:
            return False
        if tag and tag.lower() not in (t.lower() for t in e.tags):
            return False
        if q and q.lower() not in f"{e.name} {e.path} {e.description}".lower():
            return False
        return True

    rows = [e for e in entries if keep(e)]
    return {
        # live: the catalogue of the running server
        # (installed at startup). False: built from the spec snapshot for inspection.
        "live": live,
        "summary": {
            "operations_in_specs": len(entries),
            "configured": sum(e.kind == "configured" for e in entries),
            "available_read": sum(e.kind == "available" and e.access == "read" for e in entries),
            "write_listed_only": sum(e.access == "write" for e in entries),
            "granted_to_skills": sum(e.name in grants for e in entries),
            "matching_filters": len(rows),
        },
        "tools": [
            {
                "name": e.name,
                "kind": e.kind,
                "access": e.access,
                "callable": e.callable,
                "service": e.service,
                "operation_id": e.operation_id,
                "method": e.method,
                "path": e.path,
                "tags": list(e.tags),
                "granted_by": grants.get(e.name, []) if e.kind == "configured" else [],
                "description": e.description,
                **({"parameters": e.parameters} if schemas else {}),
            }
            for e in rows
        ],
    }


@router.post("/ask", operation_id="askQuestion", summary="Ask a question; the answer streams as events",
             response_class=StreamingResponse,
             responses={200: SSE_RESPONSE, 401: {"model": Problem, "description": "No bearer token"},
                        404: {"model": Problem, "description": "Unknown agent_id"}},
             description=ASK_DESCRIPTION)
async def ask(req: AskRequest, request: Request,
              creds: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    app = request.app
    if creds is None or not creds.credentials:
        raise HTTPException(401, "a bearer token is required: it is forwarded to every service the answer needs",
                            headers={"WWW-Authenticate": "Bearer"})
    if req.agent_id not in app.state.agents:
        raise HTTPException(404, f"unknown agent '{req.agent_id}'")

    thread_id = req.thread_id or str(uuid.uuid4())
    deps = app.state.make_deps(req.agent_id, user_token=creds.credentials)
    state = initial_state(
        request_id=str(uuid.uuid4()),
        thread_id=thread_id,
        question=req.question,
        agent=app.state.agents[req.agent_id],
        caller=deps.caller,
    )
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": app.state.settings.recursion_limit,
        **callbacks_for(thread_id=thread_id, agent_id=req.agent_id, question=req.question),
    }
    events = stream_ui_events(
        app.state.graph, state, context=deps, config=config,
        ledger_store=app.state.ledger_store,
    )
    return StreamingResponse(
        to_sse_bytes(events),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, "X-Thread-Id": thread_id},
    )


@router.get("/costs", response_model=Costs, operation_id="getCosts", summary="Cost and quality across runs",
            response_description="Totals, per-unit rollup and recent runs")
async def costs(request: Request, limit: int = Query(50, ge=1, le=1000, description="Recent runs to include")):
    """Cost and quality across runs. `by_unit` is where the money actually goes
    -- a skill that always burns its full tool-step budget shows up here long
    before anyone notices the latency.

    **Returns:** cost totals, a per-unit rollup, the most recent runs and the tracing setup.

    **Does not return:** individual model calls; those are in Langfuse when tracing is on.
    """
    store = request.app.state.ledger_store
    return {
        "rollup": await store.rollup(),
        "recent": await store.recent(limit),
        "tracing": {
            "langfuse": tracing_enabled(),
            "host": request.app.state.settings.langfuse_host if tracing_enabled() else None,
        },
    }


@router.get("/runs/{thread_id}", response_model=Run, operation_id="getRun", summary="Checkpoints of one run",
            response_description="The run's checkpoints, newest first",
            responses={404: {"model": Problem, "description": "No run with this thread id"}})
async def get_run(request: Request,
                  thread_id: str = Path(description="Thread id from askQuestion's X-Thread-Id header")):
    """**Returns:** each checkpoint's id, the nodes still to run and whether an answer exists.

    **Does not return:** the answer or the events; those stream from `askQuestion`.
    """
    config = {"configurable": {"thread_id": thread_id}}
    history = []
    async for snap in request.app.state.graph.aget_state_history(config):
        history.append({
            "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
            "next": list(snap.next),
            "has_answer": snap.values.get("answer") is not None,
        })
    if not history:
        raise HTTPException(404, f"no run found for thread '{thread_id}'")
    return {"thread_id": thread_id, "checkpoints": history}
