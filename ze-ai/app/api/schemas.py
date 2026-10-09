"""Response models for the HTTP API, so the OpenAPI document says what every
endpoint returns."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

TAG = "ze-ai"


class Problem(BaseModel):
    detail: str = Field(description="Human-readable reason")


class ModelPlane(BaseModel):
    gateway: str = Field(description="MLflow gateway route every model call goes through")
    endpoints: dict[str, str] = Field(description="MLflow endpoint each role calls (planner, skill, prose, judge)")
    rate_limit_rps: float | None = Field(default=None, description="Model calls per second allowed, null if "
                                                                  "unlimited")


class Health(BaseModel):
    ok: bool = Field(description="Always true when the server answers")
    skills: int = Field(description="Skills loaded")
    agents: int = Field(description="Agents loaded")
    model_plane: ModelPlane = Field(description="Which MLflow endpoints answer questions")


class AgentOut(BaseModel):
    id: str = Field(description="Agent id, the `agent_id` of askQuestion")
    display_name: str = Field(description="Name shown in the UI")
    summary: str = Field(description="What the agent is for")
    skills: list[str] = Field(description="Skill ids the agent can use")


class SkillOut(BaseModel):
    id: str = Field(description="Skill id")
    kind: str = Field(description="domain (answers from data) or output (formats the answer)")
    entities: list[str] = Field(description="Domain entities the skill covers (domain/vocabulary.yaml)")
    operations: list[str] = Field(description="Kinds of question it answers, e.g. lookup, rank, explain")
    summary: str = Field(description="What the skill does")
    answers: list[str] = Field(description="Example questions it answers")
    provides: list[str] = Field(description="Slots it produces for later skills")
    cannot: list[str] = Field(description="What it explicitly does not do")
    tool_grants: list[str] = Field(description="Tools the skill's model can call")
    cost_class: str = Field(description="low, medium or high")


class ToolOut(BaseModel):
    name: str = Field(description="Tool name, as granted by skills")
    group: str = Field(description="Tool group")
    service: str = Field(description="The service whose spec the tool is generated from")
    operation_id: str = Field(description="The `<service>.<operationId>` tools.yaml maps it to")
    access: str = Field(description="read or write")
    cost_class: str = Field(description="low, medium or high")


class CatalogSummary(BaseModel):
    operations_in_specs: int = Field(description="Operations across every service spec")
    configured: int = Field(description="Operations configured in tools.yaml (callable by skills)")
    available_read: int = Field(description="GET operations not configured: candidates for new tools")
    write_listed_only: int = Field(description="Write operations: never callable by a skill")
    granted_to_skills: int = Field(description="Configured tools some skill grants")
    matching_filters: int = Field(description="Rows returned after filters")


class CatalogTool(BaseModel):
    name: str = Field(description="Tool name if configured, else the operationId")
    kind: str = Field(description="configured (in tools.yaml) or available (in the spec only)")
    access: str = Field(description="read or write")
    callable: bool = Field(description="Whether skills can call it: configured reads only")
    service: str = Field(description="Service the operation belongs to")
    operation_id: str = Field(description="The operationId in that service's spec")
    method: str = Field(description="HTTP method")
    path: str = Field(description="HTTP path")
    tags: list[str] = Field(description="Spec tags")
    granted_by: list[str] = Field(description="Skills that grant it (configured reads only)")
    description: str = Field(description="Model-facing description if configured, else the spec's")
    parameters: dict[str, Any] | None = Field(default=None, description="Argument JSON schema (omitted with schemas=false)")


class ToolCatalog(BaseModel):
    live: bool = Field(description="True: the catalogue of the running tool server. False: built from the spec "
                                   "snapshot for inspection")
    summary: CatalogSummary = Field(description="Counts")
    tools: list[CatalogTool] = Field(description="Matching operations")


class Costs(BaseModel):
    rollup: dict[str, Any] = Field(description="`totals` (runs, cost, tokens) and `by_unit` (per skill or node: runs, "
                                               "cost, steps) across every recorded run")
    recent: list[dict[str, Any]] = Field(description="The most recent runs, newest first, one ledger row each")
    tracing: dict[str, Any] = Field(description="`langfuse`: whether tracing is on; `host`: where traces go")


class Checkpoint(BaseModel):
    checkpoint_id: str | None = Field(description="LangGraph checkpoint id")
    next: list[str] = Field(description="Nodes still to run; non-empty means the run did not finish")
    has_answer: bool = Field(description="Whether an answer exists at this checkpoint")


class Run(BaseModel):
    thread_id: str = Field(description="The run's thread id")
    checkpoints: list[Checkpoint] = Field(description="Checkpoints, newest first")


SSE_EVENTS = """Server-sent events, one JSON envelope per event: `{type, seq, ts, data}`.

| type | data |
|---|---|
| `run_start` | request_id, thread_id, question, agent_id |
| `resolved` | what was recognised in the question (ids, dates, time window) |
| `plan_ready` | steps (skills to run), output_skill_id, unanswerable, unanswerable_reason |
| `node_start` / `node_end` | a pipeline node starting or ending |
| `model_call` / `model_result` / `model_usage` | one model call, its result, its tokens |
| `tool_call` / `tool_result` | one tool call by a skill (service, tool, args) and its result |
| `skill_done` | skill_id, status, narrative, facts, caveats, steps_used |
| `join_conflict` | note: two skills disagree |
| `validate_result` | ok, problems, retry_count |
| `answer_ready` | blocks (the rendered answer), skills_used, facts (cited values), degraded |
| `run_cost` | tokens, model calls, tool calls and cost for the run |
| `done` | the stream is complete |
"""

SSE_RESPONSE = {
    "description": "An event stream (text/event-stream) that ends with `done`. The thread id is in the X-Thread-Id header.",
    "content": {"text/event-stream": {"schema": {"type": "string"}}},
    "headers": {"X-Thread-Id": {"description": "Thread id of the run, for getRun and follow-up questions",
                                "schema": {"type": "string"}}},
}
