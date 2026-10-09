"""FastAPI app. Everything expensive -- tool clients, registries,
checkpointer, compiled graph -- is built ONCE in the lifespan and stored on
app.state, never per request (an async saver opened per request thrashes
connections and locks SQLite)."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app import prompt_store
from app.agents.manifest import load_all_agent_manifests
from app.config import get_settings
from app.deps import Caller, Deps
from app.graph.build import build_graph
from app.llm import planner_llm
from app.obs.ledger import LedgerStore
from app.obs.tracing import flush as flush_traces, tracing_enabled
from app.skills.registry import load_and_validate

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Every tool is generated from a service's OpenAPI spec and configured by
    # domain/tools.yaml. Installing has to happen before load_and_validate,
    # which binds tool objects into each skill's compiled subgraph.
    tool_stack = AsyncExitStack()
    from app.tools.openapi.server import install_from_settings
    await install_from_settings(tool_stack)

    # Fails fast at boot if any manifest is wired wrong, rather than letting a
    # broken skill surface as a mysterious runtime error.
    skills = load_and_validate()
    agents = load_all_agent_manifests()

    # Skills resolve their model lazily (so manifest validation needs no API
    # key), which would otherwise defer a misconfigured model plane to the
    # first user question. Check it here instead so the failure is at startup
    # with a clear message.
    planner_llm()

    checkpoint_path = Path(settings.checkpoint_sqlite_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    ledger_store = LedgerStore(settings.obs_sqlite_path)
    if tracing_enabled():
        logger.info("langfuse tracing is on")
    # First fetch of each prompt blocks on the network; do them all now rather
    # than on the first request that needs each one.
    prompt_sources = prompt_store.warm()
    if prompt_sources:
        served = sum(1 for s in prompt_sources.values() if s.startswith("langfuse"))
        logger.info("prompts: %d/%d served from Langfuse (label %r); rest from repo files",
                    served, len(prompt_sources), settings.langfuse_prompt_label)

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        app.state.settings = settings
        app.state.skills = skills
        app.state.agents = agents
        app.state.ledger_store = ledger_store
        app.state.graph = build_graph(checkpointer=checkpointer)

        def make_deps(agent_id: str, *, user_token: str | None) -> Deps:
            agent = agents[agent_id]
            return Deps(
                settings=settings,
                caller=Caller(agent_id=agent_id, tool_deny=tuple(agent.scope.tool_deny)),
                user_token=user_token,
            )

        app.state.make_deps = make_deps
        yield
        flush_traces()
        await tool_stack.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="ze-ai", description="GenAI service: agents -> skills -> tools generated from the "
                  "services' OpenAPI specs", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    from app.api.routes_ask import router

    app.include_router(router)
    return app


app = create_app()
