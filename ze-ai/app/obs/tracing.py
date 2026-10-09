"""Optional Langfuse tracing.

Strictly additive to the cost ledger, which never depends on it. Langfuse's
handler is CALLBACK-driven, so it sees LangChain/LangGraph internals -- every
model call, tool call and node, with the full subgraph tree -- but it does NOT
see anything published through `get_stream_writer()`. Our semantic events
(validation verdicts, guard rejections, skill envelopes) travel on that channel,
so the two are complementary and neither substitutes for the other.

Disabled unless both keys are set: no import, no client, no network.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from functools import lru_cache
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _client() -> Any | None:
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse
    except ModuleNotFoundError:
        logger.warning("LANGFUSE_* keys are set but the langfuse package is not installed; continuing untraced")
        return None

    try:
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            **({"environment": settings.langfuse_environment} if settings.langfuse_environment else {}),
        )
        logger.info("langfuse tracing enabled -> %s", settings.langfuse_host)
        return client
    except Exception:  # noqa: BLE001 -- tracing must never block startup
        logger.warning("langfuse configured but failed to start; continuing untraced",
                       exc_info=True)
        return None


def tracing_enabled() -> bool:
    return _client() is not None


_IN_TOOL_ERROR: contextvars.ContextVar[bool] = contextvars.ContextVar("in_tool_error", default=False)


def _warning_on_tool_error_handler():
    """A CallbackHandler that records a failed TOOL call as WARNING, not ERROR.

    A failed tool call is not a failed run: the skill loop reads the error and
    tries again (the model guesses a column name, MariaDB says no, it corrects the
    query). Langfuse counts ERROR-level observations and paints them red, so every
    recovered call reads as a failure. Only tool errors are downgraded -- a chain,
    model or node error still comes through as ERROR, and the message is kept.
    """
    from langfuse.langchain import CallbackHandler

    class ToolErrorsAsWarnings(CallbackHandler):
        def on_tool_error(self, error, **kwargs):
            token = _IN_TOOL_ERROR.set(True)
            try:
                return super().on_tool_error(error, **kwargs)
            finally:
                _IN_TOOL_ERROR.reset(token)

        def _get_error_level_and_status_message(self, error):
            level, message = super()._get_error_level_and_status_message(error)
            if level == "ERROR" and _IN_TOOL_ERROR.get():
                level = "WARNING"
            return level, message

    return ToolErrorsAsWarnings()


def callbacks_for(*, thread_id: str, agent_id: str, question: str,
                  tool_errors_as_warnings: bool = False) -> dict:
    """Config fragment to merge into a graph invocation.

    Returns {} when tracing is off, so callers can always splat it in without
    branching. In langfuse v4 the handler carries no credentials of its own --
    it reads the globally-initialised client -- so `_client()` must have run.
    """
    if _client() is None:
        return {}
    try:
        from langfuse.langchain import CallbackHandler

        return {
            "callbacks": [_warning_on_tool_error_handler() if tool_errors_as_warnings else CallbackHandler()],
            # Becomes the trace name, so cost can be grouped per agent; without
            # it every trace is called "LangGraph".
            "run_name": f"ask:{agent_id}",
            "metadata": {
                # langfuse_* keys are lifted onto the trace itself, which is what
                # makes a run findable by thread id later.
                "langfuse_session_id": thread_id,
                "langfuse_tags": [f"agent:{agent_id}"],
                "question": question[:500],
            },
        }
    except Exception:  # noqa: BLE001
        logger.warning("could not build langfuse callback for this run", exc_info=True)
        return {}


def trace_scope(config: dict):
    """Stamp the run's agent, session and trace name onto EVERY span in it.

    The callback handler sets these on the root span only, and Langfuse v4
    aggregates per observation: cost lives on the generations, which sat
    under traceName/tags/sessionId = null, so "cost by agent" came out as one
    unlabelled bucket. propagate_attributes rides on contextvars, and LangGraph
    copies the context into every node task it spawns, so wrapping the stream
    loop reaches all of them.

    Reads what callbacks_for already put in the config, so the two cannot
    disagree. A no-op context when tracing is off.
    """
    meta = config.get("metadata") or {}
    if "callbacks" not in config or _client() is None:
        return contextlib.nullcontext()
    try:
        from langfuse import propagate_attributes

        tags = meta.get("langfuse_tags") or []
        agent = next((t.split(":", 1)[1] for t in tags if t.startswith("agent:")), None)
        return propagate_attributes(
            session_id=meta.get("langfuse_session_id"),
            tags=tags,
            trace_name=config.get("run_name"),
            metadata={"agent_id": agent} if agent else None,
        )
    except Exception:  # noqa: BLE001
        logger.warning("could not propagate langfuse trace attributes", exc_info=True)
        return contextlib.nullcontext()


def flush() -> None:
    """Langfuse batches in the background; call on shutdown so a short-lived
    process does not drop its last traces."""
    client = _client()
    if client is not None:
        try:
            client.flush()
        except Exception:  # noqa: BLE001
            pass
