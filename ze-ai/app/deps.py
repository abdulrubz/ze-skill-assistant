"""What every graph node and tool call reaches through `runtime.context`. One
instance per graph invocation, passed via `context=` -- never global state, so
tests can swap in fakes.

There is no API client or database handle here: every call to a service goes
through a tool generated from its OpenAPI spec (app/tools/)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from app.config import Settings


@dataclass(frozen=True)
class Caller:
    """Who is asking and what they may do. Resolved once by Admission from the
    agent manifest, never mutated. This is the identity half of the
    authorization check in app/tools/authz.py; the other half is the per-skill
    tool grant."""

    agent_id: str
    tool_deny: tuple[str, ...] = field(default_factory=tuple)
    mode: Literal["agent", "free_form"] = "agent"


@dataclass
class Deps:
    settings: Settings
    caller: Caller
    # The end user's bearer token, forwarded to every service call
    # (app/tools/auth.py). Lives HERE, never in graph state: state is
    # checkpointed and traced, runtime context is neither. repr=False keeps it
    # out of any log line that prints a Deps.
    user_token: str | None = field(default=None, repr=False)
    # "Today" for relative dates, resolved once per run by the resolve node
    # (AS_OF_DATE when pinned, else the real date).
    business_date: str | None = None
    # Threaded explicitly rather than relying on get_stream_writer() inside a
    # skill subgraph: skill subgraphs are invoked with a manual `.ainvoke()`
    # from run_skill.py (needed for dynamic per-skill Send fan-out with
    # per-skill state schemas), which is a SEPARATE Pregel run. Verified
    # empirically that get_stream_writer() there does NOT see the parent
    # astream()'s queue, so run_skill captures the real writer (where it does
    # work) and threads it in here for app/tools/authz.py to use.
    stream_writer: Callable[[dict], None] | None = None
