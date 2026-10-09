"""Editable prompts: Langfuse is the live source, the repo file is the fallback.

Every prompt a person might want to tune has a stable NAME here, and each name
has a file in the repo. With `LANGFUSE_PROMPTS=true` the text is read from
Langfuse Prompt Management at the configured label (default `production`), so
business users can edit a prompt in the Langfuse UI and have it take effect
within `langfuse_prompt_cache_ttl` seconds -- no deploy. With it off, or with
Langfuse unreachable, or with the prompt missing there, the repo file is used.
A Langfuse outage therefore degrades to "prompts as of the last pull", never
to a broken run.

What is editable is deliberately only the PROSE. The parts of a skill's system
prompt generated from its manifest -- `cannot`, standing caveats, the step
budget, the closing citation rule -- stay in code
(app/skills/factory.py), so no prompt edit can delete a safety rule.

Keeping git in step: scripts/prompts.py `pull` writes the live versions back
into these files, so review, tests and evals see what production runs; `push`
seeds Langfuse from the files. See that script's docstring for the workflow.

Every name lives in Langfuse under a folder: `pipeline/...` for the three
pipeline prompts, `skills/<skill_id>` for each domain skill's playbook, and
`evals/answer-judge` for the eval rubric.
Output skills are excluded -- their prompt files are loader placeholders; the
rendering is code.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.skills.manifest import PROMPTS_DIR as SKILL_PROMPTS_DIR, SkillManifest, load_all_manifests

logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).parent / "prompts"

PLANNER = "pipeline/planner"
EXTRACTION = "pipeline/extraction"
LEAD_IN = "pipeline/lead-in"
# The rubric the eval judge grades with (scripts/langfuse_eval.py). Not part of
# a run's pipeline; it lives in this catalog only so the same Langfuse editing
# and push/pull sync cover it.
EVAL_JUDGE = "evals/answer-judge"

_PIPELINE_FILES = {
    PLANNER: "planner.md",
    EXTRACTION: "extraction.md",
    LEAD_IN: "lead_in.md",
    EVAL_JUDGE: "eval_judge.md",
}


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    # The Langfuse prompt client, ONLY when the text actually came from
    # Langfuse. It is what links a generation to a prompt version; a fallback
    # must not be linked, or Langfuse would credit repo text to a version.
    handle: Any = None

    @property
    def version(self) -> int | None:
        return getattr(self.handle, "version", None)


def skill_prompt_name(skill_id: str) -> str:
    return f"skills/{skill_id}"


def langfuse_name(name: str) -> str:
    """The name in Langfuse: LANGFUSE_PROMPT_PREFIX + name."""
    return get_settings().langfuse_prompt_prefix + name


def catalog() -> dict[str, Path]:
    """Every editable prompt name -> its repo file. The sync script's list."""
    names = {name: PIPELINE_DIR / fn for name, fn in _PIPELINE_FILES.items()}
    for m in load_all_manifests().values():
        if m.kind == "domain":
            names[skill_prompt_name(m.id)] = SKILL_PROMPTS_DIR / m.prompt
    return names


def read_file(path: Path) -> str:
    return path.read_text().strip()


def _client():
    settings = get_settings()
    if not settings.langfuse_prompts:
        return None
    from app.obs.tracing import _client as langfuse_client

    return langfuse_client()


def get(name: str, path: Path) -> Prompt:
    fallback = read_file(path)
    client = _client()
    if client is None:
        return Prompt(name, fallback)

    settings = get_settings()
    try:
        # The SDK caches per (name, label). A stale entry is returned at once
        # and refreshed in a background thread, so only the very first fetch
        # of a name blocks -- warm() does those at startup.
        fetched = client.get_prompt(
            langfuse_name(name),
            label=settings.langfuse_prompt_label,
            type="text",
            fallback=fallback,
            cache_ttl_seconds=settings.langfuse_prompt_cache_ttl,
            max_retries=1,
            fetch_timeout_seconds=3,
        )
    except Exception:  # noqa: BLE001 -- an unreachable prompt store must not fail a run
        logger.warning("prompt %r: Langfuse fetch failed, using the repo file", name, exc_info=True)
        return Prompt(name, fallback)

    if fetched.is_fallback:
        return Prompt(name, fallback)
    # Raw text, NOT compile(): compile() substitutes {{var}}, and a business
    # user typing braces into a playbook should get braces, not a blank.
    text = (fetched.prompt or "").strip()
    if not text:
        logger.warning("prompt %r is empty in Langfuse; using the repo file", name)
        return Prompt(name, fallback)
    return Prompt(name, text, fetched)


def pipeline(name: str) -> Prompt:
    return get(name, PIPELINE_DIR / _PIPELINE_FILES[name])


def skill(manifest: SkillManifest) -> Prompt:
    return get(skill_prompt_name(manifest.id), SKILL_PROMPTS_DIR / manifest.prompt)


def linked(*prompts: Prompt):
    """Context for ONE model call: links its generation to the prompt version.

    Only the first Langfuse-served prompt is linked -- a generation carries a
    single prompt reference. Callers pass the primary prompt first.
    """
    handle = next((p.handle for p in prompts if p.handle is not None), None)
    if handle is None:
        return contextlib.nullcontext()
    try:
        from langfuse import propagate_attributes

        return propagate_attributes(prompt=handle)
    except Exception:  # noqa: BLE001
        logger.warning("could not link prompt to generation", exc_info=True)
        return contextlib.nullcontext()


def warm() -> dict[str, str]:
    """Fetch every prompt once at startup so no request pays the first-fetch
    latency, and report where each one is being served from."""
    if _client() is None:
        return {}
    sources = {}
    for name, path in catalog().items():
        p = get(name, path)
        sources[name] = f"langfuse v{p.version}" if p.handle is not None else "repo file"
    return sources
