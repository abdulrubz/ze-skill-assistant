"""AgentManifest: scope + skills + prompt suffix. A named agent and free-form
chat traverse the EXACT same graph -- Admission loads whichever manifest
applies into state, and the planner's candidate catalog is the projection of
that agent's skills. The agent is a scope, not a different code path."""

from __future__ import annotations

import yaml
from pydantic import BaseModel, Field

from app.paths import AGENT_PROMPTS_DIR as PROMPTS_DIR, AGENTS_DIR as MANIFESTS_DIR


class AgentScope(BaseModel):
    tool_deny: list[str] = Field(
        default_factory=list, description="Explicit deny list, enforced on top of skill grants"
    )


class AgentManifest(BaseModel):
    id: str
    display_name: str
    summary: str = ""
    scope: AgentScope = Field(default_factory=AgentScope)
    skills: list[str]
    max_retries: int = 2
    prompt_suffix: str | None = None

    def prompt_suffix_text(self) -> str:
        if not self.prompt_suffix:
            return ""
        path = PROMPTS_DIR / self.prompt_suffix
        return path.read_text() if path.exists() else ""


def load_all_agent_manifests() -> dict[str, AgentManifest]:
    agents: dict[str, AgentManifest] = {}
    for path in sorted(MANIFESTS_DIR.glob("*.yaml")):
        agent = AgentManifest.model_validate(yaml.safe_load(path.read_text()))
        if agent.id != path.stem:
            raise ValueError(f"agent id '{agent.id}' must match filename '{path.stem}.yaml'")
        agents[agent.id] = agent
    return agents
