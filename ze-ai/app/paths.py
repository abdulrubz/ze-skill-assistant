"""Where things live. The engine is `app/`; everything a project changes lives
under `domain/` (skills, agents, tools.yaml, vocabulary, eval banks), and the
API specs it is built from live under `specs/` and `generated/`.

Starting a new project on this template means editing domain/ and .env, and
nothing in app/."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOMAIN = ROOT / "domain"
SKILLS_DIR = DOMAIN / "skills"
SKILL_PROMPTS_DIR = SKILLS_DIR / "prompts"
AGENTS_DIR = DOMAIN / "agents"
AGENT_PROMPTS_DIR = AGENTS_DIR / "prompts"
TOOLS_YAML = DOMAIN / "tools.yaml"
VOCABULARY_YAML = DOMAIN / "vocabulary.yaml"
EVALS_DIR = DOMAIN / "evals"

# Snapshot of the spec repo (ze-api-specs) at the commit pinned in specs.lock.json:
# specs/<service>.openapi.json, one per service. Written by `tool_spec.py sync`.
SPECS_DIR = ROOT / "specs"
SPECS_LOCK = ROOT / "specs.lock.json"
# generated/<service>.tools.json, one per spec. Written by `tool_spec.py generate`.
GENERATED_DIR = ROOT / "generated"

VAR = ROOT / "var"
