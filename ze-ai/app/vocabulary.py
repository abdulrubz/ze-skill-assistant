"""domain/vocabulary.yaml: the words of the domain, loaded once.

Four human decisions live there, each consumed by one part of the engine:

  entities    the domain's nouns, with the cue phrases that signal each in a
              question                            -> app/skills/capability.py
  operations  which question shapes this domain supports (a subset of the
              engine's, whose cues are generic English) -> capability.py
  resolve     ids the deterministic resolve node pulls out of the question
              text, as regexes                     -> app/graph/nodes/resolve.py
  slots       what skills hand each other (`requires`/`provides`)
                                                   -> app/skills/slots.py
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.paths import VOCABULARY_YAML


class ResolvePattern(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: str = Field(description="Regex; group 1 if present, else the whole match")
    many: bool = Field(default=False, description="Collect every match as a list instead of the first")
    upper: bool = False

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        re.compile(v)
        return v


class Vocabulary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: dict[str, list[str]]
    operations: list[str]
    resolve: dict[str, ResolvePattern] = Field(default_factory=dict)
    slots: list[str] = Field(default_factory=list)


@lru_cache
def _load(path: str) -> Vocabulary:
    return Vocabulary.model_validate(yaml.safe_load(Path(path).read_text()))


def vocabulary(path: Path = VOCABULARY_YAML) -> Vocabulary:
    return _load(str(path))
