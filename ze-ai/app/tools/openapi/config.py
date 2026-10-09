"""domain/tools.yaml: the human half of the tool layer.

The generated half (generated/<service>.tools.json, from the API specs) says what every
operation IS: where it lives, what it takes, what it returns. tools.yaml says
which operations agents may use: the tool's name, the description the model
reads, and, where a response is large, which of its fields reach the model.
Arguments, their descriptions, pinned values and schemas are never restated
here: they come from the generated file (and the spec's agent hints).

This module parses tools.yaml, looks operations up in the generated file, and
projects responses; drift.py checks the two fit, server.py and writes.py turn
them into tools.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.tools.openapi.generate import load_generated

from app.paths import TOOLS_YAML as TOOLS_PATH


class ArgSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str | None = None


class NestedField(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    from_: str = Field(alias="from")
    fields: dict[str, OutField]


OutField = Union[str, NestedField]
NestedField.model_rebuild()


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    as_: str | None = Field(default=None, alias="as")
    fields: dict[str, OutField]


class ToolConfig(BaseModel):
    """One tool: the operation it is, and the human decisions about it. Its
    arguments, pinned values and schemas all come from the generated file."""

    model_config = ConfigDict(extra="forbid")
    # `<service>.<operationId>`, e.g. `orders.getOrder`: the service is the
    # spec repo folder name, and its base URL is SERVICE_<NAME>_URL.
    operation: str
    description: str
    group: str = "tools"
    cost_class: Literal["low", "medium", "high"] = "low"
    # Optional, for token cost only: which response fields reach the model,
    # under what names. Without it the model gets the whole response (minus
    # audit logs). Use it where a response is large.
    output: OutputSpec | None = None

    @property
    def service(self) -> str:
        return self.operation.split(".", 1)[0]


class ToolsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tools: dict[str, ToolConfig]


@lru_cache
def _load_tools(path: str) -> ToolsFile:
    return ToolsFile.model_validate(yaml.safe_load(Path(path).read_text()))


def load_tools(path: Path = TOOLS_PATH) -> ToolsFile:
    return _load_tools(str(path))


# --- the generated file ------------------------------------------------------

def operation(generated: dict, operation_id: str) -> dict | None:
    return generated.get("operations", {}).get(operation_id)


def _unwrap(node: Any) -> Any:
    """The one schema inside `allOf: [X]` (a description attached to a $ref) or
    `anyOf: [X, {type: null}]` (an optional model); otherwise the node itself."""
    if not isinstance(node, dict):
        return node
    if "allOf" in node and len(node["allOf"]) == 1:
        return node["allOf"][0]
    branches = [b for b in node.get("anyOf", []) if b.get("type") != "null"]
    if "anyOf" in node and len(branches) == 1:
        return branches[0]
    return node


def resolve(doc: dict, node: Any) -> Any:
    """Follow local $refs and single-schema wrappers until a concrete node."""
    seen = 0
    while seen < 32:
        node = _unwrap(node)
        if not (isinstance(node, dict) and "$ref" in node):
            break
        ref = node["$ref"]
        if not ref.startswith("#/"):
            raise ValueError(f"only local $refs are supported, got {ref}")
        node = doc
        for part in ref[2:].split("/"):
            node = node[part]
        seen += 1
    return node


def output_schema(generated: dict, op: dict) -> dict | None:
    schema = op.get("output_schema")
    return resolve(generated, schema) if schema else None


def agent_inputs(generated: dict, tool: ToolConfig) -> list[dict]:
    op = operation(generated, tool.operation)
    return op["agent"]["inputs"] if op else []


def model_args_schema(generated: dict, tool: ToolConfig) -> dict:
    """The argument schema the model sees: the operation's agent inputs,
    straight from the generated file."""
    props, req = {}, []
    for entry in agent_inputs(generated, tool):
        props[entry["name"]] = entry["schema"]
        if entry["required"]:
            req.append(entry["name"])
    return {"type": "object", "properties": props, "required": req}


# --- response projection ------------------------------------------------------

def _get(data: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(data, dict):
            return None
        data = data.get(part)
    return data


def _project_fields(data: Any, fields: dict[str, OutField]) -> dict:
    out: dict[str, Any] = {}
    for name, src in fields.items():
        if isinstance(src, str):
            out[name] = _get(data, src)
            continue
        value = _get(data, src.from_)
        if isinstance(value, list):
            out[name] = [_project_fields(v, src.fields) for v in value]
        elif isinstance(value, dict):
            out[name] = _project_fields(value, src.fields)
        else:
            out[name] = value
    return out


def strip_noise(data: Any) -> Any:
    """Audit logs only: they are empty unless asked for and never useful to a
    model. Nulls and empty lists stay -- here they carry meaning (a null
    paymentMethodId is "no payment method")."""
    if isinstance(data, dict):
        return {k: strip_noise(v) for k, v in data.items() if k != "auditLogs"}
    if isinstance(data, list):
        return [strip_noise(v) for v in data]
    return data


def project(data: Any, output: OutputSpec | None) -> Any:
    """Shape a raw API payload into what the model sees."""
    if output is None:
        return strip_noise(data)
    if isinstance(data, list):
        rows = [_project_fields(row, output.fields) for row in data]
        return {"count": len(rows), output.as_ or "items": rows}
    return _project_fields(data, output.fields)


# --- "Returns:" section from the spec's field descriptions --------------------

def _field_doc(doc: dict, schema: dict | None, path: str) -> tuple[dict | None, str | None]:
    node, desc = resolve(doc, schema), None
    for part in path.split("."):
        if not isinstance(node, dict):
            return None, None
        if node.get("type") == "array":
            node = resolve(doc, node.get("items"))
        prop = (node.get("properties") or {}).get(part)
        if prop is None:
            return None, None
        desc = prop.get("description") or _unwrap(prop).get("description") or resolve(doc, prop).get("description")
        node = resolve(doc, prop)
    return node, desc


def _short(desc: str | None, limit: int | None) -> str | None:
    if not desc or limit is None:
        return desc
    first = desc.split(". ")[0].rstrip(".")
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def _describe(doc: dict, schema: dict | None, fields: dict[str, OutField], prefix: str,
              limit: int | None = None) -> list[str]:
    lines: list[str] = []
    for out_name, src in fields.items():
        path = src if isinstance(src, str) else src.from_
        node, desc = _field_doc(doc, schema, path)
        desc = _short(desc, limit)
        is_list = isinstance(node, dict) and node.get("type") == "array"
        label = f"{prefix}{out_name}{'[]' if is_list else ''}"
        lines.append(f"- {label}: {desc}" if desc else f"- {label}")
        if not isinstance(src, str) and node is not None:
            inner = resolve(doc, node.get("items")) if is_list else node
            lines += _describe(doc, inner, src.fields, prefix=f"{label}.", limit=limit)
    return lines


def returns_section(generated: dict, tool: ToolConfig) -> str:
    """Markdown listing what the model will see, with the spec's descriptions."""
    op = operation(generated, tool.operation)
    schema = output_schema(generated, op) if op else None
    if schema is None:
        return ""
    is_array = schema.get("type") == "array"
    row = resolve(generated, schema.get("items")) if is_array else schema
    output = tool.output
    if output is not None:
        lines = _describe(generated, row, output.fields, prefix=f"{output.as_ or 'items'}[]." if is_array else "",
                          limit=90)
        head = f"Returns {{count, {output.as_ or 'items'}: [...]}}:" if is_array else "Returns:"
    else:
        # The whole response: field names only. Tool definitions are re-sent on
        # every step of a skill's loop, and the operation's own description
        # already says what matters about them.
        names = [n for n in (row.get("properties") or {}) if n != "auditLogs"]
        if not names:
            return ""
        return ("Returns a list of objects with: " if is_array else "Returns: ") + ", ".join(names)
    return "\n".join([head, *lines]) if lines else ""


def model_description(generated: dict, tool: ToolConfig) -> str:
    """The human description, then what the tool returns in the spec's words."""
    section = returns_section(generated, tool)
    return f"{tool.description.strip()}\n\n{section}" if section else tool.description.strip()


def default_generated() -> dict:
    return load_generated()
