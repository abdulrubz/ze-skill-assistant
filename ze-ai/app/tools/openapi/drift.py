"""Drift check: does domain/tools.yaml still fit the generated tool files?

tools.yaml names operations and, for a few large responses, which fields to
keep. Everything else about a tool is generated, so drift here is about the
operation itself (gone, undocumented, deprecated), the output projection's
field paths, and inputs the spec's agent hints leave unsatisfiable.

This is what lets API teams move independently. Run it in CI whenever a spec
changes (`scripts/tool_spec.py check`); it fails with one line per broken
tool, naming the tool and exactly what moved, instead of letting the change
surface later as a model quietly getting worse answers.

Errors break the build. Notes (e.g. a new optional parameter the model cannot
see yet) are informational -- the allow-list means nothing new reaches the
model until someone decides it should.

What it cannot catch: a field that keeps its name but changes meaning. That is
what the eval banks are for.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.tools.openapi.config import (
    NestedField,
    ToolConfig,
    ToolsFile,
    operation,
    output_schema,
    resolve,
)


@dataclass(frozen=True)
class Finding:
    tool: str
    level: str  # "error" | "note"
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():5} {self.tool}: {self.message}"


_JSON_TYPES = {"boolean": bool, "integer": int, "number": (int, float), "string": str}


def _check_fixed_value(name: str, value, schema: dict) -> str | None:
    if "enum" in schema and value not in schema["enum"]:
        return f"fixed value {value!r} for '{name}' is not in the allowed values {schema['enum']}"
    expected = _JSON_TYPES.get(schema.get("type"))
    if expected and not (isinstance(value, expected) and not (expected is int and isinstance(value, bool))):
        return f"fixed value {value!r} for '{name}' is not a {schema.get('type')}"
    return None


def _walk(doc: dict, schema: dict | None, path: str, base: str = "") -> tuple[dict | None, str | None]:
    """Follow a dotted path through a response schema; (schema, error)."""
    node = resolve(doc, schema)
    walked = [base] if base else []
    for part in path.split("."):
        if node is None:
            break
        if node.get("type") == "array":
            return None, f"'{'.'.join(walked) or '(root)'}' is a list; use a nested {{from: ...}} for '{part}'"
        props = node.get("properties", {})
        if part not in props:
            where = ".".join(walked) or "the response"
            return None, f"field '{part}' no longer exists in {where}"
        node = resolve(doc, props[part])
        walked.append(part)
    return node, None


def _check_fields(doc, schema, fields, tool, findings, prefix="", base=""):
    for out_name, src in fields.items():
        if isinstance(src, str):
            _, err = _walk(doc, schema, src, base)
            if err:
                findings.append(Finding(tool, "error", f"output '{prefix}{out_name}': {err}"))
            continue
        assert isinstance(src, NestedField)
        node, err = _walk(doc, schema, src.from_, base)
        if err:
            findings.append(Finding(tool, "error", f"output '{prefix}{out_name}': {err}"))
            continue
        inner = resolve(doc, node.get("items")) if node.get("type") == "array" else node
        here = ".".join(x for x in (base, src.from_) if x) + ("[]" if node.get("type") == "array" else "")
        _check_fields(doc, inner, src.fields, tool, findings, prefix=f"{prefix}{out_name}.", base=here)


def check_tool(generated: dict, name: str, tool: ToolConfig) -> list[Finding]:
    findings: list[Finding] = []
    op = operation(generated, tool.operation)
    if op is None:
        if "." not in tool.operation:
            return [Finding(name, "error", f"operation '{tool.operation}' must be '<service>.<operationId>'")]
        return [Finding(name, "error", f"operation '{tool.operation}' no longer exists in the {tool.service} spec")]
    where = f"{op['method']} {op['path']}"

    if not op["agent_tool"]:
        findings.append(Finding(name, "error", f"'{tool.operation}' is marked x-not-an-agent-tool in its spec"))
    if not op["documented"]:
        findings.append(Finding(name, "error",
                                f"'{tool.operation}' is not documented for agents (no **Returns:** section): "
                                f"ask the {tool.service} team to document it in their spec first"))
    if op["deprecated"]:
        findings.append(Finding(name, "error", f"'{tool.operation}' is marked deprecated in the spec"))
    if op["kind"] == "write":
        findings.append(Finding(name, "error", f"'{tool.operation}' is a write ({where}); the tool layer is "
                                               "read-only, so only reads may be configured"))

    # Inputs come from the generated file; what can still go wrong is a
    # required input nobody can supply, or a pinned value the schema no
    # longer accepts. Both are fixed in the spec's agent hints.
    agent = op["agent"]
    props = op["input_schema"]["properties"]
    exposed = {e["key"] for e in agent["inputs"]}
    body_filled = any(e["key"].startswith("body[0].") for e in agent["inputs"])
    for key in op["input_schema"].get("required", []):
        if key in exposed or key in agent["fixed"] or (key == "body" and body_filled):
            continue
        if props.get(key, {}).get("in") == "header":
            continue  # audit headers are filled in by the tool layer
        findings.append(Finding(name, "error", f"required input '{key}' of {where} is neither an agent input nor "
                                               "fixed: fix the spec's agent hints"))
    for hint in agent.get("unresolved_hints", []):
        findings.append(Finding(name, "error", f"the spec's agent hint '{hint}' names an input {where} no longer "
                                               "has: fix the hint in the service's spec"))
    for key, value in agent["fixed"].items():
        schema = props.get(key) if key in props else None
        problem = _check_fixed_value(key, value, resolve(generated, schema)) if schema else None
        if problem:
            findings.append(Finding(name, "error", problem))

    if tool.output is not None:
        schema = output_schema(generated, op)
        if schema is None:
            findings.append(Finding(name, "error", "the success response no longer has a JSON schema"))
        else:
            is_array = schema.get("type") == "array"
            if is_array and not tool.output.as_:
                findings.append(Finding(name, "error", "the response is now a list; set output.as"))
            if not is_array and tool.output.as_:
                findings.append(Finding(name, "error", "the response is no longer a list; remove output.as"))
            row = resolve(generated, schema.get("items")) if is_array else schema
            _check_fields(generated, row, tool.output.fields, name, findings)
    return findings


def check(generated: dict, tools: ToolsFile) -> list[Finding]:
    findings: list[Finding] = []
    for name, tool in tools.tools.items():
        findings += check_tool(generated, name, tool)
    return findings


def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "error"]
