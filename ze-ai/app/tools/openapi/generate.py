"""The generated half of the tool layer: spec snapshots -> generated tool files.

    specs/<service>.openapi.json    (each service's spec, synced from the spec repo)
        --generate()-->              generated/<service>.tools.json  (never hand-edited)
        + domain/tools.yaml          (the AI team's choices: names, descriptions)
        = the tools skills can use

Operations and schemas are NAMESPACED by service -- `orders.getOrder`,
`#/schemas/orders.Order` -- because operationIds and schema names collide
across teams (every service has a `getById` and an `HTTPValidationError`).
load_generated() merges every service's file into one document of that shape.

Each generated file lists EVERY operation in its spec as a candidate tool:
where it lives (method, path), whether it reads or writes, its input schema
(every parameter and body field, with where each goes) and its output schema.
tools.yaml then picks a few of them by operationId and says how the model sees
them. Regenerating after a spec change shows, in the file's diff, exactly which
tool schemas moved; `scripts/tool_spec.py check` fails when the committed file
is stale or tools.yaml no longer fits it.

Input names follow FastMCP's OpenAPI tool generator, which runs the tools at
runtime: parameters by their spec name, object body fields flattened beside
them, a parameter that collides with a body field suffixed `__<location>`,
and an array body as one `body` argument (tests/test_generated_tools.py
checks the two agree).

Each operation also gets `agent`: what a tool generated from it looks like to
a model, derived from the schema and the spec's own agent hints (x-agent,
x-agent-hidden) rather than from anything in tools.yaml:

  inputs    the arguments a model sets, under their spec names, with the
            spec's descriptions: every path, query and body input except
            headers (filled in by the tool layer), inputs the spec marks
            hidden or fixed, and a body field that repeats a path or query
            parameter's name. An array body's element fields stand in for
            the array. With an `inputs` hint, only those.
  fixed     inputs always sent with a pinned value
  defaults  values used when the model leaves an input out
  idempotency_key  the input a write's idempotency key goes in
"""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


from app.paths import GENERATED_DIR, SPECS_DIR

METHODS = ("get", "put", "post", "delete", "patch")
NOT_A_TOOL = "x-not-an-agent-tool"
READ_ONLY = "x-read-only"
SCHEMA_PREFIX = "#/components/schemas/"
GENERATED_SCHEMA_PREFIX = "#/schemas/"


def spec_path(service: str) -> Path:
    return SPECS_DIR / f"{service}.openapi.json"


def generated_path(service: str) -> Path:
    return GENERATED_DIR / f"{service}.tools.json"


def services() -> list[str]:
    """Every service with a spec snapshot."""
    return sorted(p.name.removesuffix(".openapi.json") for p in SPECS_DIR.glob("*.openapi.json"))


def spec_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_refs(node: Any, service: str) -> Any:
    """Component refs point into the generated document's own `schemas`
    section, under the service's namespace."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "$ref" and isinstance(v, str) and v.startswith(SCHEMA_PREFIX):
                out[k] = f"{GENERATED_SCHEMA_PREFIX}{service}.{v[len(SCHEMA_PREFIX):]}"
            else:
                out[k] = _rewrite_refs(v, service)
        return out
    if isinstance(node, list):
        return [_rewrite_refs(v, service) for v in node]
    return node


def _deref(spec: dict, node: Any) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        target = spec
        for part in node["$ref"][2:].split("/"):
            target = target[part]
        node = target
    return node


def _body_schema(spec: dict, op: dict) -> dict | None:
    body = _deref(spec, op.get("requestBody"))
    if not body:
        return None
    content = body.get("content", {})
    media = content.get("application/json") or next(iter(content.values()), {})
    return media.get("schema")


def _raw_inputs(spec: dict, item: dict, op: dict) -> dict:
    params = [_deref(spec, p) for p in [*item.get("parameters", []), *op.get("parameters", [])]]
    body = _body_schema(spec, op)
    body_obj = _deref(spec, body) if body else None
    body_props: dict = {}
    body_required: list[str] = []
    array_body = False
    if body_obj is not None:
        if body_obj.get("type") == "array" or "properties" not in body_obj:
            array_body = True
        else:
            body_props = body_obj.get("properties", {})
            body_required = list(body_obj.get("required", []))

    props: dict[str, dict] = {}
    required: list[str] = []
    for p in params:
        name = p["name"]
        key = f"{name}__{p['in']}" if name in body_props else name
        entry = {**copy.deepcopy(p.get("schema", {})), "in": p["in"], "spec_name": name}
        if p.get("description"):
            entry["description"] = p["description"]
        if p.get("x-agent-hidden"):
            entry["agent_hidden"] = True
        props[key] = entry
        if p.get("required"):
            required.append(key)
    for name, schema in body_props.items():
        props[name] = {**copy.deepcopy(schema), "in": "body", "spec_name": name}
        if name in body_required:
            required.append(name)
    if array_body:
        props["body"] = {**copy.deepcopy(body), "in": "body", "spec_name": "(body)",
                         "description": "The whole request body"}
        if _deref(spec, op.get("requestBody")).get("required"):
            required.append("body")
    return {"type": "object", "properties": props, "required": required}


def _agent(spec: dict, op: dict, inputs: dict, service: str) -> dict:
    hints = op.get("x-agent") or {}
    props, required = inputs["properties"], set(inputs["required"])
    # Hints use spec names; fixed values are keyed by generated input name. A
    # path or query parameter wins a name it shares with a body field.
    by_spec: dict[str, str] = {}
    for key, entry in sorted(props.items(), key=lambda kv: kv[1]["in"] == "body"):
        if key != "body":
            by_spec.setdefault(entry["spec_name"], key)
    element: dict = {}
    if "body" in props:
        items = _deref(spec, _deref(spec, props["body"]).get("items"))
        element = (items or {}).get("properties", {})
        element_required = set((items or {}).get("required", []))
        for field in element:
            by_spec.setdefault(field, f"body[0].{field}")

    known = set(by_spec)
    # A hint naming an input the operation no longer has would silently stop
    # pinning a value or guarding a write; record it for the drift check.
    unresolved = sorted(
        {f"fixed.{n}" for n in (hints.get("fixed") or {}) if n not in known}
        | {f"defaults.{n}" for n in (hints.get("defaults") or {}) if n not in known}
        | {f"hidden.{n}" for n in (hints.get("hidden") or []) if n not in known}
        | {f"inputs.{n}" for n in (hints.get("inputs") or []) if n not in known}
        | ({f"idempotency_key.{hints['idempotency_key']}"} if hints.get("idempotency_key") and
           hints["idempotency_key"] not in known else set())
    )
    fixed = {by_spec[name]: value for name, value in (hints.get("fixed") or {}).items() if name in by_spec}
    hidden = set(hints.get("hidden") or [])
    allow = set(hints["inputs"]) if hints.get("inputs") else None
    params = {e["spec_name"] for e in props.values() if e["in"] in ("path", "query")}

    out: list[dict] = []

    def consider(key: str, name: str, schema: dict, location: str, is_required: bool):
        if location == "header" or schema.get("agent_hidden") or name in hidden or key in fixed:
            return
        if allow is not None and name not in allow:
            return
        if location == "body" and not key.startswith("body[") and name in params:
            return  # the path/query parameter of the same name is the one a model sets
        clean = {k: v for k, v in schema.items() if k not in ("in", "spec_name", "agent_hidden")}
        out.append({"name": name, "key": key, "in": location, "required": is_required, "schema": _rewrite_refs(clean, service)})

    for key, entry in props.items():
        if key == "body" and element:
            for field, schema in element.items():
                consider(f"body[0].{field}", field, copy.deepcopy(schema), "body", field in element_required)
            continue
        consider(key, entry["spec_name"], entry, entry["in"], key in required)

    defaults = dict(hints.get("defaults") or {})
    idem = hints.get("idempotency_key")
    return {
        "inputs": out,
        "fixed": fixed,
        "defaults": defaults,
        "idempotency_key": by_spec.get(idem) if idem else None,
        "unresolved_hints": unresolved,
    }


def _output(spec: dict, op: dict, service: str) -> tuple[dict | None, list[int], str | None]:
    codes = sorted(int(c) for c in op.get("responses", {}) if str(c).startswith("2"))
    schema, description = None, None
    for code in codes:
        resp = _deref(spec, op["responses"][str(code)])
        description = description or resp.get("description")
        content = resp.get("content", {})
        media = content.get("application/json")
        if media and media.get("schema"):
            schema = media["schema"]
            break
    return (_rewrite_refs(copy.deepcopy(schema), service) if schema else None), codes, description


def generate(spec: dict, digest: str, service: str) -> dict:
    operations: dict[str, dict] = {}
    for path, item in spec.get("paths", {}).items():
        for method in METHODS:
            op = item.get(method)
            if not isinstance(op, dict) or "operationId" not in op:
                continue
            output, codes, response_description = _output(spec, op, service)
            description = op.get("description") or ""
            raw_inputs = _raw_inputs(spec, item, op)
            operations[f"{service}.{op['operationId']}"] = {
                "service": service,
                "operation_id": op["operationId"],
                "method": method.upper(),
                "path": path,
                "kind": "read" if method == "get" or op.get(READ_ONLY) else "write",
                "agent_tool": not op.get(NOT_A_TOOL),
                "documented": "**Returns:**" in description,
                "deprecated": bool(op.get("deprecated")),
                "tags": op.get("tags", []),
                "summary": op.get("summary", ""),
                "description": description,
                "input_schema": _rewrite_refs(raw_inputs, service),
                "agent": _agent(spec, op, raw_inputs, service),
                "output_schema": output,
                "success_codes": codes,
                "response_description": response_description,
            }
    schemas = spec.get("components", {}).get("schemas", {})
    return {
        "_generated": f"by scripts/tool_spec.py generate from specs/{service}.openapi.json. Do not edit: regenerate.",
        "service": service,
        "spec_sha256": digest,
        "operations": dict(sorted(operations.items())),
        "schemas": {f"{service}.{k}": _rewrite_refs(copy.deepcopy(v), service) for k, v in schemas.items()},
    }


def write_generated(service: str) -> dict:
    path = spec_path(service)
    doc = generate(json.loads(path.read_text()), spec_digest(path), service)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    generated_path(service).write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    _load_all.cache_clear()
    return doc


def write_all() -> dict[str, dict]:
    """Regenerate every service, and drop generated files whose spec is gone."""
    docs = {s: write_generated(s) for s in services()}
    for stale in GENERATED_DIR.glob("*.tools.json"):
        if stale.name.removesuffix(".tools.json") not in docs:
            stale.unlink()
    _load_all.cache_clear()
    return docs


def merge(docs: list[dict]) -> dict:
    """One document over every service: operations and schemas are already
    namespaced, so this is a plain union."""
    merged: dict = {"operations": {}, "schemas": {}, "services": {}}
    for doc in docs:
        merged["operations"].update(doc["operations"])
        merged["schemas"].update(doc["schemas"])
        merged["services"][doc["service"]] = doc["spec_sha256"]
    return merged


@lru_cache
def _load_all() -> dict:
    return merge([json.loads(p.read_text()) for p in sorted(GENERATED_DIR.glob("*.tools.json"))])


def load_generated() -> dict:
    return _load_all()


def stale_services() -> list[str]:
    """Services whose generated file is missing or no longer matches the spec,
    plus generated files with no spec left."""
    out = []
    for s in services():
        g = generated_path(s)
        if not g.exists() or json.loads(g.read_text()).get("spec_sha256") != spec_digest(spec_path(s)):
            out.append(s)
    out += [p.name.removesuffix(".tools.json") for p in GENERATED_DIR.glob("*.tools.json")
            if not spec_path(p.name.removesuffix(".tools.json")).exists()]
    return sorted(out)


def is_stale() -> bool:
    return bool(stale_services())
