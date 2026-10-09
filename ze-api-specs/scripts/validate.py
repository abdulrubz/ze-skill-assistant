#!/usr/bin/env python
"""Validate every spec in services/*/openapi.json against the repo convention.

    python scripts/validate.py            # all services
    python scripts/validate.py orders     # one

Errors fail CI; warnings print but pass. The rules are what ze-ai's tool
generator relies on -- see README.md. Standard library only, so it runs in
any CI image without an install step.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"^[a-z][a-z0-9-]*$")
METHODS = ("get", "put", "post", "delete", "patch")


def check(service_dir: Path) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    name = service_dir.name
    if not NAME.match(name):
        errors.append(f"service folder '{name}' must be lowercase kebab-case")
    path = service_dir / "openapi.json"
    if not path.exists():
        return [f"{name}: missing openapi.json"], []
    try:
        spec = json.loads(path.read_text())
    except ValueError as exc:
        return [f"{name}: openapi.json is not valid JSON: {exc}"], []
    version = str(spec.get("openapi", spec.get("swagger", "")))
    if not version.startswith("3."):
        errors.append(f"{name}: OpenAPI 3.x required, found '{version or 'none'}' "
                      "(convert Swagger 2.0 first, e.g. with swagger2openapi)")
    seen: dict[str, str] = {}
    for route, item in spec.get("paths", {}).items():
        for method in METHODS:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            where = f"{name}: {method.upper()} {route}"
            opid = op.get("operationId")
            if not opid:
                errors.append(f"{where}: missing operationId (tools are keyed by it)")
                continue
            if opid in seen:
                errors.append(f"{where}: operationId '{opid}' already used by {seen[opid]}")
            seen[opid] = f"{method.upper()} {route}"
            desc = op.get("description") or ""
            if not (op.get("summary") or desc):
                errors.append(f"{where} ({opid}): no summary or description")
            if "**Returns:**" not in desc:
                warnings.append(f"{where} ({opid}): description has no '**Returns:**' section")
            for p in op.get("parameters", []):
                if "$ref" not in p and not p.get("description"):
                    warnings.append(f"{where} ({opid}): parameter '{p.get('name')}' has no description")
    return errors, warnings


def main() -> int:
    wanted = set(sys.argv[1:])
    dirs = sorted(d for d in (ROOT / "services").iterdir() if d.is_dir() and (not wanted or d.name in wanted))
    if not dirs:
        print("no services found")
        return 1
    total_errors = 0
    for d in dirs:
        errors, warnings = check(d)
        total_errors += len(errors)
        status = "FAIL" if errors else "ok"
        print(f"{status:4} {d.name}  ({len(warnings)} warning(s))")
        for e in errors:
            print(f"  ERROR {e}")
        for w in warnings:
            print(f"  warn  {w}")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
