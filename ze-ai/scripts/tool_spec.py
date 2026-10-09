#!/usr/bin/env python
"""The spec snapshot, the generated tool files, and tools.yaml's fit to them.

    python scripts/tool_spec.py sync                  # pull the spec repo, regenerate, report changes, check
    python scripts/tool_spec.py sync --ref <sha|tag>  # ...at a specific commit
    python scripts/tool_spec.py generate              # regenerate generated/ from specs/
    python scripts/tool_spec.py check                 # CI: generated files current + tools.yaml fits them
    python scripts/tool_spec.py list [--q TEXT] [--service NAME]   # operations to choose from

Four places, three owners:

  ze-api-specs (git)              every service's OpenAPI spec      (API teams)
  specs/ + specs.lock.json        the snapshot this app runs on,    (sync; reviewed in a PR)
                                  pinned to a spec-repo commit
  generated/<service>.tools.json  every operation as a candidate    (generate; never edited)
  domain/tools.yaml               which tools agents may use        (AI team)

`sync` copies from SPEC_REPO_PATH (the monorepo's ../ze-api-specs), or clones
SPEC_REPO_URL at SPEC_REPO_REF when the spec repo lives on its own,
snapshots services/*/openapi.json into specs/, pins the commit in
specs.lock.json, regenerates, lists which configured tools' operations
changed, and runs the check. CI (.github/workflows/spec-sync.yml at the repo
root) runs it whenever a spec changes and opens a PR with the result, so a
tool never changes without a reviewed diff -- and never changes at runtime.

`check` exits non-zero when a generated file is stale or any tools.yaml entry
no longer fits its spec, one line per problem naming the tool and what moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.paths import GENERATED_DIR, SPECS_DIR, SPECS_LOCK  # noqa: E402
from app.tools.openapi.config import load_tools  # noqa: E402
from app.tools.openapi.drift import check, errors  # noqa: E402
from app.tools.openapi.generate import (  # noqa: E402
    load_generated,
    services,
    spec_digest,
    stale_services,
    write_all,
)

BOLD, DIM, RESET, RED, GREEN, YELLOW = "\033[1m", "\033[2m", "\033[0m", "\033[31m", "\033[32m", "\033[33m"


def run_check(generated: dict, verbose: bool) -> int:
    tools = load_tools()
    findings = check(generated, tools)
    bad = errors(findings)
    for f in findings:
        if f.level == "error":
            print(f"  {RED}ERROR{RESET} {BOLD}{f.tool}{RESET}: {f.message}")
        elif verbose:
            print(f"  {DIM}note  {f.tool}: {f.message}{RESET}")
    broken = sorted({f.tool for f in bad})
    if bad:
        print(f"\n{RED}{BOLD}{len(bad)} problem(s){RESET} in {len(broken)} tool(s): {', '.join(broken)}"
              f"  {DIM}({len(tools.tools) - len(broken)} tool(s) unaffected){RESET}")
        return 1
    print(f"{GREEN}tools.yaml fits the specs{RESET}: {len(tools.tools)} tool(s) OK")
    return 0


def _fingerprint(generated: dict, operation: str) -> str | None:
    op = generated.get("operations", {}).get(operation)
    if op is None:
        return None
    keep = {k: op[k] for k in ("method", "path", "kind", "input_schema", "output_schema", "documented")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _checkout(ref: str | None) -> tuple[Path, dict, Path | None]:
    """(services dir, lock info, temp dir to delete)."""
    settings = get_settings()
    if settings.spec_repo_path:
        root = Path(settings.spec_repo_path).expanduser().resolve()
        try:
            commit = _git("rev-parse", "HEAD", cwd=root)
            dirty = bool(_git("status", "--porcelain", "services", cwd=root))
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit, dirty = None, True
        # Recorded as configured (e.g. ../ze-api-specs in the monorepo), never
        # as an absolute path from whoever ran the sync.
        return root / "services", {"repo": settings.spec_repo_path, "ref": "(local path)", "commit": commit,
                                   "dirty": dirty}, None
    if not settings.spec_repo_url:
        sys.exit(f"{RED}set SPEC_REPO_URL (or SPEC_REPO_PATH for a local checkout){RESET}")
    ref = ref or settings.spec_repo_ref
    tmp = Path(tempfile.mkdtemp(prefix="ze-specs-"))
    _git("init", "-q", cwd=tmp)
    _git("remote", "add", "origin", settings.spec_repo_url, cwd=tmp)
    _git("fetch", "-q", "--depth", "1", "origin", ref, cwd=tmp)
    _git("checkout", "-q", "FETCH_HEAD", cwd=tmp)
    return tmp / "services", {"repo": settings.spec_repo_url, "ref": ref,
                              "commit": _git("rev-parse", "HEAD", cwd=tmp)}, tmp


def sync(ref: str | None, verbose: bool) -> int:
    src, lock, tmp = _checkout(ref)
    try:
        found = sorted(d.name for d in src.iterdir() if (d / "openapi.json").is_file()) if src.is_dir() else []
        if not found:
            print(f"{RED}no services/<name>/openapi.json found in {lock['repo']}{RESET}")
            return 1
        bad = []
        for name in found:
            version = str(json.loads((src / name / "openapi.json").read_text()).get("openapi", ""))
            if not version.startswith("3."):
                bad.append(f"{name} (openapi '{version or 'none'}')")
        if bad:
            print(f"{RED}OpenAPI 3.x required{RESET}: {', '.join(bad)} -- the spec repo's validate check should "
                  "have caught this")
            return 1

        old = load_generated() if any(GENERATED_DIR.glob("*.tools.json")) else {"operations": {}}
        SPECS_DIR.mkdir(exist_ok=True)
        for stale in SPECS_DIR.glob("*.openapi.json"):
            if stale.name.removesuffix(".openapi.json") not in found:
                stale.unlink()
                print(f"  {YELLOW}removed{RESET} {stale.name} (no longer in the spec repo)")
        for name in found:
            shutil.copyfile(src / name / "openapi.json", SPECS_DIR / f"{name}.openapi.json")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    docs = write_all()
    lock["services"] = {s: spec_digest(SPECS_DIR / f"{s}.openapi.json") for s in services()}
    SPECS_LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"synced {len(found)} service spec(s) from {lock['repo']} @ {(lock.get('commit') or '?')[:12]}: "
          + ", ".join(f"{s} ({len(d['operations'])} ops)" for s, d in docs.items()))
    if lock.get("dirty"):
        print(f"{YELLOW}warning{RESET}: the local spec checkout has uncommitted changes; "
              "the lock's commit does not describe them")

    new = load_generated()
    changed = []
    for name, tool in load_tools().tools.items():
        before, after = _fingerprint(old, tool.operation), _fingerprint(new, tool.operation)
        if before != after:
            changed.append((name, tool.operation, "removed" if after is None else "added" if before is None
                            else "changed"))
    if changed:
        print(f"{YELLOW}operations used by tools.yaml that changed:{RESET}")
        for name, op, state in changed:
            print(f"  {name} ({op}): {state}")
    else:
        print(f"{DIM}no operation tools.yaml uses has changed{RESET}")
    return run_check(new, verbose)


def list_operations(query: str | None, service: str | None) -> int:
    generated = load_generated()
    configured = {t.operation: name for name, t in load_tools().tools.items()}
    rows = []
    for key, op in generated["operations"].items():
        if service and op["service"] != service:
            continue
        if query and query.lower() not in f"{key} {op['path']} {op['summary']}".lower():
            continue
        rows.append((key, op))
    for key, op in rows:
        if key in configured:
            mark = f"{GREEN}configured as {configured[key]}{RESET}"
        elif op["kind"] == "write":
            mark = f"{DIM}write: not allowed (tools are read-only){RESET}"
        elif not op["agent_tool"]:
            mark = f"{DIM}x-not-an-agent-tool{RESET}"
        else:
            mark = "documented" if op["documented"] else f"{DIM}undocumented{RESET}"
        print(f"  {BOLD}{key}{RESET}  {DIM}{op['method']} {op['path']} ({op['kind']}){RESET}\n"
              f"      {op['summary']}  [{mark}]")
    print(f"\n{len(rows)} operation(s); only documented reads can be configured")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sync", "generate", "check", "list"])
    ap.add_argument("--ref", help="sync: spec repo branch, tag or commit (default SPEC_REPO_REF)")
    ap.add_argument("--q", help="list: substring of operation, path or summary")
    ap.add_argument("--service", help="list: only this service")
    ap.add_argument("--verbose", action="store_true", help="also print non-failing notes")
    args = ap.parse_args()
    if args.command == "sync":
        return sync(args.ref, args.verbose)
    if args.command == "generate":
        docs = write_all()
        print("regenerated: " + ", ".join(f"{s} ({len(d['operations'])} ops)" for s, d in docs.items()))
        return run_check(load_generated(), args.verbose)
    if args.command == "list":
        return list_operations(args.q, args.service)
    stale = stale_services()
    if stale:
        print(f"{RED}{BOLD}generated tool files are out of date{RESET} for {', '.join(stale)}: "
              "run scripts/tool_spec.py generate and commit the result")
        return 1
    return run_check(load_generated(), args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
