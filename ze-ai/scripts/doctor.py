#!/usr/bin/env python
"""Check every dependency this service needs, in the order they matter.

    python scripts/doctor.py            # everything
    python scripts/doctor.py --no-model # skip the (cheap) live model calls

Run it first on a fresh machine: clone, copy .env.example to .env, fill it in,
run this. Each line is pass / warn / FAIL with the fix. Exit code 1 if
anything FAILED.

  1. the spec snapshot, generated tools and tools.yaml agree (offline)
  2. domain/ loads: skills, agents, vocabulary (offline)
  3. the spec repo is reachable (for the next sync)
  4. every service tools.yaml uses has a URL and answers
  5. a user token is available for the CLI and evals
  6. the MLflow gateway answers and has every endpoint the roles name
  7. each endpoint can generate, call tools across turns, and return
     structured output -- the three things the pipeline depends on
  8. Langfuse accepts the keys; prompts resolve (if prompt management is on)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings, service_env_var, service_url  # noqa: E402

GREEN, RED, YELLOW, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
FAILED: list[str] = []


def ok(what: str, detail: str = "") -> None:
    print(f"  {GREEN}pass{RESET}  {what}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


def warn(what: str, detail: str = "") -> None:
    print(f"  {YELLOW}warn{RESET}  {what}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


def fail(what: str, fix: str) -> None:
    FAILED.append(what)
    print(f"  {RED}FAIL{RESET}  {what}\n        {DIM}-> {fix}{RESET}")


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


def check_specs() -> list[str]:
    section("1. specs and tools")
    from app.paths import SPECS_LOCK
    from app.tools.openapi.config import load_tools
    from app.tools.openapi.drift import check, errors
    from app.tools.openapi.generate import load_generated, services, stale_services

    if not services():
        fail("no spec snapshot in specs/", "python scripts/tool_spec.py sync")
        return []
    lock = json.loads(SPECS_LOCK.read_text()) if SPECS_LOCK.exists() else {}
    ok(f"{len(services())} service spec(s): {', '.join(services())}",
       f"spec repo commit {(lock.get('commit') or '?')[:12]}")
    stale = stale_services()
    if stale:
        fail(f"generated tools out of date for {stale}", "python scripts/tool_spec.py generate")
    problems = errors(check(load_generated(), load_tools()))
    if problems:
        fail(f"{len(problems)} tools.yaml problem(s): {problems[0]}", "python scripts/tool_spec.py check")
    else:
        ok(f"tools.yaml fits the specs ({len(load_tools().tools)} tools)")
    return sorted({t.service for t in load_tools().tools.values()})


def check_domain() -> None:
    section("2. domain")
    try:
        from app.agents.manifest import load_all_agent_manifests
        from app.skills.registry import load_and_validate

        skills = load_and_validate()
        agents = load_all_agent_manifests()
        ok(f"{len(skills)} skill(s), {len(agents)} agent(s) pass the boot self-test")
    except Exception as exc:  # noqa: BLE001
        fail(f"domain/ does not load: {str(exc).splitlines()[0]}", "fix the manifest named above")


def check_spec_repo() -> None:
    section("3. spec repo")
    s = get_settings()
    if s.spec_repo_path:
        ok("SPEC_REPO_PATH", s.spec_repo_path) if Path(s.spec_repo_path).expanduser().is_dir() else \
            fail(f"SPEC_REPO_PATH {s.spec_repo_path} does not exist", "point it at a ze-api-specs checkout")
        return
    if not s.spec_repo_url:
        warn("SPEC_REPO_URL not set", "only needed to sync specs; the committed snapshot still works")
        return
    try:
        out = subprocess.run(["git", "ls-remote", s.spec_repo_url, s.spec_repo_ref], capture_output=True,
                             text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            ok(f"{s.spec_repo_url} @ {s.spec_repo_ref}", out.stdout.split()[0][:12])
        else:
            fail(f"cannot read {s.spec_repo_url} @ {s.spec_repo_ref}",
                 (out.stderr.strip().splitlines() or ["check the URL, ref and your git credentials"])[-1])
    except (OSError, subprocess.TimeoutExpired) as exc:
        fail(f"git ls-remote failed: {exc}", "is git installed and the network reachable?")


async def check_services(used: list[str]) -> None:
    section("4. services")
    async with httpx.AsyncClient(timeout=5) as http:
        for svc in used:
            url = service_url(svc)
            if not url:
                fail(f"{svc}: {service_env_var(svc)} is not set", f"add {service_env_var(svc)}=<base url> to .env")
                continue
            try:
                r = await http.get(url.rstrip("/") + "/health")
                # Any HTTP answer proves reachability; a 401/404 on /health is fine.
                ok(f"{svc}: {url}", f"HTTP {r.status_code}")
            except httpx.HTTPError as exc:
                fail(f"{svc}: {url} unreachable ({type(exc).__name__})", "is the service up and the URL right?")


def check_token() -> None:
    section("5. user token")
    if get_settings().ze_dev_token:
        ok("ZE_DEV_TOKEN set", "used by the CLI and evals only; the API forwards each caller's own token")
    else:
        warn("ZE_DEV_TOKEN not set", "needed for scripts/ask.py and evals; the API does not use it")


async def check_gateway(live: bool) -> None:
    section("6-7. MLflow gateway")
    s = get_settings()
    roles = {r: s.endpoint_for(r) for r in ("planner", "skill", "prose", "judge")}
    headers = {"Authorization": f"Bearer {s.mlflow_gateway_token}"} if s.mlflow_gateway_token else {}
    try:
        async with httpx.AsyncClient(timeout=10, headers=headers) as http:
            r = await http.get(f"{s.gateway_chat_base}/models")
            r.raise_for_status()
            existing = {m["id"] for m in r.json().get("data", [])}
    except Exception as exc:  # noqa: BLE001
        fail(f"{s.gateway_chat_base} unreachable ({type(exc).__name__}: {str(exc)[:80]})",
             "check MLFLOW_GATEWAY_URL (and MLFLOW_GATEWAY_TOKEN if it has auth on)")
        return
    ok(f"gateway answers: {s.mlflow_gateway_url}", f"{len(existing)} endpoint(s)")
    for role, ep in roles.items():
        if ep in existing:
            ok(f"{role:8s} -> {ep}")
        else:
            fail(f"{role:8s} -> {ep}: no such endpoint", "create it in MLflow (AI Gateway -> Endpoints) or fix "
                 f"GATEWAY_ENDPOINT or GATEWAY_{role.upper()}_ENDPOINT")
    if live:
        for ep in sorted(set(roles.values()) & existing):
            await check_endpoint(ep)


async def check_endpoint(endpoint: str) -> None:
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_core.tools import tool
    from pydantic import BaseModel

    from app.llm import _build, message_text

    @tool
    def get_order(order_id: str) -> str:
        """Look up an order by id."""
        return "{}"

    class Shape(BaseModel):
        total: float
        status: str

    model = _build(endpoint, temperature=0.0)
    label = f"{endpoint}:"
    try:
        r = await model.ainvoke("Reply with exactly: OK")
        ok(f"{label} generation", message_text(r.content).strip()[:30])
    except Exception as exc:  # noqa: BLE001
        fail(f"{label} generation: {str(exc)[:160]}", "check the endpoint's model and secret in MLflow")
        return
    try:
        bound = model.bind_tools([get_order])
        msgs = [HumanMessage("Look up order ORD-10001 with the tool, then answer in one line.")]
        first = await bound.ainvoke(msgs)
        calls = first.tool_calls
        if not calls or calls[0]["args"].get("order_id") != "ORD-10001":
            fail(f"{label} tool calling: no correct call ({calls})", "the model behind it must support tool calls")
        else:
            msgs += [first, ToolMessage('{"order_id": "ORD-10001", "status": "shipped"}',
                                        tool_call_id=calls[0]["id"])]
            await bound.ainvoke(msgs)  # the second turn is what thought signatures break
            ok(f"{label} tool calling across turns")
    except Exception as exc:  # noqa: BLE001
        fail(f"{label} tool calling: {str(exc)[:160]}", "see app/llm.py MlflowChat")
    try:
        parsed = await model.with_structured_output(Shape).ainvoke("Order total 12.5, status pending. Extract.")
        ok(f"{label} structured output", str(parsed))
    except Exception as exc:  # noqa: BLE001
        fail(f"{label} structured output: {str(exc)[:160]}", "the planner and fact extraction need this")


def check_langfuse() -> None:
    section("8. Langfuse")
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        warn("LANGFUSE_* keys not set", "tracing, prompt management and evals are off")
        return
    from app.obs.tracing import _client

    client = _client()
    try:
        if client is None or not client.auth_check():
            raise RuntimeError("auth_check returned false")
        ok(f"keys accepted by {s.langfuse_host}", f"environment={s.langfuse_environment or '(default)'}")
    except Exception as exc:  # noqa: BLE001
        fail(f"Langfuse rejected the keys or is unreachable: {str(exc)[:100]}", "check LANGFUSE_HOST and keys")
        return
    if s.langfuse_prompts:
        from app import prompt_store

        sources = prompt_store.warm()
        missing = [n for n, src in sources.items() if not src.startswith("langfuse")]
        if missing:
            warn(f"{len(missing)} prompt(s) served from repo files, not Langfuse",
                 f"python scripts/prompts.py push   ({', '.join(missing[:3])}...)")
        else:
            ok(f"all {len(sources)} prompts served from Langfuse",
               f"label {s.langfuse_prompt_label}, prefix '{s.langfuse_prompt_prefix}'")
    else:
        ok("prompt management off", "LANGFUSE_PROMPTS=true to serve prompts from Langfuse")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-model", action="store_true", help="skip live model calls")
    args = ap.parse_args()

    used = check_specs()
    check_domain()
    check_spec_repo()
    await check_services(used)
    check_token()
    await check_gateway(live=not args.no_model)
    check_langfuse()

    print()
    if FAILED:
        print(f"{RED}{BOLD}{len(FAILED)} check(s) failed.{RESET}")
        return 1
    print(f"{GREEN}{BOLD}ready.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
