#!/usr/bin/env python
"""Keep the repo's prompt files and Langfuse Prompt Management in step.

With LANGFUSE_PROMPTS=true the app reads prompts from Langfuse (see
app/prompt_store.py), so Langfuse is where edits happen and the repo files are
the fallback plus the reviewable record. This script moves text between them:

    python scripts/prompts.py status          # what differs, nothing written
    python scripts/prompts.py push            # create prompts missing in Langfuse
    python scripts/prompts.py pull            # write live versions into the repo
    python scripts/prompts.py push --force skills/revenue_analytics
                                              # repo text -> new live version

The workflow this supports:
  1. `push` once, to seed Langfuse from the repo.
  2. Business users edit in the Langfuse UI. Each save is a new version; the
     app serves whichever version carries the label (default `production`).
  3. `pull` regularly, and commit the result. That is the code-review record of
     what changed, and it keeps tests and evals running the text production runs.

`push` never touches a prompt that already exists in Langfuse: doing so would
silently replace a business user's live edit with an older repo copy. `--force`
does exactly that, on purpose, for the names given (or `--all`) -- use it when
a developer changed a prompt in git and wants it live.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import prompt_store  # noqa: E402
from app.config import get_settings  # noqa: E402

BOLD, DIM, RESET, RED, GREEN, YELLOW = (
    "\033[1m", "\033[2m", "\033[0m", "\033[31m", "\033[32m", "\033[33m"
)


def _client():
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        sys.exit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set")
    from langfuse import Langfuse

    # A missing prompt is an expected answer here ("run push"), not a warning.
    logging.getLogger("langfuse").setLevel(logging.ERROR)
    return Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key,
                    host=s.langfuse_host)


def _live(client, name: str, label: str):
    """The labelled version, or None if the prompt (or the label) does not exist.
    cache_ttl_seconds=0: this script must see Langfuse, not the SDK's cache."""
    try:
        return client.get_prompt(prompt_store.langfuse_name(name), label=label, type="text", cache_ttl_seconds=0,
                                 max_retries=0, fetch_timeout_seconds=10)
    except Exception as exc:  # noqa: BLE001 -- the SDK raises a 404 as a generic error
        if "not found" in str(exc).lower() or "404" in str(exc):
            return None
        raise


def _rows(client, label):
    for name, path in prompt_store.catalog().items():
        repo = prompt_store.read_file(path)
        live = _live(client, name, label)
        yield name, path, repo, live


def status(client, label: str) -> int:
    drift = 0
    for name, path, repo, live in _rows(client, label):
        if live is None:
            print(f"  {YELLOW}missing{RESET}   {name}  {DIM}(not in Langfuse -- run push){RESET}")
            drift += 1
        elif live.prompt.strip() == repo:
            print(f"  {GREEN}in sync{RESET}   {name}  {DIM}v{live.version}{RESET}")
        else:
            print(f"  {RED}differs{RESET}   {name}  {DIM}live v{live.version} vs {path.name} "
                  f"-- pull to take the live text{RESET}")
            drift += 1
    return 1 if drift else 0


def push(client, label: str, force: list[str], force_all: bool) -> int:
    for name, path, repo, live in _rows(client, label):
        forced = force_all or name in force
        if live is not None and not forced:
            continue
        if live is not None and live.prompt.strip() == repo:
            print(f"  {DIM}unchanged {name}{RESET}")
            continue
        created = client.create_prompt(
            name=prompt_store.langfuse_name(name), prompt=repo, type="text", labels=[label],
            commit_message=f"{'force-' if live is not None else ''}push from git: {path.name}",
        )
        verb = "replaced live" if live is not None else "created"
        print(f"  {GREEN}{verb}{RESET}  {name}  {DIM}v{created.version} [{label}]{RESET}")
    client.flush()
    return 0


def pull(client, label: str) -> int:
    changed = 0
    for name, path, repo, live in _rows(client, label):
        if live is None:
            print(f"  {YELLOW}skipped{RESET}   {name}  {DIM}(not in Langfuse){RESET}")
            continue
        text = live.prompt.strip()
        if text == repo:
            continue
        diff = difflib.unified_diff(repo.splitlines(), text.splitlines(),
                                    f"repo/{path.name}", f"langfuse/{name} v{live.version}",
                                    lineterm="", n=1)
        print("\n".join(diff))
        path.write_text(text + "\n")
        print(f"  {GREEN}wrote{RESET}     {path}  {DIM}<- v{live.version}{RESET}\n")
        changed += 1
    print(f"{BOLD}{changed} file(s) updated{RESET}" + (" -- review and commit them." if changed else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "push", "pull"])
    ap.add_argument("--label", default=get_settings().langfuse_prompt_label,
                    help="label to read/write (default: LANGFUSE_PROMPT_LABEL)")
    ap.add_argument("--force", nargs="*", default=[], metavar="NAME",
                    help="push: overwrite these live prompts with the repo text")
    ap.add_argument("--all", action="store_true", help="push --force every prompt")
    args = ap.parse_args()

    client = _client()
    if args.command == "status":
        return status(client, args.label)
    if args.command == "push":
        return push(client, args.label, args.force, args.all)
    return pull(client, args.label)


if __name__ == "__main__":
    raise SystemExit(main())
