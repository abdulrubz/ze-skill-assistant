#!/usr/bin/env python
"""Build a static, self-contained HTML map of the agents, skills and tools.

    python scripts/agent_graph.py                       # reads the API on :8000
    python scripts/agent_graph.py --url http://127.0.0.1:8000 --out docs/agent-graph.html

The page is one HTML file with the data embedded, so it opens from disk and can
be emailed or hosted as-is. Re-run it after a manifest changes; nothing in the
file updates itself. The data comes from the same endpoints the product serves
(`/agents`, `/skills`, `/tools`), so the map cannot drift from what the system
will actually allow.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fetch(base: str, path: str):
    try:
        with urllib.request.urlopen(f"{base.rstrip('/')}{path}", timeout=15) as r:
            return json.load(r)
    except OSError as exc:
        sys.exit(f"could not read {base}{path}: {exc}\nIs the ze-ai API running? (port 8000 by default)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=str(HERE.parent / "docs" / "agent-graph.html"))
    args = ap.parse_args()

    data = {
        "agents": fetch(args.url, "/agents"),
        "skills": fetch(args.url, "/skills"),
        "tools": fetch(args.url, "/tools"),
        "source": args.url,
        "generated": time.strftime("%Y-%m-%d"),
    }
    # `</` inside a script block would end it early; no value here contains it, but
    # the embed must not depend on that staying true.
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (HERE / "agent_graph.template.html").read_text().replace("/*__DATA__*/null", blob)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB; "
          f"{len(data['agents'])} agents, {len(data['skills'])} skills, {len(data['tools'])} tools)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
