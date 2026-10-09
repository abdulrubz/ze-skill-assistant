#!/usr/bin/env python
"""Print a dev JWT the services accept, e.g. for ze-ai's ZE_DEV_TOKEN.

    python scripts/mint_token.py alice "Alice Analyst" --hours 720
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.auth import mint  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("sub", nargs="?", default="dev-user")
p.add_argument("name", nargs="?")
p.add_argument("--hours", type=int, default=8)
a = p.parse_args()
print(mint(a.sub, a.name, ttl_s=a.hours * 3600))
