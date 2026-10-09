#!/usr/bin/env python
"""Run both services: customers on :9101, orders on :9102 (Ctrl-C stops both)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICES = {"customers": int(os.environ.get("CUSTOMERS_PORT", 9101)), "orders": int(os.environ.get("ORDERS_PORT", 9102))}

procs = [subprocess.Popen([sys.executable, "-m", "uvicorn", f"services.{name}.app:app", "--port", str(port),
                           "--host", "127.0.0.1"], cwd=ROOT) for name, port in SERVICES.items()]
try:
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    for p in procs:
        p.terminate()
