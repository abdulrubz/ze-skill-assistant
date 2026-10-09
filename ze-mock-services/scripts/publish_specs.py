#!/usr/bin/env python
"""Write each service's OpenAPI spec into a ze-api-specs checkout.

    python scripts/publish_specs.py ../ze-api-specs

Real services do this from their own CI (open a PR against ze-api-specs, or
push to a branch it watches). The spec is generated from the code, never
hand-written, so it cannot drift from what the service actually serves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.customers.app import app as customers  # noqa: E402
from services.orders.app import app as orders  # noqa: E402

target = Path(sys.argv[1] if len(sys.argv) > 1 else "../ze-api-specs").resolve()
for name, app in {"customers": customers, "orders": orders}.items():
    out = target / "services" / name / "openapi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(app.openapi(), indent=2, sort_keys=False) + "\n")
    print(f"wrote {out}")
