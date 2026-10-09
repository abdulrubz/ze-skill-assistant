# ze-ai

A GenAI backend service that answers natural-language questions from the
organisation's microservices. Questions run through a registry of **agents →
skills → tools**, orchestrated by a deterministic LangGraph pipeline. Every tool
is **generated from a service's OpenAPI spec**, and every answer is **cited**:
each number traces back to the tool call that produced it.

- **Model plane:** the MLflow AI Gateway and nothing else. The app knows endpoint
  names; MLflow decides the model (e.g. a Databricks serving endpoint).
- **Tools:** generated from the specs in `../ze-api-specs` (the sibling folder in
  this monorepo). Tools are read-only, and every call carries the end user's own
  bearer token.
- **Observability, prompt management and evals:** Langfuse.

Backend only: an HTTP API (`POST /ask` streams server-sent events) plus a CLI.
There is no UI.

New to the code? Start with the [codebase walkthrough](docs/codebase-walkthrough.md).

```
                      ze-api-specs (git)            MLflow AI Gateway ──> Databricks / any provider
                      services/<svc>/openapi.json          ▲
                             │ tool_spec.py sync           │ /gateway/mlflow/v1/chat/completions
                             ▼                             │ (model = endpoint name)
 client ──Bearer──> POST /ask ─> admission → resolve → plan → skills ⇉ join → validate → respond
                                                              │
                                     generated tools ─────────┘── user's token ──> microservices
                                                              │
                                                     Langfuse: traces, prompts, eval datasets
```

---

## Quick start (fresh machine / VDI)

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/) (or plain `venv`
+ `pip`), git, and network access to the gateway, Langfuse and the services.

```bash
git clone git@github.com:abdulrubz/ze-skill-assistant.git && cd ze-skill-assistant/ze-ai
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env          # fill in URLs and keys (see Configuration)
python scripts/doctor.py      # checks every dependency, prints the fix for each failure
```

When doctor says `ready.`:

```bash
python scripts/ask.py "who are our top 3 customers by spend?"     # CLI (uses ZE_DEV_TOKEN)
uvicorn app.app:app --port 8000                                    # the API
curl -N -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" \
     -H 'content-type: application/json' -d '{"question":"who is Globex"}'
```

The API's own Swagger UI is at `/docs`, with an Authorize button for the bearer
token.

### The whole stack on a laptop

To exercise everything without the real services:

```bash
# 1. two dummy services (customers :9101, orders :9102) -- the sibling ze-mock-services folder
cd ../ze-mock-services && uv venv && uv pip install -e ".[dev]" && python scripts/run.py &
python scripts/mint_token.py alice "Alice Analyst" --hours 720     # -> ZE_DEV_TOKEN in ze-ai/.env

# 2. an MLflow gateway with Gemini-backed endpoints standing in for Databricks
cd ../ze-ai && uv pip install -e ".[gateway]"
MLFLOW_CRYPTO_KEK_PASSPHRASE=<any local passphrase> mlflow server --port 5050 \
    --backend-store-uri sqlite:///.dev/mlflow.db &
GOOGLE_API_KEY=... python scripts/mlflow_gateway.py dev-provision

# 3. Langfuse: any instance; set the LANGFUSE_* keys in .env
python scripts/prompts.py push                                      # seed prompts once
python scripts/doctor.py
```

---

## How it works

### Layers

| layer | where | what |
|---|---|---|
| pipeline | `app/graph/` | Explicit `StateGraph`. Only `plan` and the skill loops call a model; join, validation and termination are deterministic code. |
| agents | `domain/agents/` | A scope: which skills an agent may use, plus a prompt suffix. Every agent runs the same graph. |
| skills | `domain/skills/` | A YAML manifest + a playbook prompt, compiled into a bounded ReAct subgraph by one factory (`app/skills/factory.py`). Adding a skill changes no code. |
| tools | `domain/tools.yaml` + `generated/` | Generated from service specs; `tools.yaml` picks which operations agents may use and how they are described. |

The planner reasons over skill **declarations** only (`entities`, `operations`,
`answers`, `requires`, `provides`, `cannot`), never tool names or prompts. A
boot self-test refuses to start the service if a manifest is mis-wired.

### Answers are cited and checked

Every value a skill reports is a `Fact` with a citation ref and a `Provenance`
naming the tool call that produced it. The final answer's numbers are built
from those facts, not from model prose. Before anything is returned,
deterministic checks run (`app/validation/`): every citation resolves, every
fact points at a real tool call, derived values recompute, and no number in the
narrative is uncited. A failing step is retried on its own (bounded);
exhaustion marks the answer `degraded` rather than hiding it.

### Tools from the spec repo

```
ze-api-specs (git)          services/<svc>/openapi.json        API teams own these
  └─ tool_spec.py sync ──>  specs/<svc>.openapi.json           snapshot, pinned in specs.lock.json
                            generated/<svc>.tools.json         every operation as a candidate tool
                            domain/tools.yaml                  which ones agents may use (AI team)
```

- Operations are named `<service>.<operationId>` (e.g. `orders.getOrder`), and
  schemas are namespaced the same way, so specs from different teams never
  collide.
- Each service's base URL comes from `SERVICE_<NAME>_URL`.
- `tools.yaml` holds only human decisions: the tool name, the description the
  model reads, and optionally which response fields reach the model. Arguments
  and schemas are never restated; they come from the spec.
- Read-only: the drift check rejects a write operation in `tools.yaml`.

**When a spec changes** under `ze-api-specs/services/`, the root workflow
`.github/workflows/spec-sync.yml` runs `tool_spec.py sync` and opens a PR with
the new snapshot and generated files. Tools never change at runtime and
never without a reviewed diff. New operations show up in `tool_spec.py list` as
candidates; nothing reaches a model until someone adds it to `tools.yaml`.

```bash
python scripts/tool_spec.py sync                 # pull specs, regenerate, report changes, check
python scripts/tool_spec.py list --service orders
python scripts/tool_spec.py check                # CI: snapshot, generated files and tools.yaml agree
```

### The user's token, forwarded

`POST /ask` requires `Authorization: Bearer <token>` and forwards that token
unchanged on every service call, so each service authorizes the real user. The
assistant can never see more than the person asking, and there is no service
account. The token lives only in request-scoped runtime context
(`app/tools/auth.py`). It is never in graph state, prompts, trace events,
Langfuse or the checkpointer. A call with no token is refused before it is
sent.

### Model plane: MLflow only

Every call goes to MLflow's provider-agnostic route,
`{MLFLOW_GATEWAY_URL}/gateway/mlflow/v1/chat/completions`, with `model` = an
endpoint name (`GATEWAY_ENDPOINT`, `GATEWAY_JUDGE_ENDPOINT`, optional per-role
overrides). To change model, edit the endpoint in MLflow; this app does not
change. Budgets and usage tracking are MLflow's. The rate limit
(`MODEL_REQUESTS_PER_SECOND`) is here, because MLflow caps spend but not rate.

`app/llm.py`'s `MlflowChat` adds two things the OpenAI client drops: providers'
tool-call **thought signatures** (Gemini 3 rejects the second turn of every tool
loop without them), and reasoning tokens reported outside `completion_tokens`
(billed as output, so the cost ledger counts them).

### Langfuse

- **Tracing:** every run is a trace (session = thread id, tagged by agent, in
  `LANGFUSE_ENVIRONMENT`). Generations are named by role (`planner`,
  `skill:<id>`, `prose`), so cost groups by skill.
- **Prompt management:** with `LANGFUSE_PROMPTS=true`, the pipeline prompts and
  each skill's playbook are served from Langfuse at `LANGFUSE_PROMPT_LABEL` and
  are editable in its UI without a deploy. Repo files are the fallback and the
  review record (`scripts/prompts.py push | pull | status`). Generated parts of
  a skill prompt (`cannot`, caveats, the step budget, the citation rule) stay in
  code, so no edit can remove them. `LANGFUSE_PROMPT_PREFIX` namespaces prompts
  and datasets when apps share a project.
- **Evals:** `domain/evals/` banks become Langfuse datasets. Each run is scored
  by code checks and an LLM judge, and gated against `eval_targets.yaml`.

```bash
python scripts/eval_coverage.py                                  # routing, planner only, fast
python scripts/langfuse_eval.py sync
python scripts/langfuse_eval.py run --bank answer --label baseline
python scripts/langfuse_eval.py report --run <run-name>
```

---

## Adapting it to your services

Everything project-specific is in `domain/`, the spec snapshot and `.env`.
`app/` should not need to change.

1. **Specs:** each service publishes `services/<name>/openapi.json` to the spec
   repo (OpenAPI 3, an `operationId` on every operation, a `**Returns:**`
   section in descriptions; see ze-api-specs' README). Run
   `python scripts/tool_spec.py sync` and set `SERVICE_<NAME>_URL` for each one.
2. **Tools:** `python scripts/tool_spec.py list`, then add the reads agents
   should use to `domain/tools.yaml`.
3. **Vocabulary:** in `domain/vocabulary.yaml`, set your nouns (`entities` +
   cue phrases), the question shapes you support (`operations`), id patterns
   pulled from question text (`resolve`), and what skills hand each other
   (`slots`).
4. **Skills:** add `domain/skills/<id>.yaml` + `prompts/<id>.md`, granting tools
   from `tools.yaml`. Write many `answers` examples, declare `cannot`. Delete the
   dummy `customer_profile` / `order_analysis`.
5. **Agents:** add `domain/agents/<id>.yaml` scoping skills. Keep `general`.
6. **Evals:** replace `domain/evals/*_bank.yaml` with real questions and verified
   answers. Replace `tests/test_domain.py` with guards for your declarations.
7. `python scripts/doctor.py && pytest -q && python scripts/eval_coverage.py`.

The boot self-test tells you exactly what is mis-wired: an ungranted tool, an
unknown slot or entity, or an entity no skill claims.

---

## Configuration

All configuration is environment variables (or `.env`), documented in
[.env.example](.env.example). In summary: the MLflow gateway URL and endpoint
names, the spec repo URL and ref, `SERVICE_<NAME>_URL` per service, Langfuse
host and keys, and `ZE_DEV_TOKEN` for the CLI and evals only. The URLs and keys
for Langfuse, MLflow and the services all live here, so the same image runs in
any environment.

## Scripts

| script | what |
|---|---|
| `doctor.py` | checks specs, domain, spec repo, services, gateway endpoints (live tool-call round trip), Langfuse |
| `ask.py` | one question from the CLI with a live trace |
| `tool_spec.py` | `sync` / `generate` / `check` / `list` |
| `prompts.py` | `push` / `pull` / `status` between repo prompt files and Langfuse |
| `eval_coverage.py` | planner routing over the routing bank |
| `langfuse_eval.py` | `sync` banks to datasets, `run` scored experiments, `report` vs targets |
| `eval_cost.py` | end-to-end cost + quality, comparable across changes |
| `mlflow_gateway.py` | `show` endpoints, `budget`, `retention`, `dev-provision` (local only) |
| `agent_graph.py` | static HTML map of agents, skills and tools from a running API |

## Deliberately left out

- **Write tools** and the human-approval node: the tool layer is read-only. The
  engine this came from had a confirm-before-execute flow; add it back when a
  write is genuinely needed.
- **Provider-specific prompt caching:** it was Gemini-only, with sharp edges.
  Whether caching applies now depends on the endpoint's provider.
- **Direct SQL / analytics tools:** everything goes through service APIs.
- **A UI:** the API streams typed events (`answer_ready` carries cited blocks);
  any frontend can render them.

## Facts worth knowing

- MLflow's unified route returns thought signatures, but the stock OpenAI client
  drops them. Without `MlflowChat`, Gemini 3 tool loops fail on turn two with
  "missing a thought_signature".
- MLflow's own spend figure leaves out reasoning tokens, so its budget runs low.
  Set budgets below the real cap. This app's ledger (`GET /costs`) counts them.
- Langfuse OSS does not protect prompt labels: anyone with project edit access
  can move `production`. Governance is a process rule.
- LangGraph evaluates a conditional edge once per parallel task, against that
  task's own writes only. Wave scheduling therefore happens in its own node
  (`schedule`), after the merge. Before that fix, every multi-skill plan ran each
  skill twice (`tests/test_wave_scheduling.py`).
