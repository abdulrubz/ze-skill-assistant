# Codebase walkthrough

A map of `ze-ai`, folder by folder, from the big picture down to each file's
job. Read this first; then open the files in the order suggested at the end.

## The one-sentence version

Someone asks a question in plain English. `ze-ai` works out which "expert"
(skill) should answer it. That expert calls your company's REST APIs to fetch
real data. The service returns an answer where every number is tied back to
the exact API call it came from.

That's the whole job. Every folder exists to support one part of that sentence.

## Three projects, one system

| Project | Role | Analogy |
|---|---|---|
| `ze-mock-services` | Pretend microservices (customers, orders) | The kitchens that hold the data |
| `ze-api-specs` | One git repo holding every service's Swagger file | The catalogue describing what each kitchen can do |
| `ze-ai` | The GenAI service | The waiter who takes your question, knows which kitchen to ask, and brings back a checked answer |

The rest of this document is about `ze-ai`.

## The top level of `ze-ai`

```
ze-ai/
├── app/          ← the ENGINE. Generic. You should rarely edit this.
├── domain/       ← YOUR CONTENT. Skills, agents, tools.yaml, vocabulary, eval questions.
├── specs/        ← copies of each service's Swagger file (pulled from ze-api-specs)
├── generated/    ← tool definitions machine-generated from specs/. Never hand-edited.
├── scripts/      ← command-line helpers for humans (ask a question, sync specs, run evals...)
├── tests/        ← automated checks
├── docs/         ← this walkthrough
├── .github/      ← CI: run tests on every push; auto-PR when specs change
└── config files  ← pyproject.toml, .env.example, Dockerfile, specs.lock.json, uv.lock
```

**The most important idea:** `app/` vs. `domain/`.

- `app/` is like a car engine. It knows how to plan, call tools, check answers
  and talk to MLflow, but it knows nothing about customers or orders.
- `domain/` is the steering, seats and paint: what *this particular* project
  is about.

When you adapt this for a new client, you change `domain/`, `specs/` and
`.env`, and leave `app/` alone.

---

## `app/`: the engine, folder by folder

The folders are described in the order a question travels through them.

### Entry points: `app/app.py`, `app/config.py`, `app/paths.py`, `app/deps.py`

| File | What it does |
|---|---|
| `app.py` | Starts the web server (FastAPI). At startup it builds everything once: the tools, the skills, the graph, the database connections. If anything is mis-wired, it refuses to start. |
| `config.py` | Reads settings from environment variables / `.env`: MLflow URL, Langfuse keys, service URLs, and so on. One class, `Settings`, holds them all. |
| `paths.py` | Just a list of folder locations ("skills live in `domain/skills/`"), so no other file hard-codes paths. |
| `deps.py` | Defines `Deps`: a small bag of things every step of a run might need. Who is asking (`Caller`), their bearer token, today's date, and a way to send live progress events. One is created per question. |

### `app/api/`: the HTTP front door

This is how the outside world talks to the service.

| File | What it does |
|---|---|
| `routes_ask.py` | The URLs. `POST /ask` is the main one: it checks for a bearer token (401 without one) and starts a run. Others: `/health`, `/agents`, `/skills`, `/tools`, `/tools/mcp` (the full catalogue), `/costs`, `/runs/{id}`. |
| `adapter.py` | Turns the internal progress of a run into a stream of events the client can read: "plan ready", "tool called", "answer ready", and so on. |
| `sse.py`, `events.py` | The format of that stream. SSE (server-sent events) is a simple way for a server to push a series of messages over one HTTP response. |
| `schemas.py` | Describes the shape of every response, so the auto-generated API docs at `/docs` are accurate. |

### `app/graph/`: the pipeline (the "brain's to-do list")

This uses **LangGraph**, a library where you draw a flowchart in code. Each box
is a **node** (a Python function), and arrows decide what runs next.

```
admission → resolve → plan → run_skill (in parallel) → schedule → join → validate → respond
```

| File | What it does |
|---|---|
| `build.py` | Draws the flowchart: which nodes exist and how they connect. Also holds the `schedule` node, which decides the next batch of skills only after every parallel skill has finished. |
| `state.py` | The "clipboard" passed between nodes: the question, the plan, the skill results, retry count, the final answer. |
| `routing.py` | The decision logic on the arrows. For example: "after planning, fan out one `run_skill` per step", or "validation failed, re-run only the step that failed". |
| `nodes/admission.py` | Gatekeeper: is this agent allowed to answer at all? |
| `nodes/resolve.py` | Pulls obvious things out of the question with no AI: dates ("last quarter"), IDs like `CUST-1004`, currency. |
| `nodes/plan.py` | **AI call #1.** The planner reads the question plus a description of every skill and picks which skill(s) should answer. |
| `nodes/run_skill.py` | Runs one chosen skill. Several can run at the same time. |
| `nodes/join.py` | Collects all skill results and spots contradictions between them. |
| `nodes/validate.py` | Runs the citation checks (see `validation/` below). If something fails, the run loops back and retries. |
| `nodes/respond.py` | Builds the final answer, or a polite refusal ("ask the General agent instead"). |

### `app/skills/`: how a skill works

A **skill** is an "expert" for one area, for example `order_analysis`. It is
defined by a YAML file in `domain/`, and this folder turns that YAML into
something runnable.

| File | What it does |
|---|---|
| `manifest.py` | Reads a skill's YAML and checks its format. |
| `registry.py` | Loads every skill and runs the **boot self-test**: does every tool it grants exist in `tools.yaml`? Does every entity it claims exist in the vocabulary? It also builds the short skill descriptions the planner sees. |
| `factory.py` | **The heart of a skill.** It turns a manifest into a little loop: the AI thinks, calls a tool, reads the result, thinks again (up to `max_tool_steps`), then a second AI call extracts the findings into structured, cited "facts". This loop pattern is called **ReAct** (Reason + Act). |
| `capability.py` | Classifies a question with no AI: what is it about (customer? order?) and what does it ask for (rank? lookup? trend?). Used to shortlist skills before the planner sees them. |
| `slots.py` | The named pieces of information skills can need or produce, e.g. `customer_id` or `top_customers`. |
| `envelope.py` | The data shapes. A `Fact` is one value plus its citation and the tool call it came from. A `SkillResult` is a narrative plus its facts. |
| `answer.py`, `output.py` | Turn facts into the final answer: text blocks, metric blocks, tables. Numbers come from facts, never from the AI's prose. |

### `app/tools/`: how APIs become tools

| File | What it does |
|---|---|
| `openapi/generate.py` | Reads each Swagger file in `specs/` and writes a simplified description of every operation into `generated/`. Names are prefixed with the service (`orders.getOrder`) so two teams' APIs can't clash. |
| `openapi/config.py` | Reads **`domain/tools.yaml`**: the friendly tool names and descriptions, each mapped to one API operation. |
| `openapi/drift.py` | Checks that `tools.yaml` still matches the specs: operation still exists, is documented, isn't a write. This is what CI runs. |
| `openapi/server.py` | Builds the actual callable tools, one HTTP client per service with its base URL, for only the operations listed in `tools.yaml`. |
| `registry.py` | The list of tools skills are allowed to use. Anything not in here can't be called. |
| `authz.py` | **The single chokepoint every tool call passes through.** It re-checks permission, puts the user's token in place, times the call, emits "tool called" events, and turns errors into messages the AI can learn from. |
| `auth.py` | Holds the user's bearer token for the duration of a call and attaches it to every outgoing HTTP request, without ever putting it where the AI or the logs can see it. |

### `app/validation/`: the fact-checker

| File | What it does |
|---|---|
| `groundedness.py` | Every `[f_01]` citation must point to a real fact; every fact must point to a real tool call; no number in the text may be uncited. |
| `invariants.py` | Recomputes derived values. If the AI says "A + B = C", it checks that's arithmetically true. |

No AI is involved here, on purpose: the checker can't be persuaded.

### `app/llm.py`: talking to the AI model

Every AI call goes through this file, and it only talks to the **MLflow
gateway** by endpoint name (`ze-ai`, `ze-ai-judge`). The model behind an
endpoint (e.g. a Databricks serving endpoint) is configured in MLflow, not
here. The file contains `MlflowChat`, a small fix that preserves Gemini's
"thought signatures" across tool-calling turns, and a rate limiter.

### `app/obs/`: observability (watching what happened)

| File | What it does |
|---|---|
| `tracing.py` | Sends every run to **Langfuse**, so you can see each step, prompt, token count and cost in its UI. |
| `usage.py`, `pricing.yaml` | Count tokens and convert them to dollars, using the per-model prices in the YAML. |
| `ledger.py` | Saves a cost/quality record of every run to a local SQLite file; served at `/costs`. |
| `emit.py` | Emits the "model used X tokens" event after each AI call. |

### `app/prompt_store.py` + `app/prompts/`

`app/prompts/` holds the engine's own instructions to the AI: how to plan
(`planner.md`), how to extract facts (`extraction.md`), how to write the
lead-in sentence (`lead_in.md`), and how the eval judge grades answers
(`eval_judge.md`).

`prompt_store.py` decides where each prompt comes from. If
`LANGFUSE_PROMPTS=true`, it fetches the live version from Langfuse, so
business users can edit prompts in the Langfuse UI without a deploy.
Otherwise, or if Langfuse is down, it uses the file.

### `app/vocabulary.py`

Loads `domain/vocabulary.yaml`, the domain's words (see below).

---

## `domain/`: your project's content

This is the part you'll edit most, and it's all YAML and Markdown, no Python.

```
domain/
├── tools.yaml          ← which APIs the AI may use, with friendly names and descriptions
├── vocabulary.yaml     ← the domain's nouns, question types, ID patterns, slot names
├── skills/
│   ├── customer_profile.yaml   + prompts/customer_profile.md
│   ├── order_analysis.yaml     + prompts/order_analysis.md
│   ├── prose_synthesis.yaml    ← "output" skill: formats the answer as prose
│   └── table_composition.yaml  ← "output" skill: formats the answer as a table
├── agents/
│   ├── general.yaml            ← can use every skill (the default)
│   └── customer_directory.yaml ← deliberately narrow: customer lookups only
└── evals/
    ├── answer_bank.yaml        ← test questions with verified correct answers
    ├── routing_bank.yaml       ← test questions with the skill that should handle each
    └── eval_targets.yaml       ← the pass/fail bar for eval scores
```

| Piece | What it does |
|---|---|
| **`tools.yaml`** | Developers give each API a better name and a description the AI reads, mapped by `<service>.<operationId>` (e.g. `orders.getTopCustomers`). Only tools listed here can be granted to skills or called at all; the service refuses to start otherwise. |
| **A skill** | Two files. The YAML says what the skill answers (example questions, entities, what it cannot do, which tools it may use). The `.md` is its playbook: practical advice to the AI, like "for rankings, call the ranking tool once per currency". |
| **An agent** | A permission scope: a named list of skills. Every agent runs the exact same pipeline; it just sees fewer or more skills. |
| **`vocabulary.yaml`** | Teaches the engine your domain's words. "customer, client, company, account" all mean the entity *customer*. IDs look like `CUST-\d{4}`. Supported question types: rank, list, trend, and so on. |
| **`evals/`** | The questions used to measure quality, with their verified answers and the scores a run must reach. |

---

## `specs/`, `generated/`, `specs.lock.json`: what the APIs look like

```
ze-api-specs (git) ──sync──▶ specs/orders.openapi.json         (exact copy of the Swagger)
                             specs.lock.json                    (which git commit it came from)
                                   │ generate
                                   ▼
                             generated/orders.tools.json        (every operation, simplified)
                                   │ filtered by
                                   ▼
                             domain/tools.yaml                  (only the ones you allow)
```

These are committed to git, so the service never needs to reach the spec repo
while running. Updating them is a deliberate, reviewed step:
`python scripts/tool_spec.py sync` locally, or the auto-PR from
`.github/workflows/spec-sync.yml`.

---

## `scripts/`: commands for humans

| Script | When you'd use it |
|---|---|
| `doctor.py` | **First thing on a new machine.** Checks every URL, key and endpoint and tells you how to fix each failure. |
| `ask.py` | Ask one question from the terminal and watch it think, step by step. |
| `tool_spec.py` | `sync` specs from git, `list` available operations, `check` that `tools.yaml` still fits. |
| `prompts.py` | Copy prompts between the repo and Langfuse (`push`, `pull`, `status`). |
| `eval_coverage.py` | Quick test: does the planner pick the right skill for each routing question? |
| `langfuse_eval.py` | Full scored evaluation in Langfuse, with an AI judge. |
| `eval_cost.py` | Measure cost and quality end to end, and compare before vs. after a change. |
| `mlflow_gateway.py` | Inspect MLflow endpoints, set budgets, or create local dev endpoints. |
| `agent_graph.py` | Generate an HTML diagram of agents, skills and tools. |

## `tests/`

The automated checks, run with `pytest -q`. A few worth knowing:

- `test_generated_tools.py`: each tool calls the right service, carrying the user's token.
- `test_wave_scheduling.py`: each skill runs exactly once.
- `test_tool_drift.py`: `tools.yaml` fits the specs.
- `test_domain.py`: guards for this project's specific skills. Replace it when you replace the dummy domain.

`conftest.py` makes sure tests never read your personal `.env`.

## The config files

| File | Purpose |
|---|---|
| `pyproject.toml` | The project's Python dependencies (LangGraph, FastAPI, langchain-openai, fastmcp, langfuse...). |
| `uv.lock` | The exact pinned versions, so every machine installs precisely what was tested. |
| `.env.example` | Every setting with comments. Copy it to `.env` and fill it in. |
| `Dockerfile` | Packages the service as a container for deployment. |
| `.github/workflows/ci.yml` | Runs the drift check and tests on every push. |
| `.github/workflows/spec-sync.yml` | When a spec changes, re-syncs and opens a PR. |

---

## Putting it together: one question's journey

*"Who are our top 3 customers by spend in USD?"*

1. **`api/routes_ask.py`** receives it, checks the bearer token, and creates `Deps`.
2. **`graph/nodes/resolve.py`** spots "USD" → `currency=USD`.
3. **`skills/capability.py`** sees *customer × rank* and shortlists `order_analysis`.
4. **`graph/nodes/plan.py`** (AI, via `llm.py` and MLflow) picks `order_analysis` and the table output skill.
5. **`skills/factory.py`** runs the skill's loop:
   - The AI calls `rank_customers_by_spend` (from `tools.yaml` → `orders.getTopCustomers`).
   - Then it calls `get_customer` three times for the names.
   - Each call passes through **`tools/authz.py`** and **`tools/auth.py`**, which attach the user's token.
6. The extraction step turns the transcript into cited facts.
7. **`validation/`** checks every citation and number.
8. **`skills/output.py`** builds the answer, and **`api/adapter.py`** streams it back.
9. Throughout, **`obs/tracing.py`** has been recording everything to Langfuse.

---

## Suggested reading order for the deep dive

Each step builds on the one before:

1. **`domain/`**: the parts you'll touch most, and the vocabulary everything else refers to.
2. **The tools layer**: `tools.yaml` → `generate.py` → `server.py` → `authz.py` / `auth.py`.
3. **One skill end to end**: `manifest.py` → `factory.py` → `envelope.py`.
4. **The pipeline**: `build.py`, then each node in order.
5. **Validation and output**: how citations are checked and the answer is assembled.
6. **The model plane, Langfuse and the scripts.**
