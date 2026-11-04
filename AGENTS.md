# AGENTS.md — ARIN5102 Course Agent Project

This is a course project where students implement `StudentAgent` (an AI agent)
that is evaluated on three benchmark environments. The repo structure,
toolchain, and course contract are highly specific; read this before making
any changes.

## Project Structure

- `student_release/` — student-facing package (only this is submitted)
- `shared/` — schemas and policy docs (shared with staff tooling)
- `third_party/` — benchmark setup scripts and lock files (locked, not student code)
- `staff_side/` — placeholder; private repo contains staff tooling and hidden cases

**Git & Repository Setup:**
- The repository remote `origin` points to your private repo:
  `https://github.com/canpolatbulbul/2026-Fall-Agent-ARIN5102-Course-Project.git`
- Push your work to this private repo to avoid public solution leaks.
- Do not commit secrets (`env.student.local.sh`, API keys) or student metadata (`submission.json`).

**Editable by students:**
- `student_release/src/course_agent/student_agent.py`
- `student_release/src/course_agent/student_tools/` (new helper modules)
- `student_release/pyproject.toml` and `uv.lock` (add deps only)

**Locked (do not modify):**
- `student_release/public_runner/` and `public_runner/envs/`
- `student_release/src/course_agent/protocol.py`, `runtime.py`, `agent_process.py`
- All of `third_party/`

## Package Manager & Python

- Uses **`uv`** with a lockfile (`uv.lock`). Always use `uv sync --locked`.
- Student code: Python **3.11** (managed by `.python-version` in the root).
- Benchmark venvs: Python **3.12**, separate per benchmark under `third_party/`.
- To run any command: `uv run --locked <command>`, or use `.venv/bin/python` directly.
- Adding deps: edit `pyproject.toml`, then `uv add --locked <pkg>` to update `uv.lock`.
  Only PyPI deps with prebuilt wheels for linux x86_64 + macOS are allowed.
  No git/url/path deps, no source-only packages, no single artifact >50 MB,
  no total download >300 MB.

## Key Commands

| Action | Command |
|--------|---------|
| Install deps | `uv sync --locked` |
| Run all tests | `uv run --locked pytest -q -rs` (from `student_release/`) |
| Smoke test | `uv run --locked python smoke_local.py` |
| Simplified run (no API keys) | `uv run --locked run-tasks --backend simplified` |
| Official public run | `uv run --locked run-tasks --backend official --case tau2_airline:tau2_49` |
| Single benchmark | `uv run --locked run-tasks --backend simplified --benchmark mcpmark` |
| Resume a batch | `uv run --locked run-tasks --resume trajectories/<run_id>` |
| Parallel cases | `uv run --locked run-tasks --max-workers 4` |
| Build submission | `uv run --locked python build_submission.py --output submission.zip` |

**Env setup before official runs:**
```bash
cp env.student.sh.example env.student.local.sh
# edit COURSE_API_KEY
source env.student.local.sh
```

Grading runs with their own credentials; `env.student.local.sh` is gitignored
and rejected by the submission builder.

## StudentAgent Contract

The locked interface in `student_release/src/course_agent/student_agent.py`:

```python
class StudentAgent:
    def __init__(self, llm_client=None)   # may add state
    @property
    def llm_client(self) -> CourseLLMClient  # must remain accessible
    def respond(self, request) -> AgentResponse  # dispatches to solver
    def solve_mcpmark(self, request) -> AgentResponse
    def solve_tau2_airline(self, request) -> AgentResponse
    def solve_deepplanning(self, request) -> AgentResponse
```

**Section markers** in `student_agent.py` (do not remove):
`##### LOCKED COURSE INTERFACE`, `##### RESPONSE SCHEMA REQUIREMENTS`,
`##### OPTIONAL COURSE HELPERS`, `##### STUDENT-DESIGNED BENCHMARK LOGIC`,
`##### ADD YOUR OWN HELPERS`

Return values: `AgentResponse.tool_call(...)`, `AgentResponse.tool_calls(...)`,
`AgentResponse.final(...)`. The response is validated before the runner
processes it.

**Tool calls are executed sequentially, never in parallel.** Multiple tools
in one response are run one at a time; plan for this in agent design.

## LLM Client

- **Only model allowed:** `gpt-5-mini` via HKUST GenAI.
- Use the provided `CourseLLMClient` (OpenAI SDK wrapper). Do not bypass it.
- **Always pass `limits=request.limits`** to every `self.llm_client.chat(...)` call.
  This enforces per-case timeouts, token limits, and retry budgets.
- The client is **synchronous**. SDK retries are disabled; the client manages its
  own retry loop up to `llm_max_attempts` (default 2).
- Output tokens use `max_completion_tokens` (includes reasoning tokens).
- Direct API calls that bypass `CourseLLMClient` won't be tracked in trajectories.

## Tool Call Policy (Critical)

Per-benchmark limits enforced by the runner:

| Benchmark | Max calls/turn | Max mutating/turn | Terminal tool | Mutating tools |
|-----------|:---:|:---:|:---:|---|
| mcpmark | 4 | 1 | — | Anything not starting with `read/list/search/query/select/get` etc., `readOnlyHint` helps |
| tau2_airline | 3 | 1 | `respond_to_user` | book, cancel, update_*, send_certificate, respond_to_user |
| deepplanning | 5 | 3 | `submit_plan` | add/delete_product/coupon, submit_plan |

- **Read before mutate**: All read-only calls must precede any mutating call.
- **Terminal tools cannot be combined** with any other call in the same response.
- If a mutating tool fails, remaining calls in that response are skipped.
- Max tool calls per turn can be found in `task_configs/public_cases.json`
  under `limits.benchmarks.*`.

## Tau2-Specific

- `request.observation["domain_policy"]` contains the full airline policy
  **every turn** — always include it in LLM context.
- Use `context_id` to maintain conversation state across turns.
- Tau2 user simulator is a separate model (`openai/gpt-4o-mini`) that shares
  `COURSE_API_KEY`. It is not counted as student-agent LLM usage.
- Tau2 runs as a subprocess. Timeouts and errors produce partial trajectories;
  `request_interrupted` marks unanswered requests.

## MCPMark-Specific

- Course subset: `filesystem` + `postgres` only. No Notion/GitHub/Playwright.
- Requires Docker/Podman + Postgres 17 client tools + project-local Node.js.
- An internal runner lock serializes MCPMark cases (only one at a time).
- Filesystem cases use a local MCP server (`@modelcontextprotocol/server-filesystem`).
- Postgres cases use `postgres-mcp==0.3.0+course1` in a separate venv.

## DeepPlanning-Specific

- Official tasks expose shopping tools (search, add/delete cart, add/delete coupon).
- Simplified smoke backend may only expose `submit_plan`.
- Uses Qwen-Agent checkout with pinned Shopping data from HuggingFace.
- No Docker/Node/Postgres required.

## Scoring

- **Binary**: 1.0 for full completion, 0.0 otherwise (hidden cases only).
- Baseline anchors: MCPMark 31.25%, tau2 41.67%, DeepPlanning 16.67%.
- Score below baseline → linear 0-60 scaling. Score above → 60-100 scaling.
- Final score = weighted average across benchmarks by hidden-case count.
- `analysis_score` is diagnostic only (benchmark-native scale), not used for grading.

## Submission

```bash
cp submission.example.json submission.json
# edit student_id, student_name, email
uv run --locked python build_submission.py --output submission.zip
```

Build checks: valid submission.json, dependency policy, smoke_local.py pass,
simplified run pass. Review `submission_build_log.json` before submitting.

## Automated Test Suite

- Tests are under `student_release/tests/`. No API keys needed.
- Benchmark-environment-dependent tests are auto-skipped when not installed.
- Test layout: protocol tests, runtime/LLM tests, agent contract tests,
  CLI/runner integration tests, submission builder tests, smoke tests, etc.
- Run: `uv run --locked pytest -q -rs` from `student_release/`.
- Simplify with: `uv run --locked pytest -q -rs --ignore=tests/test_public_runner_backends.py`
  to skip official-backend-heavy tests.

## Chokepoints / Gotchas

- **Source `env.student.local.sh` in every new terminal** — the runner reads
  `COURSE_API_KEY`, `OPENAI_BASE_URL`, and model names from environment
  variables. Without sourcing, official runs will fail with credential errors.
  Do not commit this file; the submission builder rejects it.
- **Always pass `limits=request.limits`** to `llm_client.chat()` — the defaults
  (300s timeout, 64k input, 4096 output, 2 attempts) come from env vars, not
  the task config.
- **Tool calls are sequential**, not parallel — the runner executes each tool in
  the list one at a time and returns all results in the next observation.
- **`respond_to_user` is terminal for tau2** — cannot be combined with other tools.
  Once called, the turn ends.
- **MCPMark has no terminal tool** — the agent returns `final` when done.
- **Benchmark env paths** are auto-discovered from `third_party/` — set
  `TAU2_REPO` / `DEEPPLANNING_REPO` / `MCPMARK_REPO` only for self-managed checkouts.
- **Concurrent MCPMark is impossible** — an internal file lock serializes all
  MCPMark official cases. Other benchmarks can still run concurrently.
- **Old Azure env vars (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`) are
  stale** — the migrated client uses `OPENAI_BASE_URL` and `COURSE_API_KEY`.
- **The submission builder does a simplified run** as a packaging check — this
  is not representative of actual benchmark performance.
