# PLAN.md — ARIN5102 Agent Project

This file tracks the current state, decisions, and next steps for the project.
Any agent (OpenCode, Antigravity, etc.) can read this + `AGENTS.md` and
continue the work seamlessly.

## Current Status

**Phase:** Implementation — Core Agent Loop

## Approach Decision

We chose **Approach 2: ReAct-Style** (structured reasoning before each action)
as the primary design, with a plan to evolve per-benchmark modules
(Approach 3) only where the simple loop falls short.

**Why:**
- Well-documented pattern that works across all three benchmarks
- Relatively little code, easy to debug via trajectory inspection
- Natural error recovery ("thought: that result was empty, try something else")
- Easy to add planning later for DeepPlanning if needed

## What's Implemented

- Environment setup (all 3 benchmarks installed and verified)
- AGENTS.md with full project conventions
- This PLAN.md

*Nothing yet in student_agent.py beyond the starter template.*

## What Needs to Be Built (Order)

### Phase 1: Core ReAct Loop (shared across benchmarks)

- [ ] `_build_system_prompt()` — per-benchmark system prompt generator
- [ ] `_react_step()` — single ReAct turn: format prompt → call LLM → parse response → return tool call or final
- [ ] `_format_observation()` — convert request.observation + tool_results into structured text for the LLM
- [ ] Conversation memory management via `context_id` and `_case_state()`

### Phase 2: Benchmark Solvers

- [ ] `solve_mcpmark()` — filesystem and postgres tasks
- [ ] `solve_tau2_airline()` — conversational airline agent
- [ ] `solve_deepplanning()` — shopping planning agent

### Phase 3: Testing & Iteration

- [ ] Run simplified backend
- [ ] Run MCPMark official cases
- [ ] Inspect trajectories, fix failures
- [ ] Run tau2 official cases
- [ ] Inspect trajectories, fix failures
- [ ] Run DeepPlanning official cases
- [ ] Inspect trajectories, fix failures

### Phase 4: Polish & Submit

- [ ] Run full test suite
- [ ] Run simplified full suite
- [ ] Build submission.zip
- [ ] Verify submission_build_log.json

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Agent pattern | ReAct (Thought/Action/Observation) | Simplest reliable pattern, easy to debug |
| LLM calling | Always pass `limits=request.limits` | Enforced by course contract |
| Tool calling | Single tool per turn preferred | Simpler to debug; multi-tool only when clearly beneficial |
| State management | Dict keyed by `context_id` | Already provided by template's `_case_state()` |
| Per-benchmark prompts | One system prompt per benchmark | Each benchmark has very different tool sets and patterns |

## Test Results

*To be filled as we run tests.*

| Run | Date | Backend | Cases | Pass | Fail | Notes |
|-----|------|---------|-------|------|------|-------|
| — | — | — | — | — | — | — |

## Agent Handoff Notes

If switching to a new agent (Antigravity, etc.):

1. Read `AGENTS.md` first — it covers project structure, all commands, and gotchas.
2. Read `PLAN.md` — this file, you're reading it now.
3. Read `student_release/src/course_agent/student_agent.py` — the current implementation.
4. Read `student_release/task_configs/public_cases.json` for per-benchmark limits.
5. Check `student_release/trajectories/` for latest test run results.
6. Continue from the "What Needs to Be Built" checklist above.
7. Source `env.student.local.sh` before any official run.
8. All benchmark environments are under `third_party/` and are pre-installed.
9. Tests: `uv run --locked pytest -q -rs` from `student_release/`.
10. Smoke test: `uv run --locked python smoke_local.py`.
11. Simplified run: `uv run --locked run-tasks --backend simplified`.
12. Official run: `uv run --locked run-tasks --backend official --case <benchmark>:<case_key>`.
