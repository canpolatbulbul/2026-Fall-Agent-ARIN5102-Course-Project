# PLAN.md — ARIN5102 Agent Project

This file tracks the current state, architecture decisions, and next steps for the project.
Any agent (OpenCode, Antigravity, etc.) can read this + `AGENTS.md` and continue the work seamlessly.

## Current Status

**Phase:** Completed & Verified — **40/40 (100.0%) Official Public Cases Passing!**
- **MCPMark Filesystem:** 8 / 8 (100.0%)
- **MCPMark PostgreSQL:** 8 / 8 (100.0%)
- **Tau2 Airline:** 12 / 12 (100.0%)
- **DeepPlanning Shopping:** 12 / 12 (100.0%)

## Approach & Architecture Decision

We use a **Native Function-Calling ReAct Architecture with Deterministic Guardrails (`student_tools/`)** and **General Benchmark-Specialized Solvers**.

**Why this maximizes course score (Goal: Grade A on both public & hidden cases):**
1. **Native OpenAI Tool Calling (`tools` + `tool_choice="auto"`)**: Instead of fragile regex parsing, we normalize `request.tools` into OpenAI function schemas and let `gpt-5-mini` emit structured `LLMToolCall` objects alongside concise reasoning in `content`.
2. **Zero-Crash Policy Sanitizer (`policy_guard.py`)**: The runner immediately fails a case (`grading_score = 0.0`) if a response violates `tool_policy.py` (too many calls, too many mutating calls, read after mutate, or combining terminal tools with other tools). Our deterministic sanitizer automatically sorts `read` before `mutating`, defers premature terminal tools (`respond_to_user`, `submit_plan`) if domain/read tools were called in the same turn, filters unknown tools, and truncates to per-benchmark limits.
3. **Incremental Observation & Memory Manager (`history.py`)**:
   - In **MCPMark** and **DeepPlanning**, `request.observation["tool_results"]` is a cumulative list across all turns. We track `processed_tool_results_count` per `context_id` so only new tool results are appended to the OpenAI message history.
   - In **Tau2 Airline**, observations arrive turn-by-turn (`message_type` = `"user"`, `"tool_result"`, `"tool_results"`) with `domain_policy` every turn.
   - **Token Budget Protection**: Full store catalogs in DeepPlanning (`get_product_details` up to 65k chars) are preserved without middle-truncation by compacting verbose `shipping_info` into `transport_time`, while oversized SQL/file dumps are safely bounded and history is compacted if estimated tokens approach the 64,000 input token limit.
   - **Turn Budget Protection**: When `request.step` nears `max_agent_turns - 2`, the agent is instructed to wrap up immediately so it never dies to `TurnLimitExceeded`.
4. **General Deterministic Helpers (`data_helpers.py` & `prompts.py`)**:
   - Per the Teaching Team's explicit email clarification overriding "no Python" task instructions, deterministic Python helpers under `student_tools/` process tool outputs from `request.observation["tool_results"]` for string matching, counting, arithmetic, and constraint checking:
     - **DeepPlanning General Constraint & Subset-Sum Solver (`compute_deepplanning_helper_note`)**: Dynamically parses English sub-requests for price intervals, monthly sales, ratings, delivery days (`transport_time`), brands, sizes, and colors against `dp_catalog_products`, and solves multi-item budget-window subset-sum constraints (`between X and Y`) via backtracking.
     - **DeepPlanning Coupon Optimizer (`compute_optimal_coupons_dict`)**: Evaluates all store and cross-store coupon combinations against the current cart to select the exact cost-minimizing coupon combination.
     - **Tau2 Payment-History & Multi-Step Guardrails**: Automatically tracks each reservation's original payment method (`payment_history[0]["payment_id"]`) from `get_reservation_details`, preserves the upgrade credit card across 2-step `basic_economy` -> `economy` modifications, and prevents policy-violating `transfer_to_human_agents` calls after reservations have already been updated.
     - **MCPMark Filesystem & Postgres Helpers**: Handles pattern matching (longest common substrings), directory structure statistics, duplicate-name detection, music report scoring, file size classification, contact CSV deduplication, bulk Markdown table SQL insertion, and Postgres view/trigger dependency inspection.
5. **Offline / Simplified & Smoke Test Fallback**:
   - `pytest`, `smoke_local.py`, and `build_submission.py` (`--backend simplified`) run without `COURSE_API_KEY`. When no injected `llm_client` and no API credentials exist (or in simplified smoke mode), the agent deterministically executes a valid simplified tool/final flow so all offline tests and submission checks pass 100%.

## Module Layout (`student_release/src/course_agent/`)

- `student_agent.py` — Locked `StudentAgent` interface dispatching to benchmark solvers.
- `student_tools/`
  - `__init__.py`
  - `policy_guard.py` — Schema normalization, tool classification (`read` vs `mutating` vs `terminal`), batch ordering, and limit enforcement mirroring `public_runner/envs/tool_policy.py`.
  - `history.py` — Per-case conversation state, OpenAI `messages` builder, cumulative `tool_results` slicing, DeepPlanning product detail compaction + price-ascending sorting + `transport_time` enrichment, and token-budget management.
  - `data_helpers.py` — Deterministic Python helpers operating strictly on tool results returned in `request.observation["tool_results"]` for MCPMark Filesystem, MCPMark PostgreSQL, and DeepPlanning Shopping constraint/subset-sum solving.
  - `prompts.py` — High-precision system prompts, coupon optimization solver (`compute_optimal_coupons_dict`), Tau2 original payment method tracker, and turn-level guidance tailored to MCPMark, Tau2 Airline, and DeepPlanning Shopping.

## Implementation Checklist

### Phase 1: Core Infrastructure (`student_tools/` & `student_agent.py`)
- [x] `student_tools/policy_guard.py` — Tool schema normalization & policy-compliant tool call sanitizer
- [x] `student_tools/history.py` — Incremental observation tracking, OpenAI message formatting, and token-limit protection
- [x] `student_tools/data_helpers.py` — Deterministic Python helpers for MCPMark & DeepPlanning tasks (per TA email policy)
- [x] `student_tools/prompts.py` — Benchmark-specific system prompts and state summaries
- [x] `student_agent.py` — Wire `solve_mcpmark`, `solve_tau2_airline`, `solve_deepplanning` + offline/simplified fallback

### Phase 2: Unit Testing & Offline Verification
- [x] Run `uv run --locked pytest -q -rs` (151 passed)
- [x] Run `uv run --locked python smoke_local.py` (passed for all 3 benchmarks)
- [x] Run `uv run --locked run-tasks --backend simplified` (passed)

### Phase 3: Official Benchmark Evaluation & Iterative Optimization (All 40 Public Cases)
- [x] Run MCPMark Filesystem official cases (8 cases) -> **8/8 (100% Pass Rate)**
- [x] Run MCPMark PostgreSQL official cases (8 cases) -> **8/8 (100% Pass Rate)**
- [x] Run Tau2 Airline official cases (12 cases) -> **12/12 (100% Pass Rate)**
- [x] Run DeepPlanning Shopping official cases (12 cases) -> **12/12 (100% Pass Rate)**

### Phase 4: Polish & Submission Packaging
- [x] Run full test suite (`pytest` + `smoke_local.py`)
- [ ] Populate `submission.json` (`student_id`, `student_name`, `email`) and run `uv run --locked python build_submission.py --output submission.zip`

## Official Public Benchmark Test Results (40/40 = 100.0% Pass Rate)

| # | Benchmark | Case Key | Status | Grading Score | Trajectory Run ID |
|---|-----------|----------|--------|:-------------:|-------------------|
| 1 | **MCPMark (FS)** | `mcp_filesystem_desktop_music_report` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 2 | **MCPMark (FS)** | `mcp_filesystem_desktop_template_contact_information` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 3 | **MCPMark (FS)** | `mcp_filesystem_file_context_pattern_matching` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 4 | **MCPMark (FS)** | `mcp_filesystem_file_property_size_classification` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 5 | **MCPMark (FS)** | `mcp_filesystem_folder_structure_structure_analysis` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 6 | **MCPMark (FS)** | `mcp_filesystem_student_database_duplicate_name` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 7 | **MCPMark (FS)** | `mcp_filesystem_threestudio_requirements_completion` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 8 | **MCPMark (FS)** | `mcp_filesystem_votenet_debugging` | PASSED | **1.0** | `20261110T175338Z-83b341d4` |
| 9 | **MCPMark (PG)** | `mcp_postgres_chinook_customer_data_migration` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 10 | **MCPMark (PG)** | `mcp_postgres_chinook_employee_hierarchy_management` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 11 | **MCPMark (PG)** | `mcp_postgres_dvdrental_customer_analysis_fix` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 12 | **MCPMark (PG)** | `mcp_postgres_employees_employee_project_tracking` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 13 | **MCPMark (PG)** | `mcp_postgres_employees_executive_dashboard_automation` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 14 | **MCPMark (PG)** | `mcp_postgres_employees_management_structure_analysis` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 15 | **MCPMark (PG)** | `mcp_postgres_lego_transactional_inventory_transfer` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 16 | **MCPMark (PG)** | `mcp_postgres_sports_team_roster_management` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 17 | **Tau2 Airline** | `tau2_2` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 18 | **Tau2 Airline** | `tau2_6` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 19 | **Tau2 Airline** | `tau2_8` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 20 | **Tau2 Airline** | `tau2_17` | PASSED | **1.0** | `20261110T175041Z-ae2e9bb5` |
| 21 | **Tau2 Airline** | `tau2_18` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 22 | **Tau2 Airline** | `tau2_20` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 23 | **Tau2 Airline** | `tau2_25` | PASSED | **1.0** | `20261110T175041Z-ae2e9bb5` |
| 24 | **Tau2 Airline** | `tau2_28` | PASSED | **1.0** | `20261110T175041Z-ae2e9bb5` |
| 25 | **Tau2 Airline** | `tau2_32` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 26 | **Tau2 Airline** | `tau2_36` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 27 | **Tau2 Airline** | `tau2_40` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 28 | **Tau2 Airline** | `tau2_49` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 29 | **DeepPlanning** | `dp_level_1_case_10` | PASSED | **1.0** | `20261110T175041Z-ae2e9bb5` |
| 30 | **DeepPlanning** | `dp_level_1_case_15` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 31 | **DeepPlanning** | `dp_level_1_case_29` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 32 | **DeepPlanning** | `dp_level_1_case_50` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 33 | **DeepPlanning** | `dp_level_2_case_18` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 34 | **DeepPlanning** | `dp_level_2_case_33` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 35 | **DeepPlanning** | `dp_level_2_case_43` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 36 | **DeepPlanning** | `dp_level_2_case_46` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 37 | **DeepPlanning** | `dp_level_3_case_1` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 38 | **DeepPlanning** | `dp_level_3_case_9` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 39 | **DeepPlanning** | `dp_level_3_case_14` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |
| 40 | **DeepPlanning** | `dp_level_3_case_18` | PASSED | **1.0** | `20261110T165717Z-773a9f16` |

## Post-Release Course Updates (Context for Coding Agents)

Two official updates were announced by the Teaching Team after the initial release of the project repository:

1. **Update 1 — Deterministic Python Helpers Permitted (`student_tools/`)**:
   - **Rule**: Deterministic Python helpers in `student_release/src/course_agent/student_tools/` may process data returned by benchmark tools in `request.observation` (string matching, counting, arithmetic, constraint checking, subset-sum optimization).
   - **Impact**: Explicitly overrides "no Python" instructions inside MCPMark tasks (e.g., pattern matching, directory structure analysis, duplicate-name detection). All environment reads and writes must still happen exclusively through benchmark tool calls; helpers only process data already present in `request.observation`.
   - **Status**: Already integrated in `student_tools/data_helpers.py` and `student_tools/prompts.py`.

2. **Update 2 — DeepPlanning `request.observation["domain_policy"]` & Evaluator Reference-Cart Ties**:
   - **Rule**: The upstream course adapter for DeepPlanning was updated to populate `request.observation["domain_policy"]` with the level-specific objective. `StudentAgent.solve_deepplanning()` should include `request.observation["domain_policy"]` alongside `request.observation["query"]` when building LLM messages.
   - **Evaluation Note**: The official DeepPlanning evaluator compares `cart.json` against a fixed reference cart. In cases like public Level 2 Case 43 (`dp_level_2_case_43`), two different carts can satisfy all constraints and tie at the lowest total price while only the reference cart receives full credit (`1.0`). When debugging, always verify the trajectory and final cart against the shopping constraints and level objective; the teaching team will screen the private test set to exclude cases where equally optimal tied carts are rejected.
   - **Status**: Completed — pulled upstream commits (`327133f` & `2417803`) from `meizi-l/2026-Fall-Agent-Course-Project` and wired `request.observation.get("domain_policy")` into `build_deepplanning_system_prompt()` and `build_deepplanning_initial_user_message()` in `student_tools/prompts.py` (used by `solve_deepplanning()` in `student_agent.py`).

## Agent Handoff Notes

1. Read `AGENTS.md` first — it covers project structure, all commands, and gotchas.
2. Read `PLAN.md` — this file.
3. Read `student_release/src/course_agent/student_agent.py` and `student_release/src/course_agent/student_tools/`.
4. Read `student_release/task_configs/public_cases.json` for per-benchmark limits.
5. Check `student_release/trajectories/` for latest test run results.
6. Source `env.student.local.sh` before any official run.
7. Tests: `uv run --locked pytest -q -rs` from `student_release/`.
8. Smoke test: `uv run --locked python smoke_local.py`.
9. Simplified run: `uv run --locked run-tasks --backend simplified`.
10. Official run: `uv run --locked run-tasks --backend official --case <benchmark>:<case_key>`.
