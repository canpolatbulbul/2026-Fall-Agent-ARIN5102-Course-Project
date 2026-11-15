# Design: Deterministic compute-and-emit path for MCPMark (Phase A)

Date: 2026-11-15
Status: REVISED after adversarial review — pending user approval

## Intent / why

The 52-task census (28 pass = 53.8%) shows the dominant failure is many *successful*
tool calls followed by wrong/incomplete **computation or content** (timeline 15/43 entries,
duplicates 14-vs-20, budget 9/16 lines). The model has the data and still miscomputes or
corrupts it. The course permits deterministic Python helpers that transform data already in
`request.observation`, provided all environment access still goes through `request.tools`
calls. This design computes the exact answer in Python from the raw (untruncated) tool
results and emits the tool calls directly.

## What the adversarial review changed

- **Episode termination was undefined and would have caused 0.0.** MCPMark has no terminal
  tool; the runner scores only on `response.kind == "final"` (or `TurnLimitExceeded` → 0).
  Yielding back to the LLM after deterministic writes lets the LLM re-mutate and overwrite
  correct artifacts (exactly the observed corruption). The solver must own the final.
- **The deterministic path must be sanitized.** `validate_tool_batch` runs before execution
  and a violation is a non-retryable runtime error → 0.0. The LLM path is protected by
  `sanitize_tool_calls`; the deterministic path must route through it too.
- **Do not build a registry for two solvers.** `get_pending_mcpmark_batch_reads` is already a
  compute-and-emit hook. Extend the existing mechanism with explicit branches; a registry is
  justified only once there are ~4+ solvers (which Phase B will have).
- **`time_classification` is far riskier than described** and is deferred: the vendored server
  requires the immediate parent dir to exist (its `mkdir recursive` does not help because
  `validatePath` rejects first) — the recorded failure was exactly that; `get_file_info`
  returns a timezone-dependent `Date.toString()`; the verifier is hardcoded to specific
  files/days. Treated as its own hardened unit later, not Phase A.
- **Recognition collisions are real:** `large_file.txt` appears in both `file_splitting` and
  `pattern_matching`. Recognition must be instruction-signature + structure, never `case_id`
  (hidden keys differ), and must be side-effect-free when it does not recognize.
- **The "free offline replay" harness does not exist yet** and `trajectories/` is gitignored;
  building the replay path is real cost, acknowledged below.
- **Phase A yield is small (~0.5–0.9 project points).** The points are in Phase B's general
  computation solvers. Recommendation: ship the framework + `file_splitting`, then go
  straight to the three general solvers.

## Goals

- For a recognized task, compute the exact artifacts from raw tool results and emit them, with
  the solver owning the final so the LLM can never corrupt the result.
- Never bypass benchmark tools; only parse tool outputs and emit `request.tools` calls.
- No regression: conservative recognition; if unrecognized, behavior is byte-identical to
  today. A solver may never make a passing case fail.
- Each solver provable offline via replay through the task's own `verify.py`.

## Non-goals

- Not touching tau2_airline/deepplanning; no direct filesystem/DB access.
- Not addressable by solvers: the 7 census tasks that die to the 300 s timeout (LLM-latency
  bound). Left alone.
- No NLP understanding of arbitrary rewording; recognition is a conservative conjunction.

## Constraints (verified)

- mcpmark per-turn: `max_tool_calls_per_turn=4`, `max_mutating_tool_calls_per_turn=1`, reads
  before mutations, no terminal tool; no `max_model_calls` cap.
- The filesystem server exposes only single writers (`write_file`/`edit_file`/`move_file`/
  `create_directory`); reads via `read_multiple_files` (`{paths:[...]}`) and `read_text_file`;
  `get_file_info`/`list_directory`. So multiple files = multiple turns (1 mutation/turn).
- `create_directory` cannot create a missing grandparent (server-side `validatePath`).
- Tool results for turn N appear only in turn N+1's `observation["tool_results"]`.

## Design

### One new function, explicit branches (no registry)

`student_tools/deterministic_mcpmark.py`:

```python
class DeterministicDecision:  # calls: list[SanitizedToolCall] | is_final: bool | None
    ...

def plan_deterministic_mcpmark(request, state) -> DeterministicDecision:
    # returns: calls to emit, OR is_final=True (solver owns the final), OR None (yield to LLM)
    # side-effect-free when it does not recognize.
```

Branches (one per solver) are evaluated only for `benchmark == "mcpmark"`. Each branch:
1. checks its conservative instruction signature + required tools present + `test_directory`;
2. computes progress from the cumulative `tool_results` (successful calls only);
3. emits the next step, or final when its post-condition is met.

### Hook

In `_react_step`, inside the existing `if request.benchmark == "mcpmark":` block, immediately
after the `get_pending_mcpmark_batch_reads` early-return and before `normalize_tools_for_openai`:

```python
decision = plan_deterministic_mcpmark(request, state)
if decision and decision.is_final:
    record_assistant_final(state, "")           # runner scores on final
    return AgentResponse.final(request.request_id, "")
if decision and decision.calls:
    safe = sanitize_tool_calls(benchmark="mcpmark", raw_calls=[as_dict(c) for c in decision.calls],
                               available_tools=request.tools, limits=request.limits)
    if safe:                                    # if sanitize empties it, fall through to the LLM
        return self._emit_sanitized_calls(request, state, "", safe)
```

### Safety invariants (must be enforced by construction)

- Solvers emit at most one mutating call per turn, reads before mutations; and the batch is
  still passed through `sanitize_tool_calls`.
- Per-solver attempt counter in `state`; after N failures of a deterministic call (`ok:false`)
  the solver is disabled for the case and control falls through to the LLM.
- Recognition is a conjunction of distinctive substrings plus tool presence; side-effect-free
  on a miss. `case_id` is not used for hidden (keys differ).

### Phase A solver

- `file_splitting` — signature: instruction asks to split `large_file.txt` into exactly 3
  files in `split/` (require `split_01.txt` / "exactly 3 files" AND `large_file.txt`, so it
  cannot collide with `pattern_matching`). Sequence: read `large_file.txt` (if absent) →
  `create_directory(split)` → three `write_file`s with `content[:n//3], [n//3:2n//3],
  [2n//3:]` → final. Pure index split satisfies the verifier (exact concat, ≤1 diff, exactly
  3 files). Emit on the tool name present in `request.tools` (`read_text_file`), not hardcoded.

### Phase B (next, same rails)

Highest value first:
1. `timeline_extraction` — parse dates from raw read results and write the exact entry list
   (sidesteps the 65k truncation that crippled it).
2. `duplicates_searching` — deterministic duplicate-session detection.
3. `budget_computation` — arithmetic over the read data.
Deferred: `time_classification`, `requirements_writing`, `output_analysis`, postgres analysis
(some already emitted by `get_pending_mcpmark_batch_reads`).

## Validation

- Unit tests per solver (`recognize`/`plan` over synthetic tool_results; termination; attempt
  cap; sanitize fallback).
- Offline replay: build a small harness that replays a recorded trajectory's tool results into
  the task's real test env and runs `verify.py`; require PASS. Acknowledged as real work
  (trajectories are gitignored); A1's ad-hoc replay is the template.
- No-regression: offline suite + `smoke_local.py` green; unrecognized tasks unchanged.
- No paid runs without discussing with the user.

## Left to the builder

- Confirm `read_text_file` vs `read_file` presence per task from `request.tools`.
- Exact recognition strings from each task's `description.md`.
- Replay harness location and how to seed `FILESYSTEM_TEST_DIR` from the pinned test data.
