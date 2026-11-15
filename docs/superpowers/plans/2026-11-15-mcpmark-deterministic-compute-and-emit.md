# MCPMark Deterministic Compute-and-Emit (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic compute-and-emit path to the MCPMark agent so recognized tasks are solved by Python from raw tool results (the agent emits the exact tool calls and owns the final), starting with `file_splitting`.

**Architecture:** A new `student_tools/deterministic_mcpmark.py` exposes one function `plan_deterministic_mcpmark(request, state)` returning a `DeterministicDecision` (calls to emit, or a final), or `None` to yield to the existing LLM path. It is hooked in `_react_step` right after `get_pending_mcpmark_batch_reads`, is mcpmark-only, routes emitted calls through `sanitize_tool_calls`, and is side-effect-free when it does not recognize a task. Task solvers are explicit internal branches, not a registry.

**Tech Stack:** Python 3.11, pytest, the course `course_agent` package, the vendored MCPMark filesystem server (`request.tools`).

**Spec:** `docs/superpowers/specs/2026-11-15-mcpmark-deterministic-compute-and-emit-design.md`

## Global Constraints

- Every environment access goes through `request.tools` calls. Helpers may only read `request.observation["tool_results"]`; never touch the filesystem or DB directly.
- mcpmark per-turn policy: `max_tool_calls_per_turn=4`, `max_mutating_tool_calls_per_turn=1`; read-only calls must precede any mutating call; no terminal tool.
- The sanitized batch is passed through `sanitize_tool_calls` before emission; if it empties a non-empty decision, fall through to the LLM path.
- A solver MUST own the final (`AgentResponse.final`) once its post-condition holds; it must never hand control back to the LLM after emitting mutations.
- A solver must never emit a call identical (name + arguments) to one already present in `tool_results` with a failed result (`ok is False`, or text starting with `Error`). Applies to ALL decisions, reads included.
- `parse_read_files_from_tool_results` and `get_written_files_from_tool_results` key by FULL path (the write helper also adds a basename alias). Match read lookups by `os.path.basename`, never by bare-substring containment.
- Recognition never uses `case_id`/`case_key` (hidden keys differ); it uses instruction substrings plus required tools present, and each recognition test is bound to the real `description.md` text.
- Filesystem tool names come from `request.tools` (`read_text_file`, `write_file`, `create_directory`, `read_multiple_files`, `list_directory`, `move_file`); do not hardcode a name that may be absent.
- Run tests with `uv run --locked pytest` from `student_release/`. Never run a paid/official benchmark task without asking the user.

## Review Focus

- **Read on subsequent turns:** the read-detection check must use the full-path/basename key, or the solver loops reading forever → `TurnLimitExceeded` → 0.0. Test in Task 2.
- **Failed read re-emitted:** the failure guard must cover read-only decisions too. Test in Task 1.
- **Present-but-wrong artifact:** if `split/split_0N.txt` exists with the wrong content, the solver must overwrite it, not skip and hand back to the LLM. Test in Task 2.
- **Unrecognized task / non-mcpmark:** `plan_deterministic_mcpmark` returns `None`; behavior byte-identical to today. Tests in Task 1.
- **Accepted risk (statement, not a test):** if the failure guard disables a solver after a partial deterministic write, control falls back to the LLM, which could corrupt artifacts. Accepted because it is no worse than the status quo (all these cases already fail).

---

### Task 1: Framework module, decision type, failure guard, and hook

**Files:**
- Create: `student_release/src/course_agent/student_tools/deterministic_mcpmark.py`
- Modify: `student_release/src/course_agent/student_tools/__init__.py`
- Modify: `student_release/src/course_agent/student_agent.py` (mcpmark block after `get_pending_mcpmark_batch_reads`, ~line 465)
- Test: `student_release/tests/test_deterministic_mcpmark.py`

**Interfaces:**
- Produces: `DeterministicDecision` (fields: `calls: list[SanitizedToolCall]`, `is_final: bool`).
- Produces: `plan_deterministic_mcpmark(request: AgentRequest, state: dict) -> DeterministicDecision | None`.
- Produces: `_emitted_call_failed_before(request, state, call) -> bool` — true iff `tool_results` contains an item with the same tool name and equal `tool_arguments` whose result failed.
- Consumes: `SanitizedToolCall`, `sanitize_tool_calls` (from `policy_guard`); `get_written_files_from_tool_results`, `parse_read_files_from_tool_results`, `extract_mcp_text` (from `data_helpers`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_deterministic_mcpmark.py
import pytest
from course_agent.protocol import AgentRequest
from course_agent.student_tools.deterministic_mcpmark import plan_deterministic_mcpmark


def _req(benchmark="mcpmark", case_id="filesystem/standard/desktop/project_management",
         instruction="Organize the desktop.", tool_results=None, tools=None):
    return AgentRequest.from_dict({
        "request_id": "r", "context_id": "c", "benchmark": benchmark, "case_id": case_id,
        "step": 0, "text": instruction,
        "observation": {"test_directory": "/root/work", "instruction": instruction,
                        "tool_results": tool_results or []},
        "tools": tools or [{"name": "read_text_file"}, {"name": "write_file"},
                           {"name": "create_directory"}],
        "limits": {"max_agent_turns": 80, "model_input_tokens": 64000},
    })


def test_returns_none_for_unrecognized_mcpmark_task():
    assert plan_deterministic_mcpmark(_req(), {}) is None


def test_returns_none_for_non_mcpmark_benchmark():
    assert plan_deterministic_mcpmark(_req(benchmark="tau2_airline"), {}) is None


def test_returns_none_when_disabled():
    assert plan_deterministic_mcpmark(_req(), {"det_disabled": True}) is None
```

Add, after Task 2 defines the solver, a framework-guard test (write it now, it may initially fail until Step 3's guard exists — run in Step 2 to see it fail):

```python
def test_framework_disables_when_identical_read_already_failed():
    failed_read = [{"tool_name": "read_text_file", "ok": False, "result": None,
                    "tool_arguments": {"path": "/root/work/large_file.txt"}}]
    state = {}
    req = _req(instruction="Split large_file.txt into exactly 3 files.", tool_results=failed_read)
    assert plan_deterministic_mcpmark(req, state) is None
    assert state.get("det_disabled") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_mcpmark.py`
Expected: FAIL — `ModuleNotFoundError: course_agent.student_tools.deterministic_mcpmark`

- [ ] **Step 3: Implement the module (framework + guard)**

In `deterministic_mcpmark.py`:
- module docstring stating the "no direct env access; emit request.tools calls only" rule;
- `@dataclass DeterministicDecision` with `calls: list[SanitizedToolCall] = field(default_factory=list)` and `is_final: bool = False`;
- `_instruction(request) -> str` = `observation["instruction"] or observation["task"] or request.text`;
- `_has_tool(request, name) -> bool`;
- `_emitted_call_failed_before(request, state, call) -> bool`: scan `observation["tool_results"]`; match `tool_name == call.name` and `tool_arguments == call.arguments`; failed iff `ok is False`, `skipped` truthy, `result.isError is True`, or `extract_mcp_text(item).lstrip().lower().startswith("error")`;
- the entry point:

```python
def plan_deterministic_mcpmark(request, state):
    if request.benchmark != "mcpmark" or state.get("det_disabled"):
        return None
    for solver in _SOLVERS:                       # -> (_file_splitting_solver,) in Task 2
        decision = solver(request, state)
        if decision is None:
            continue
        # Guard applies to EVERY decision (read-only included): never re-emit a call
        # identical to one that already failed in tool_results.
        for call in decision.calls:
            if _emitted_call_failed_before(request, state, call):
                state["det_disabled"] = True
                return None
        return decision
    return None
```

Export `plan_deterministic_mcpmark` and `DeterministicDecision` from `student_tools/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_mcpmark.py`
Expected: PASS

- [ ] **Step 5: Wire the hook in `_react_step`**

In `student_agent.py`, inside the existing `if request.benchmark == "mcpmark":` block, immediately after the `get_pending_mcpmark_batch_reads` early-return, add:

```python
decision = plan_deterministic_mcpmark(request, state)
if decision is not None:
    if decision.is_final:
        record_assistant_final(state, "")
        return AgentResponse.final(request.request_id, "")
    if decision.calls:
        safe_calls = sanitize_tool_calls(
            benchmark="mcpmark",
            raw_calls=list(decision.calls),
            available_tools=request.tools,
            limits=request.limits,
        )
        if safe_calls:
            return self._emit_sanitized_calls(request, state, "", safe_calls)
```

Import `plan_deterministic_mcpmark` in the existing `student_tools` import block.

- [ ] **Step 6: Add a wiring test and run the focused suite**

Add: an unrecognized mcpmark request driven through `StudentAgent.solve_mcpmark` with a `FakeLLM` returning a content-only final; assert the response is that final (LLM path unchanged).

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_mcpmark.py tests/test_student_agent.py`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add student_release/src/course_agent/student_tools/deterministic_mcpmark.py \
        student_release/src/course_agent/student_tools/__init__.py \
        student_release/src/course_agent/student_agent.py \
        student_release/tests/test_deterministic_mcpmark.py
git commit -m "feat(mcpmark): deterministic compute-and-emit hook (framework + guard)"
```

---

### Task 2: `file_splitting` solver

**Files:**
- Modify: `student_release/src/course_agent/student_tools/deterministic_mcpmark.py`
- Test: `student_release/tests/test_deterministic_mcpmark.py`

**Interfaces:**
- Produces: `_file_splitting_solver(request, state) -> DeterministicDecision | None`.
- Produces: `_split_three(content: str) -> list[str]` returning three strings whose concatenation equals `content` and whose lengths differ by ≤1.
- Consumes: `parse_read_files_from_tool_results`, `get_written_files_from_tool_results`, `SanitizedToolCall`, and the framework guard from Task 1.

- [ ] **Step 1: Write the failing tests**

```python
import os
from pathlib import Path
from course_agent.student_tools.deterministic_mcpmark import _split_three, _file_splitting_solver

TASKS = Path(__file__).resolve().parents[2] / "third_party/mcpmark/tasks/filesystem"

def _desc(rel):
    p = TASKS / rel
    return p.read_text() if p.exists() else ""


def test_split_three_preserves_content_and_balances():
    s = "abcdefghij"  # 10 chars
    parts = _split_three(s)
    assert "".join(parts) == s
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_recognition_uses_real_descriptions():
    split_desc = _desc("easy/file_context/file_splitting/description.md")
    pat_desc = _desc("easy/file_context/pattern_matching/description.md")
    assert "large_file.txt" in split_desc and "split" in split_desc
    assert "large_file.txt" in pat_desc and "split" not in pat_desc.lower()


def test_solver_emits_read_when_file_not_yet_read_and_no_failed_read():
    req = _req(instruction=_desc("easy/file_context/file_splitting/description.md"))
    d = _file_splitting_solver(req, {})
    assert d is not None and len(d.calls) == 1 and d.calls[0].name == "read_text_file"
```

Plus tests for: read detection `_has_read_basename(request, "large_file.txt")` returns True when a `read_text_file` result for `.../large_file.txt` is present; the solver emits `create_directory` next; emits `write_file` for the first part whose recorded content differs; and `is_final` when all three recorded contents equal the computed parts.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_mcpmark.py -k "split or recognition"`
Expected: FAIL (`_split_three` / `_file_splitting_solver` not defined)

- [ ] **Step 3: Implement the solver**

- Recognition: instruction contains `"large_file.txt"` AND `"split"` AND (`"split_01"` or `"exactly 3"` or `"similar character counts"`); `test_directory` present; and `read_text_file`, `write_file`, `create_directory` all in `request.tools`.
- `_split_three(c)` returns `[c[:n//3], c[n//3:2*n//3], c[2*n//3:]]`.
- Read lookup: build `{os.path.basename(p): v for p, v in parse_read_files_from_tool_results(tool_results).items()}` and read `"large_file.txt"` by basename.
- Plan logic (re-entrant, derived from `tool_results` each call):
  1. If the read is absent: emit one `read_text_file` at `<test_directory>/large_file.txt`.
  2. Compute parts from the read. Let `written = get_written_files_from_tool_results(tool_results)`.
  3. If `create_directory` for `<test_directory>/split` has not succeeded: emit it.
  4. Else, for N in 1..3: if `written` lacks `split_0N.txt` OR its recorded content != the computed part, emit `write_file({"path": "<test_directory>/split/split_0N.txt", "content": part})` for the FIRST such N, and return. (Overwrite is safe; `write_file` overwrites atomically.)
  5. If all three recorded contents equal the computed parts: return `DeterministicDecision(is_final=True)`.

(Optional hardening if stray files are a concern: one `list_directory(split)` before finalizing and, if extra names exist, `move_file` them to `<test_directory>/` before the final. Only add if a stray-file case is observed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_mcpmark.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add student_release/src/course_agent/student_tools/deterministic_mcpmark.py student_release/tests/test_deterministic_mcpmark.py
git commit -m "feat(mcpmark): deterministic file_splitting solver"
```

---

### Task 3: Integration test through the hook + offline replay through the real verifier

**Files:**
- Test: `student_release/tests/test_deterministic_replay.py`

**Interfaces:**
- Consumes: `plan_deterministic_mcpmark`, `_split_three`.
- Reads: `third_party/mcpmark/test_environments/file_context/` and `third_party/mcpmark/tasks/filesystem/easy/file_context/file_splitting/verify.py`.

- [ ] **Step 1: Write the integration test (drives the solver across turns)**

Simulate the runner loop with synthetic tool_results, asserting the exact emitted sequence and that each batch obeys the per-turn policy (`sanitize_tool_calls` returns the batch unchanged; ≤1 mutating call; reads precede mutations). Turn 1 (empty results) → `read_text_file`; after a successful read result → `create_directory`; after a successful create → `write_file` part 1, then 2, then 3; after all three → `is_final` is True. Include a variant where `split_01.txt` was written with WRONG content and assert the solver re-emits a `write_file` for it (not `None`).

- [ ] **Step 2: Run the integration test**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_replay.py -v`
Expected: PASS

- [ ] **Step 3: Add the end-to-end verifier replay (correctness proof)**

```python
import os, shutil, subprocess, sys
from pathlib import Path
import pytest
from course_agent.student_tools.deterministic_mcpmark import _split_three

REPO = Path(__file__).resolve().parents[2]
TEST_ENV = REPO / "third_party/mcpmark/test_environments/file_context"
VERIFY = REPO / "third_party/mcpmark/tasks/filesystem/easy/file_context/file_splitting/verify.py"

pytestmark = pytest.mark.skipif(not TEST_ENV.exists() or not VERIFY.exists(), reason="mcpmark test data missing")


def test_split_result_passes_real_verifier(tmp_path):
    test_dir = tmp_path / "test_directory"
    shutil.copytree(TEST_ENV, test_dir)
    content = (test_dir / "large_file.txt").read_text()
    split_dir = test_dir / "split"; split_dir.mkdir()
    for i, part in enumerate(_split_three(content), start=1):
        (split_dir / f"split_{i:02d}.txt").write_text(part)
    env = {**os.environ, "FILESYSTEM_TEST_DIR": str(test_dir)}
    r = subprocess.run([sys.executable, str(VERIFY)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
```

- [ ] **Step 4: Run it**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_replay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add student_release/tests/test_deterministic_replay.py
git commit -m "test(mcpmark): file_splitting integration + real-verifier replay"
```

---

### Task 4: Full-suite regression and docs

**Files:** none new.

- [ ] **Step 1: Run the full offline suite**

Run: `cd student_release && uv run --locked pytest -q -rs --ignore=tests/test_public_runner_backends.py`
Expected: all pass (no regressions).

- [ ] **Step 2: Run the smoke test and capture its exact output**

Run: `cd student_release && uv run --locked python smoke_local.py`
Expected: prints the smoke success line (record the exact string; assert on it only if a smoke test asserts it).

- [ ] **Step 3: Update `PLAN.md` and commit docs**

```bash
git add PLAN.md docs/superpowers/
git commit -m "docs: record deterministic compute-and-emit Phase A"
```

---

## Self-Review

- **Spec coverage:** hook + termination (Task 1 Steps 5-6), sanitize routing (Task 1 Step 5), framework failure guard covering reads (Task 1 Step 3), file_splitting with basename matching + overwrite-on-mismatch (Task 2), integration + real-verifier replay (Task 3), no-regression (Task 4). `time_classification` and Phase B solvers are explicitly deferred, matching the spec.
- **Type consistency:** `DeterministicDecision.calls` is `list[SanitizedToolCall]` everywhere; `sanitize_tool_calls` accepts them via its `getattr(item, "name")` branch (confirmed); `_split_three` and `_file_splitting_solver` names stable across tasks.
- **Review findings folded:** basename-not-substring read lookup; guard on all decisions; overwrite mismatched artifacts; integration test that actually exercises the solver/hook; recognition tests bound to real descriptions; `create_directory` added to the required-tools gate.
- **Placeholders:** none; every step names exact files, functions, and commands.
- **Phase B** (timeline_extraction, duplicates_searching, budget_computation) gets its own plan after Phase A lands.
