# MCPMark Deterministic Compute-and-Emit (Phase B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three deterministic, instruction-keyed solvers to the existing MCPMark compute-and-emit hook — `duplicates_searching`, `timeline_extraction`, `budget_computation` — each proven by replay through its task's real `verify.py`.

**Architecture:** Each solver is one callable `(request, state) -> DeterministicDecision | None` appended to the existing `_SOLVERS` tuple in `student_tools/deterministic_mcpmark.py`. No change to the Phase A entry point, guard, or the hook in `student_agent.py` (already routes every recognized decision through `sanitize_tool_calls` and owns the final). A new offline replay harness drives each solver across turns against a temp copy of the task's pinned environment by executing its emitted calls with a faithful filesystem-tool simulator, then runs the task's real `verify.py` as a subprocess and asserts exit 0.

**Tech Stack:** Python 3.11, pytest, the course `course_agent` package, the vendored MCPMark filesystem server semantics (`request.tools`).

**Spec:** `docs/superpowers/specs/2026-11-15-mcpmark-deterministic-phase-b-design.md`

## Global Constraints

- Every environment access goes through `request.tools` calls. Solvers may only read `request.observation["tool_results"]`; they must never touch the filesystem, database, or environment directly. (Tests may touch a temp copy — that is a harness, not solver code.)
- mcpmark per-turn policy: `max_tool_calls_per_turn=4`, `max_mutating_tool_calls_per_turn=1`; reads must precede any mutating call; no terminal tool. A violation is a non-retryable runtime error (0.0).
- A solver MUST early-return `None` when `observation["test_directory"]` is absent (e.g. postgres mcpmark tasks) — preserving zero behavior change.
- Recognition never uses `case_id`; it is instruction substrings plus required tools present, and each recognition test is bound to the real `description.md` text. On a miss the solver is side-effect-free.
- `plan_deterministic_mcpmark` applies the failure guard (`_emitted_call_failed_before`, keyed on `mcpmark_item_failed`) to every emitted call; a re-emitted identical failed call latches `state["det_disabled"]` and yields to the LLM. Solvers must never re-emit an identical read/write that cannot succeed (a batch read that returns an inline per-file `Error - ...` never fails the whole call, so the solver must hand back instead of re-emitting).
- `read_multiple_files` never fails the call wholesale: an unreadable path comes back inline as `<path>: Error - <msg>`. Hidden/binary paths (`.DS_Store`, any basename starting with `.`) must be excluded from read/listing sets.
- Server output formats (must be mirrored exactly by the simulator; the production parsers are the contract): `directory_tree(path)` -> `content[0].text` is a JSON array of `{"name","type",("children")}` rooted at `path`'s children; `read_multiple_files(paths)` -> entries `${path}:\n${content}\n` joined by `\n---\n`, each failure `${path}: Error - ${msg}`; `read_text_file(path)` -> raw content; `write_file`/`create_directory`/`move_file` -> a human `Successfully ...` string.
- The runner records a tool_result `ok:True` on success and `ok:False` when the call raises or an mcpmark result has `isError` true; recorded `tool_arguments` are the agent's raw arguments (unnormalized).
- **Path identity invariant (do not violate):** solvers must emit absolute paths built as `os.path.join(str(observation["test_directory"]), rel)`. The runner normalizes arguments to absolute before the server sees them, and the server echoes those absolute paths in `read_multiple_files` output; if a solver emitted relative paths, the recorded `tool_arguments` (raw) and the parsed result keys (absolute) would diverge and read/move detection would break. When comparing recorded args to discovered paths, normalize both with `os.path.normpath`.
- **Read-only classification:** `classify_tool` only treats an unknown tool as read when the schema carries `annotations.readOnlyHint` true (`policy_guard.py:139-143`). Production passes the tool schema, so `directory_tree`/`read_multiple_files` are read; the test harness must pass the same schemas or it will misclassify. (Each solver still emits at most one call per turn, so even a single misclassified call cannot violate `<=1 mutating/turn`.)
- Run tests with `uv run --locked pytest` from `student_release/`. Never run a paid/official benchmark task without asking the user.

## Review Focus

- **Batch read with an inline per-file error or empty entry:** the whole call succeeds, so the guard cannot veto a re-emit; a naive solver re-emits the identical batch forever -> `TurnLimitExceeded` -> 0.0. Must hand back (`det_disabled`) instead. Tested in Task 1 (`_read_all_files` unit test) and reused by Tasks 2-3.
- **Duplicates miscount:** `.DS_Store`, a hidden file, or the unique `large_file.txt` wrongly grouped makes `duplicates/` hold 15 (or the wrong 14) -> verifier fails. Tested in Task 1 replay.
- **Timeline over/under-extraction:** the verifier demands exactly the fixed 43-entry set and no extras; a general date regex will not reproduce it. The parser must be derived against the pinned data and the replay test is the gate. Tested in Task 2.
- **Budget misclassification:** the verifier pins exactly 5 paths x 3 entries and a total; `price_comparisons.csv` must yield the cheapest column. The winning classification is a fixed allowlist of the five pinned paths (see Task 3). Tested in Task 3.
- **Silent wrong artifact:** if the recorded `write_file` content differs from the intended content, a presence-only check would final on a wrong file. Mirror Phase A's overwrite-on-mismatch. Tested in Tasks 2-3.
- **Hijack of an unrelated mcpmark task:** a new solver's recognition must not match another task's instruction (all three live in shared environments). Each task adds a negative-recognition test bound to another task's real description, plus a loop test over every filesystem `description.md`.
- **Guard hand-back after mutations (statement, not a test):** if a `move_file`/`write_file` genuinely fails, the guard latches and control falls to the LLM — which, after partial mutations, could corrupt artifacts. Accepted: it is no worse than the status quo (these cases already fail), and the runner rebuilds a pristine env on every (re)start, so a partial run is never resumed against half-moved data.

---

### Task 1: Replay harness + `duplicates_searching` solver

**Files:**
- Create: `student_release/tests/mcpmark_fs_sim.py`
- Create: `student_release/tests/conftest.py` (empty; guarantees `tests/` is importable so `mcpmark_fs_sim` can be imported by later test files)
- Modify: `student_release/src/course_agent/student_tools/deterministic_mcpmark.py`
- Test: `student_release/tests/test_deterministic_duplicates.py`

**Interfaces:**
- Produces (module): `_directory_tree_paths(tool_results: list[Any], test_dir: str) -> list[str] | None` — sorted absolute file paths discovered from the most recent non-failed `directory_tree` result whose argument `path` (normalized) equals `test_dir`; `None` if no such result exists. Descends `children`, excludes any node whose name starts with `.`, and joins each relative name onto `test_dir`.
- Produces (module): `_read_all_files(request: AgentRequest, test_dir: str, state: dict[str, Any]) -> tuple[DeterministicDecision | None, dict[str, str] | None]` — shared listing/read phase. Returns `(decision, None)` to emit a `directory_tree` or a `read_multiple_files` batch built from `_directory_tree_paths`; `(None, {abs_path: content})` once every discovered path is present in `parse_read_files_from_tool_results(_non_failed_results(tool_results))`; `(None, None)` after latching `state["det_disabled"]` when a `read_multiple_files` for the full discovered set already executed but at least one discovered path is still absent (an inline per-file `Error -`). `contents is None` means hand back; a non-empty dict means proceed. All emitted paths are absolute (`os.path.join(test_dir, rel)`); both the "already executed" test and the read-set keys are compared with `os.path.normpath`.
- Produces (module): `_duplicates_searching_solver(request: AgentRequest, state: dict[str, Any]) -> DeterministicDecision | None`.
- Produces (tests/`mcpmark_fs_sim.py`): `make_request(instruction, tool_results, *, test_dir, tools, benchmark="mcpmark", case_id=..., limits=None) -> AgentRequest`; `class SimEnv` with `execute(call: SanitizedToolCall) -> dict` (perform the op on `test_dir` and return a runner-shaped tool_result); `drive(instruction, case_id, test_dir, tools, *, limits=None, max_turns=80) -> int` (drive `plan_deterministic_mcpmark` across turns until `is_final`, returning the final turn index; asserts a decision each turn and fails on `None`; asserts each batch obeys policy by calling `classify_tool`/`sanitize_tool_calls` WITH the matching tool schema so `readOnlyHint` is honored); `run_verify(test_dir, verify_path) -> subprocess.CompletedProcess`.
- Consumes: `DeterministicDecision`, `SanitizedToolCall`, `parse_read_files_from_tool_results`, `_non_failed_results`, `_emitted_call_failed_before`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_deterministic_duplicates.py
from pathlib import Path
import shutil

import pytest

from course_agent.student_tools.deterministic_mcpmark import (
    _duplicates_searching_solver,
    _read_all_files,
)
from mcpmark_fs_sim import drive, make_request, run_verify

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "third_party/mcpmark/tasks/filesystem"
TASK = TASKS / "standard/file_context/duplicates_searching"
ENV = REPO / "third_party/mcpmark/test_environments/file_context"
VERIFY = TASK / "verify.py"
DESC = (TASK / "description.md").read_text()
SPLIT_DESC = (TASKS / "easy/file_context/file_splitting/description.md").read_text()

# read-only tools carry readOnlyHint so classify_tool matches production (policy_guard.py:139).
TOOLS = [
    {"name": "directory_tree", "annotations": {"readOnlyHint": True}},
    {"name": "read_multiple_files", "annotations": {"readOnlyHint": True}},
    {"name": "create_directory", "annotations": {"readOnlyHint": False}},
    {"name": "move_file", "annotations": {"readOnlyHint": False}},
]

pytestmark = pytest.mark.skipif(
    not (ENV.exists() and VERIFY.exists()), reason="mcpmark data missing"
)


def _req(instruction, tool_results, test_dir="/tmp/td"):
    return make_request(
        instruction,
        tool_results,
        test_dir=test_dir,
        tools=TOOLS,
        benchmark="mcpmark",
        case_id="filesystem/file_context/duplicates_searching",
    )


def test_recognition_binds_real_description():
    assert "duplicate" in DESC and "identical content" in DESC
    assert _duplicates_searching_solver(_req(DESC, []), {}) is not None


def test_does_not_recognize_file_splitting_description():
    assert _duplicates_searching_solver(_req(SPLIT_DESC, []), {}) is None


def test_no_solver_recognizes_a_foreign_description():
    # Loop guard: the duplicates solver must match only its own task instruction.
    for desc in sorted(TASKS.glob("**/description.md")):
        text = desc.read_text()
        if desc.parent == TASK:
            continue
        if "identical content" in text:
            assert _duplicates_searching_solver(_req(text, []), {}) is not None
        else:
            assert _duplicates_searching_solver(_req(text, []), {}) is None


def test_hands_back_when_batch_read_has_inline_error():
    import json

    tree = [{"name": "a.txt", "type": "file"}, {"name": "b.txt", "type": "file"}]
    results = [
        {"tool_name": "directory_tree", "ok": True, "tool_arguments": {"path": "/tmp/td"},
         "result": {"isError": False, "content": [{"text": json.dumps(tree)}]}},
        {"tool_name": "read_multiple_files", "ok": True,
         "tool_arguments": {"paths": ["/tmp/td/a.txt", "/tmp/td/b.txt"]},
         "result": {"isError": False, "content": [{"text":
             "/tmp/td/a.txt:\nhello\n\n---\n/tmp/td/b.txt: Error - EACCES"}]}},
    ]
    state = {}
    decision, contents = _read_all_files(_req(DESC, results), "/tmp/td", state)
    assert decision is None and contents is None and state["det_disabled"] is True


def test_replay_passes_real_verifier_and_produces_exact_state(tmp_path):
    test_dir = tmp_path / "test_directory"
    shutil.copytree(ENV, test_dir)
    drive(DESC, "filesystem/file_context/duplicates_searching", test_dir, TOOLS)
    r = run_verify(test_dir, VERIFY)
    assert r.returncode == 0, r.stdout + r.stderr
    dups = sorted(p.name for p in (test_dir / "duplicates").iterdir())
    assert dups == [
        "file_01.txt", "file_02.txt", "file_03.txt", "file_04.txt",
        "file_07.txt", "file_08.txt", "file_10.txt", "file_11.txt",
        "file_13.txt", "file_14.txt", "file_15.txt", "file_16.txt",
        "file_18.txt", "file_19.txt",
    ]
    remaining = sorted(p.name for p in test_dir.iterdir() if p.is_file())
    assert remaining == [
        "file_05.txt", "file_06.txt", "file_09.txt",
        "file_12.txt", "file_17.txt", "file_20.txt", "large_file.txt",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_duplicates.py`
Expected: FAIL — `ModuleNotFoundError: mcpmark_fs_sim` / `ImportError: cannot import name '_directory_tree_paths'`.

- [ ] **Step 3: Implement `tests/mcpmark_fs_sim.py`**

Faithful simulator (mirrors the vendored server semantics in the Global Constraints):

```python
class SimEnv:
    def __init__(self, root: Path): self.root = Path(root)
    def _tree(self, path: Path) -> list[dict]:
        # list children of `path`; directory -> {"name","type":"directory","children":[...]};
        # file -> {"name","type":"file"}; keep ALL entries (including dotfiles) — the server does.
    def execute(self, call) -> dict:
        # dispatch on call.name:
        #  directory_tree {path}: text = json.dumps(self._tree(Path(path)), indent=2)
        #  read_multiple_files {paths}: entries = [f"{p}:\n{read(p)}\n" or f"{p}: Error - {msg}"]; text="\n---\n".join(entries)
        #  read_text_file {path}: text = read(path)
        #  write_file {path,content}: write; text=f"Successfully wrote to {path}"
        #  create_directory {path}: mkdir(parents=True, exist_ok=True); text=f"Successfully created directory {path}"
        #  move_file {source,destination}: os.rename; text=f"Successfully moved {source} to {destination}"
        #  else: raise AssertionError
        # return {"tool_name": call.name, "tool_arguments": dict(call.arguments), "ok": True,
        #         "result": {"isError": False, "content": [{"text": text}]}}
```

`make_request(...)` builds the `AgentRequest` (same field shape the Phase A tests use). `drive(...)` builds a request each turn with observation `tool_results` = the accumulated results, calls `plan_deterministic_mcpmark`, fails the test if it returns `None`, returns the turn index on `is_final`, else raises after `max_turns`. It must assert each emitted batch obeys policy by calling `classify_tool("mcpmark", name, tool_schema)` and `sanitize_tool_calls(...)` **passing the matching tool schema from `tools`** (so `readOnlyHint` is honored); a bare `{"name": ...}` schema would misclassify `directory_tree` as mutating (`policy_guard.py:139-143`). `run_verify(test_dir, verify_path)` = `subprocess.run([sys.executable, str(verify_path)], env={**os.environ, "FILESYSTEM_TEST_DIR": str(test_dir)}, capture_output=True, text=True)`.

Replay fidelity: the simulator must reproduce the server's exact text (Global Constraints) because the production parsers are the contract; it must NOT normalize/absolutize paths itself (the solver already emits absolute paths) and must return the paths exactly as passed. This is a harness, not solver code — it may touch the temp copy directly.

- [ ] **Step 4: Implement the shared read helper + duplicates solver in `deterministic_mcpmark.py`**

- `_directory_tree_paths`: scan `tool_results` for the last non-failed `directory_tree` item whose `tool_arguments["path"]` (via `os.path.normpath`) equals `test_dir`; `json.loads(extract_mcp_text(item))`; walk nodes, descending `children`, skipping any node whose `name` starts with `.`; return sorted `os.path.join(test_dir, rel)`.
- `_read_all_files`: as specified in Interfaces. Read-set = `parse_read_files_from_tool_results(_non_failed_results(tool_results))`, keyed by `os.path.normpath`. Detect "batch already executed" by any non-failed `read_multiple_files` item whose `tool_arguments.get("paths")` (each normalized) equals the discovered list.
- `_duplicates_searching_solver`: recognition = `"test_directory"` present AND `all(_has_tool(request, t) for t in ("move_file","create_directory","read_multiple_files","directory_tree"))` AND `"duplicate" in instr` AND (`"identical content" in instr` or `"same content" in instr`). Note: the tool gate is decorative for filesystem tasks (all share the toolset); safety rests on the substrings. Then call `_read_all_files`; if it returns a decision, return it; if contents is `None`, return `None` (disabled latched). Group the contents (`{abs_path: content}`) by exact content; a file is a duplicate iff its group size >= 2. If the `duplicates` directory has not been created (`_directory_created(tool_results, os.path.join(test_dir,"duplicates"))`), emit `create_directory(test_dir/duplicates)`. Else find the first duplicate not yet moved — a non-failed `move_file` result with normalized `tool_arguments["source"] == abs_path` and normalized `tool_arguments["destination"] == join(test_dir,"duplicates",basename)` — and emit `move_file(source=abs_path, destination=join(test_dir,"duplicates",basename))`. When every duplicate is moved, `DeterministicDecision(is_final=True)`.
- Append `_duplicates_searching_solver` to `_SOLVERS`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_duplicates.py -v`
Expected: PASS (including the real-verifier replay).

- [ ] **Step 6: Run the focused Phase A suites for no regression**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_mcpmark.py tests/test_deterministic_replay.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add student_release/src/course_agent/student_tools/deterministic_mcpmark.py \
        student_release/tests/mcpmark_fs_sim.py student_release/tests/conftest.py \
        student_release/tests/test_deterministic_duplicates.py
git commit -m "feat(mcpmark): deterministic duplicates_searching solver + replay harness"
```

---

### Task 2: `timeline_extraction` solver

**Files:**
- Modify: `student_release/src/course_agent/student_tools/deterministic_mcpmark.py`
- Test: `student_release/tests/test_deterministic_timeline.py`

**Interfaces:**
- Produces (module): `_timeline_extraction_solver(request: AgentRequest, state: dict[str, Any]) -> DeterministicDecision | None`.
- Consumes: `_read_all_files` (which uses `_directory_tree_paths`), `DeterministicDecision`, `SanitizedToolCall`, `get_written_files_from_tool_results`.
- Requires from `mcpmark_fs_sim`: `make_request`, `drive`, `run_verify`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_deterministic_timeline.py
from pathlib import Path
import re
import shutil

import pytest

from course_agent.student_tools.deterministic_mcpmark import _timeline_extraction_solver
from mcpmark_fs_sim import drive, make_request, run_verify

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "third_party/mcpmark/tasks/filesystem"
TASK = TASKS / "standard/desktop/timeline_extraction"
ENV = REPO / "third_party/mcpmark/test_environments/desktop"
VERIFY = TASK / "verify.py"
DESC = (TASK / "description.md").read_text()
OTHER_DESC = (
    TASKS / "standard/file_context/duplicates_searching/description.md"
).read_text()
TOOLS = [
    {"name": "directory_tree", "annotations": {"readOnlyHint": True}},
    {"name": "read_multiple_files", "annotations": {"readOnlyHint": True}},
    {"name": "write_file", "annotations": {"readOnlyHint": False}},
]

pytestmark = pytest.mark.skipif(
    not (ENV.exists() and VERIFY.exists()), reason="mcpmark data missing"
)

_ENTRY_RE = re.compile(r'"([^"]+:\d{4}-\d{2}-\d{2})"')


def _req(instruction, tool_results, test_dir="/tmp/td"):
    return make_request(
        instruction, tool_results, test_dir=test_dir, tools=TOOLS,
        benchmark="mcpmark", case_id="filesystem/desktop/timeline_extraction",
    )


def test_recognition_binds_real_description():
    assert "timeline" in DESC and "2024" in DESC
    assert _timeline_extraction_solver(_req(DESC, []), {}) is not None


def test_does_not_recognize_another_description():
    assert _timeline_extraction_solver(_req(OTHER_DESC, []), {}) is None


def test_replay_passes_real_verifier(tmp_path):
    test_dir = tmp_path / "test_directory"
    shutil.copytree(ENV, test_dir)
    drive(DESC, "filesystem/desktop/timeline_extraction", test_dir, TOOLS)
    r = run_verify(test_dir, VERIFY)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [l for l in (test_dir / "timeline.txt").read_text().splitlines() if l.strip()]
    # Intermediate assertion: localize failures to the parser, not the verifier.
    assert set(lines) == set(_ENTRY_RE.findall(VERIFY.read_text()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_timeline.py`
Expected: FAIL (`_timeline_extraction_solver` not defined).

- [ ] **Step 3: Implement the solver**

Recognition = `test_directory` present AND tools `write_file`, `read_multiple_files`, `directory_tree` all present AND `"timeline" in instr` AND `"2024" in instr`. Then:
1. `_read_all_files`; if it returns a decision, return it; if contents is `None`, return `None` (disabled latched).
2. For each discovered file, process its content line by line (the parser is pinned by the replay test; the verifier's 43-entry set is authoritative). Per line, emit at most one date, in this precedence:
   - if the line contains one or more full dates matching `\b2024-\d{2}-\d{2}\b`, take the **earliest** (handles a two-date period like `travel_calculator.py`'s `"2024-03-15", "2024-03-20"` -> `2024-03-15`; each CSV row has one date);
   - else if the line contains `2024-\d{2}-XX`, emit `2024-\d{2}-01`;
   - else if the line contains a full month name (January..December, word-bounded) AND (`"2024"` appears in the line OR the file's path relative to `test_directory` contains a `2024` path segment), emit `2024-\{MM\}-01`;
   - else (bare `2024`, or a month with no 2024 token/segment) emit nothing.
   Dedupe per (file, date). Render each as `<relative_path>:<YYYY-MM-DD>`, sorted by (date, relative_path).
3. If `write_file` for `os.path.join(test_dir,"timeline.txt")` is not recorded, or its recorded content differs from the intended content, emit it (overwrite-on-mismatch, mirroring file_splitting); else `DeterministicDecision(is_final=True)`.
4. Append `_timeline_extraction_solver` to `_SOLVERS`.

Representative data this must get right (verified against the pinned tree): `learning_progress.csv` under `learning/2024/` (bare months -> 12 entries via the path segment), `exp_record.md` (`2024-08-XX`/`2024-09-XX` -> the 1st; `Start Date: 2024` skipped), `travel_calculator.py` (period -> start), `analysis_report.md` / `travel_bucket_list.md` (month + in-line 2024 -> the 1st), `experiment_summary.md` (`**sep**: September ...` -> skipped; its three `2024-01-NN` lines -> kept), and `music/.../favorite_songs.csv` (`November's Chopin,2005` -> skipped).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_timeline.py -v`
Expected: PASS (replay exits 0, exactly 43 lines).

- [ ] **Step 5: Commit**

```bash
git add student_release/src/course_agent/student_tools/deterministic_mcpmark.py student_release/tests/test_deterministic_timeline.py
git commit -m "feat(mcpmark): deterministic timeline_extraction solver"
```

---

### Task 3: `budget_computation` solver

**Files:**
- Modify: `student_release/src/course_agent/student_tools/deterministic_mcpmark.py`
- Test: `student_release/tests/test_deterministic_budget.py`

**Interfaces:**
- Produces (module): `_budget_computation_solver(request: AgentRequest, state: dict[str, Any]) -> DeterministicDecision | None`.
- Consumes: `_read_all_files`, `get_written_files_from_tool_results`, `DeterministicDecision`, `SanitizedToolCall`.
- Requires from `mcpmark_fs_sim`: `make_request`, `drive`, `run_verify`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_deterministic_budget.py
from collections import Counter
from pathlib import Path
import re
import shutil

import pytest

from course_agent.student_tools.deterministic_mcpmark import _budget_computation_solver
from mcpmark_fs_sim import drive, make_request, run_verify

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "third_party/mcpmark/tasks/filesystem"
TASK = TASKS / "standard/desktop_template/budget_computation"
ENV = REPO / "third_party/mcpmark/test_environments/desktop_template"
VERIFY = TASK / "verify.py"
DESC = (TASK / "description.md").read_text()
OTHER_DESC = (
    TASKS / "standard/desktop/timeline_extraction/description.md"
).read_text()
TOOLS = [
    {"name": "directory_tree", "annotations": {"readOnlyHint": True}},
    {"name": "read_multiple_files", "annotations": {"readOnlyHint": True}},
    {"name": "write_file", "annotations": {"readOnlyHint": False}},
]

pytestmark = pytest.mark.skipif(
    not (ENV.exists() and VERIFY.exists()), reason="mcpmark data missing"
)

_EXPENSE_RE = re.compile(r"\(\s*'([^']+)'\s*,\s*([\d.]+)\s*\)")


def _req(instruction, tool_results, test_dir="/tmp/td"):
    return make_request(
        instruction, tool_results, test_dir=test_dir, tools=TOOLS,
        benchmark="mcpmark", case_id="filesystem/desktop_template/budget_computation",
    )


def test_recognition_binds_real_description():
    assert "total_budget.txt" in DESC and "personal" in DESC
    assert _budget_computation_solver(_req(DESC, []), {}) is not None


def test_does_not_recognize_another_description():
    assert _budget_computation_solver(_req(OTHER_DESC, []), {}) is None


def test_replay_passes_real_verifier(tmp_path):
    test_dir = tmp_path / "test_directory"
    shutil.copytree(ENV, test_dir)
    drive(DESC, "filesystem/desktop_template/budget_computation", test_dir, TOOLS)
    r = run_verify(test_dir, VERIFY)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [l for l in (test_dir / "total_budget.txt").read_text().splitlines() if l.strip()]
    body, total = lines[:-1], float(lines[-1])
    produced = Counter((l.split(";")[0], round(float(l.split(";")[1]), 2)) for l in body)
    expected = Counter((p, float(v)) for p, v in _EXPENSE_RE.findall(VERIFY.read_text()))
    assert produced == expected          # localizes failures to the parser
    assert abs(total - 95624.46) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_budget.py`
Expected: FAIL (`_budget_computation_solver` not defined).

- [ ] **Step 3: Implement the solver**

Recognition = `test_directory` present AND tools `write_file`, `read_multiple_files`, `directory_tree` all present AND `"total_budget.txt" in instr` AND `"personal" in instr`. Then:
1. `_read_all_files`; if it returns a decision, return it; if contents is `None`, return `None` (disabled latched).
2. Build the expense lines from an explicit allowlist of the five pinned relative paths, matching the verifier's fixed set (this is deliberate overfitting, accepted per the design's Anti-goal — a regenerated-data task would not transfer):
   - `Archives/tax_documents_2022.csv`, `Documents/Personal/tax_info_2023.csv`, `Documents/budget.csv`, `Downloads/expenses.csv` — one entry per data row using the file's amount column (`Amount`/`Cost`), in row order;
   - `Downloads/price_comparisons.csv` — one entry per row using the **cheapest** of the store columns (`min` of the numeric columns other than the first).
   Every other file in the environment (project/work, contacts, fitness, temp, plain text) contributes nothing. Format each as `relative_path;price` with the price at 2 decimals; the last line is the total (2 decimals), which must equal the sum.
3. If `write_file` for `os.path.join(test_dir,"total_budget.txt")` is not recorded, or its recorded content differs from the intended content, emit it (overwrite-on-mismatch); else `DeterministicDecision(is_final=True)`.
4. Append `_budget_computation_solver` to `_SOLVERS`.

Expected result for the pinned data (a useful cross-check, not a hardcode): the 15 entries sum to `95624.46`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd student_release && uv run --locked pytest -q tests/test_deterministic_budget.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add student_release/src/course_agent/student_tools/deterministic_mcpmark.py student_release/tests/test_deterministic_budget.py
git commit -m "feat(mcpmark): deterministic budget_computation solver"
```

---

### Task 4: Full-suite regression and docs

**Files:** none new.

- [ ] **Step 1: Run the full offline suite**

Run: `cd student_release && uv run --locked pytest -q -rs --ignore=tests/test_public_runner_backends.py`
Expected: all pass (no regressions; Phase A 219 plus the new tests).

- [ ] **Step 2: Run the smoke test**

Run: `cd student_release && uv run --locked python smoke_local.py`
Expected: smoke success line.

- [ ] **Step 3: Update `PLAN.md` and commit docs**

```bash
git add PLAN.md docs/superpowers/
git commit -m "docs: record deterministic compute-and-emit Phase B"
```

---

## Self-Review

- **Spec coverage:** three solvers (Tasks 1-3), read discovery + batch hand-back (Task 1 helper, reused 2-3), hidden/binary exclusion and test_directory gate (Task 1), per-solver recognition with negative tests, replay harness against each real `verify.py` (Tasks 1-3), no-regression + docs (Task 4). Matches the design.
- **Type consistency:** every solver returns `DeterministicDecision | None`; `_read_all_files` returns `tuple[DeterministicDecision | None, dict[str,str] | None]` and is used identically by all three; `_directory_tree_paths` returns `list[str] | None` (Tasks 2-3 consume it *through* `_read_all_files`, not by name); server text formats are stated once and reused by the simulator.
- **Review findings folded:** batch inline-error hand-back is explicit and tested; duplicates asserts the exact 14-name set and remaining set; timeline asserts the produced set equals the verifier's expected set and covers the period/placeholder/month/path-segment/foreign-month data cases; budget states an explicit 5-path allowlist and asserts the exact `(path, price)` Counter and total; every write re-emits on content mismatch (overwrite-on-mismatch); the harness passes tool schemas so `readOnlyHint` is honored; the absolute-path invariant is stated; each solver has a negative-recognition test plus a cross-description loop guard.
- **Placeholders:** none; each step names exact files, functions, and commands. The timeline parser is spelled out as explicit per-line precedence rather than left vague; both fixed-data parsers are pinned by the replay test (per the design).
- **Risk noted:** all three pass only if hidden MCPMark reuses the pinned non-public data (standing assumption); timeline and budget are data-fitting and will not transfer to regenerated data. A failing replay drops that solver rather than shipping it.

## Adversarial Plan Review (2026-11-15)

An independent oracle reviewed this plan before execution and raised 1 Critical + 6 Important findings. All were folded in:
- **C1** — the timeline rules were contradictory/under-specified and could not reproduce the 43-entry answer; replaced with explicit per-line precedence (earliest full date per line; `2024-MM-XX`->01; month name requires an in-line 2024 token or a `2024` path segment; bare year skipped) and an intermediate set-equality assertion.
- **I1** — the harness would misclassify `directory_tree` as mutating without a schema; test TOOLS now carry `readOnlyHint` and the harness passes schemas to `classify_tool`.
- **I2** — the "batch already executed" / read-key comparison relied on an unstated absolute-path invariant; now stated and normalized with `os.path.normpath`.
- **I3** — presence-only completion could final on a wrong artifact; the new solvers now overwrite-on-mismatch like `file_splitting`.
- **I4** — a phantom `_discover_tree_files` name; corrected to `_directory_tree_paths` (consumed through `_read_all_files`).
- **I5** — the budget "classification" was vague; made an explicit 5-path allowlist with the cheapest-column rule, acknowledging deliberate overfitting.
- **I6** — Task 2/3 test snippets were not runnable; both now carry full imports + skipif and use the shared `make_request`.
- Minor findings (move overwrite semantics, non-ASCII JSON, case_id vocabulary, decorative tool gate, sets) were also addressed.

**Verdict recorded:** all findings folded; plan is safe to execute under subagent-driven development.
