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
