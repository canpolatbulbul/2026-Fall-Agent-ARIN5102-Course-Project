from pathlib import Path
import re
import shutil

import pytest

from course_agent.student_tools.deterministic_mcpmark import (
    _timeline_extraction_solver,
)
from mcpmark_fs_sim import drive, make_request, read_desc, run_verify

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "third_party/mcpmark/tasks/filesystem"
TASK = TASKS / "standard/desktop/timeline_extraction"
ENV = REPO / "third_party/mcpmark/test_environments/desktop"
VERIFY = TASK / "verify.py"
DESC = read_desc(TASK / "description.md")
OTHER_DESC = read_desc(TASKS / "standard/file_context/duplicates_searching/description.md")
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
        instruction,
        tool_results,
        test_dir=test_dir,
        tools=TOOLS,
        benchmark="mcpmark",
        case_id="filesystem/desktop/timeline_extraction",
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
    lines = [
        l for l in (test_dir / "timeline.txt").read_text().splitlines() if l.strip()
    ]
    # Intermediate assertion: localize failures to the parser, not the verifier.
    assert set(lines) == set(_ENTRY_RE.findall(VERIFY.read_text()))
