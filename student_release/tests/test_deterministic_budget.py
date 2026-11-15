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
