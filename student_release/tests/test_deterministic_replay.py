"""Cross-turn integration test for the deterministic MCPMark hook (Task 3).

Two independent proofs live here:

1. ``plan_deterministic_mcpmark`` is driven across turns with synthetic
   ``tool_results`` fed back exactly as the runner would. The emitted call
   sequence is asserted to be ``read_text_file`` -> ``create_directory`` ->
   ``write_file`` split_01..N, every batch is checked against the per-turn
   tool policy (<=1 mutating call, reads before mutations, ``sanitize_tool_calls``
   leaves the batch unchanged), and ``is_final`` is only asserted once every
   part is recorded as correct. A variant records ``split_01.txt`` with WRONG
   content and asserts the solver re-emits that write (not ``None``).

2. An offline replay writes ``_split_n`` output into a temp copy of the real
   MCPMark test data and runs the task's real ``verify.py`` as a subprocess.
   Both the easy (3-file) and standard (10-file) verifiers are exercised.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from course_agent.protocol import AgentRequest
from course_agent.student_tools.deterministic_mcpmark import (
    _split_n,
    _split_three,
    plan_deterministic_mcpmark,
)
from course_agent.student_tools.policy_guard import (
    SanitizedToolCall,
    classify_tool,
    sanitize_tool_calls,
)

REPO = Path(__file__).resolve().parents[2]
TEST_ENV = REPO / "third_party/mcpmark/test_environments/file_context"
EASY_VERIFY = (
    REPO
    / "third_party/mcpmark/tasks/filesystem/easy/file_context/file_splitting/verify.py"
)
STANDARD_VERIFY = (
    REPO
    / "third_party/mcpmark/tasks/filesystem/standard/file_context/file_splitting/verify.py"
)

pytestmark = pytest.mark.skipif(
    not TEST_ENV.exists(),
    reason="mcpmark test data missing",
)


# ---------------------------------------------------------------------------
# Part 1: cross-turn integration through the hook
# ---------------------------------------------------------------------------

TEST_DIR = "/root/work"
LARGE_PATH = os.path.join(TEST_DIR, "large_file.txt")
SPLIT_DIR = os.path.join(TEST_DIR, "split")
SOURCE_CONTENT = "abcdefghijklmnopqrstuvwxyz0123"  # 30 chars -> 3 x 10

TOOLS = [
    {"name": "read_text_file"},
    {"name": "write_file"},
    {"name": "create_directory"},
]
TOOL_NAMES = {t["name"] for t in TOOLS}

_EASY_INSTRUCTION = (
    "You need to split large_file.txt into exactly 3 files with similar "
    "character counts. Name the files split_01.txt, split_02.txt, split_03.txt."
)
_TEN_INSTRUCTION = (
    "You need to split a large text file into multiple smaller files with equal "
    "character counts. Split large_file.txt into exactly 10 files. Name the "
    "files as split_01.txt, split_02.txt, ..., split_10.txt in the split directory."
)


def _request(instruction: str, tool_results: list[dict]) -> AgentRequest:
    return AgentRequest.from_dict(
        {
            "request_id": "r",
            "context_id": "c",
            "benchmark": "mcpmark",
            "case_id": "filesystem/easy/file_context/file_splitting",
            "step": 0,
            "text": instruction,
            "observation": {
                "test_directory": TEST_DIR,
                "instruction": instruction,
                "tool_results": tool_results,
            },
            "tools": TOOLS,
            "limits": {"max_mutating_tool_calls_per_turn": 1, "max_tool_calls_per_turn": 4},
        }
    )


def _result_for(call: SanitizedToolCall, *, write_content: str | None = None) -> dict:
    """Build a synthetic successful tool_result for ``call``.

    ``write_content`` overrides the recorded content for ``write_file`` (used to
    simulate a part written with the wrong bytes).
    """
    if call.name == "read_text_file":
        return {
            "tool_name": "read_text_file",
            "ok": True,
            "result": {"isError": False, "content": [{"text": SOURCE_CONTENT}]},
            "tool_arguments": dict(call.arguments),
        }
    if call.name == "create_directory":
        return {
            "tool_name": "create_directory",
            "ok": True,
            "result": {"isError": False, "content": [{"text": ""}]},
            "tool_arguments": dict(call.arguments),
        }
    if call.name == "write_file":
        content = write_content if write_content is not None else call.arguments["content"]
        return {
            "tool_name": "write_file",
            "ok": True,
            "result": {"isError": False, "content": [{"text": ""}]},
            "tool_arguments": {"path": call.arguments["path"], "content": content},
        }
    raise AssertionError(f"unexpected tool {call.name!r}")


def _assert_batch_obeys_policy(batch: list[SanitizedToolCall]) -> None:
    # Every call targets a tool offered this turn.
    assert all(c.name in TOOL_NAMES for c in batch)
    # The policy sanitizer must leave the deterministic batch unchanged.
    sanitized = sanitize_tool_calls(
        benchmark="mcpmark",
        raw_calls=[c.to_protocol_dict() for c in batch],
        available_tools=TOOLS,
        limits={"max_mutating_tool_calls_per_turn": 1, "max_tool_calls_per_turn": 4},
    )
    assert sanitized == batch
    # At most one mutating call per batch.
    mutating = [c for c in batch if classify_tool("mcpmark", c.name) == "mutating"]
    assert len(mutating) <= 1
    # Reads must precede any mutation.
    seen_mutating = False
    for call in batch:
        kind = classify_tool("mcpmark", call.name)
        if kind == "mutating":
            seen_mutating = True
        else:
            assert not seen_mutating, "read call emitted after a mutating call"


def _run_turns(n: int, *, corrupt_first_write: bool = False):
    """Drive the hook across turns, feeding back simulated results.

    Returns ``(emitted, final_step)`` where ``emitted`` is the flat list of
    emitted call names and ``final_step`` is the turn index at which the hook
    first reported ``is_final``.
    """
    instruction = {3: _EASY_INSTRUCTION, 10: _TEN_INSTRUCTION}[n]
    results: list[dict] = []
    emitted: list[str] = []
    final_step: int | None = None
    corrupt_used = False

    for step in range(50):
        decision = plan_deterministic_mcpmark(_request(instruction, results), {})
        assert decision is not None, f"hook returned None at step {step}"
        if decision.is_final:
            final_step = step
            break
        assert decision.calls
        _assert_batch_obeys_policy(decision.calls)
        for call in decision.calls:
            emitted.append(call.name)
            write_content = None
            if (
                corrupt_first_write
                and not corrupt_used
                and call.name == "write_file"
            ):
                write_content = "WRONG"
                corrupt_used = True
            results.append(_result_for(call, write_content=write_content))
    else:  # pragma: no cover - defensive
        raise AssertionError("hook never reached is_final")

    assert final_step is not None
    # ``is_final`` must not be reported before every part is recorded correct.
    assert len(results) >= n + 2
    return emitted, final_step


@pytest.mark.parametrize("n", [3, 10])
def test_integration_emits_exact_sequence_and_obeys_policy(n):
    emitted, final_step = _run_turns(n)

    expected = ["read_text_file", "create_directory"] + ["write_file"] * n
    assert emitted == expected
    # Final only after read + create + all n writes were recorded.
    assert final_step == len(expected)


def test_integration_is_not_final_until_all_parts_correct():
    n = 3
    parts = _split_n(SOURCE_CONTENT, n)
    # read + create + only 2 of 3 correct writes: still not final.
    partial = [
        _result_for(SanitizedToolCall("read_text_file", {"path": LARGE_PATH})),
        _result_for(SanitizedToolCall("create_directory", {"path": SPLIT_DIR})),
        _result_for(
            SanitizedToolCall(
                "write_file",
                {"path": os.path.join(SPLIT_DIR, "split_01.txt"), "content": parts[0]},
            )
        ),
        _result_for(
            SanitizedToolCall(
                "write_file",
                {"path": os.path.join(SPLIT_DIR, "split_02.txt"), "content": parts[1]},
            )
        ),
    ]
    decision = plan_deterministic_mcpmark(_request(_EASY_INSTRUCTION, partial), {})
    assert decision is not None and decision.is_final is False
    assert decision.calls[0].name == "write_file"
    assert decision.calls[0].arguments["path"] == os.path.join(SPLIT_DIR, "split_03.txt")


def test_integration_reeemits_write_when_a_part_has_wrong_content():
    n = 3
    parts = _split_n(SOURCE_CONTENT, n)
    results = [
        _result_for(SanitizedToolCall("read_text_file", {"path": LARGE_PATH})),
        _result_for(SanitizedToolCall("create_directory", {"path": SPLIT_DIR})),
        _result_for(
            SanitizedToolCall(
                "write_file",
                {"path": os.path.join(SPLIT_DIR, "split_01.txt"), "content": parts[0]},
            ),
            write_content="WRONG",  # recorded content differs from computed part
        ),
    ]
    decision = plan_deterministic_mcpmark(_request(_EASY_INSTRUCTION, results), {})
    assert decision is not None
    assert decision.is_final is False
    assert decision.calls[0].name == "write_file"
    assert decision.calls[0].arguments["path"] == os.path.join(SPLIT_DIR, "split_01.txt")
    assert decision.calls[0].arguments["content"] == parts[0]


# ---------------------------------------------------------------------------
# Part 2: offline replay through the real task verifier
# ---------------------------------------------------------------------------


def _replay_and_verify(n: int, verify: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    test_dir = tmp_path / "test_directory"
    shutil.copytree(TEST_ENV, test_dir)
    content = (test_dir / "large_file.txt").read_text()
    split_dir = test_dir / "split"
    split_dir.mkdir()
    for i, part in enumerate(_split_n(content, n), start=1):
        (split_dir / f"split_{i:02d}.txt").write_text(part)
    env = {**os.environ, "FILESYSTEM_TEST_DIR": str(test_dir)}
    return subprocess.run(
        [sys.executable, str(verify)], env=env, capture_output=True, text=True
    )


@pytest.mark.skipif(not EASY_VERIFY.exists(), reason="easy verify.py missing")
def test_split_three_result_passes_real_easy_verifier(tmp_path):
    r = _replay_and_verify(3, EASY_VERIFY, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not STANDARD_VERIFY.exists(), reason="standard verify.py missing")
def test_split_n_ten_result_passes_real_standard_verifier(tmp_path):
    r = _replay_and_verify(10, STANDARD_VERIFY, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not EASY_VERIFY.exists(), reason="easy verify.py missing")
def test_replay_uses_split_three_helper(tmp_path):
    """Guard: the easy replay path is the documented ``_split_three`` wrapper."""
    test_dir = tmp_path / "test_directory"
    shutil.copytree(TEST_ENV, test_dir)
    content = (test_dir / "large_file.txt").read_text()
    assert _split_three(content) == _split_n(content, 3)
