import os
from pathlib import Path

from course_agent.protocol import AgentRequest
from course_agent.runtime import LLMResult
from course_agent.student_agent import StudentAgent
from course_agent.student_tools import SanitizedToolCall
from course_agent.student_tools.deterministic_mcpmark import (
    _emitted_call_failed_before,
    _file_splitting_solver,
    _has_read_basename,
    _parse_split_count,
    _split_n,
    _split_three,
    plan_deterministic_mcpmark,
)


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


# ---------------------------------------------------------------------------
# Guard helper: _emitted_call_failed_before
#
# Task 1 direct unit coverage per the controller ruling (the end-to-end
# plan_deterministic_mcpmark guard veto test moves to Task 2, where a solver
# exists to emit a decision for the guard to veto).
# ---------------------------------------------------------------------------


def _failed_read_result(*, path="/root/work/large_file.txt"):
    return [
        {
            "tool_name": "read_text_file",
            "ok": False,
            "result": None,
            "tool_arguments": {"path": path},
        }
    ]


def test_guard_matches_ok_false_failure():
    request = _req(tool_results=_failed_read_result())
    call = SanitizedToolCall(name="read_text_file", arguments={"path": "/root/work/large_file.txt"})
    assert _emitted_call_failed_before(request, {}, call) is True


def test_guard_matches_skipped_failure():
    tool_results = [
        {
            "tool_name": "read_text_file",
            "ok": True,
            "skipped": True,
            "result": None,
            "tool_arguments": {"path": "/root/work/large_file.txt"},
        }
    ]
    request = _req(tool_results=tool_results)
    call = SanitizedToolCall(name="read_text_file", arguments={"path": "/root/work/large_file.txt"})
    assert _emitted_call_failed_before(request, {}, call) is True


def test_guard_matches_is_error_failure():
    tool_results = [
        {
            "tool_name": "read_text_file",
            "ok": True,
            "result": {"isError": True, "content": [{"text": "boom"}]},
            "tool_arguments": {"path": "/root/work/large_file.txt"},
        }
    ]
    request = _req(tool_results=tool_results)
    call = SanitizedToolCall(name="read_text_file", arguments={"path": "/root/work/large_file.txt"})
    assert _emitted_call_failed_before(request, {}, call) is True


def test_guard_matches_error_prefix_text_failure():
    tool_results = [
        {
            "tool_name": "read_text_file",
            "ok": True,
            "result": {"isError": False, "content": [{"text": "  Error: file not found"}]},
            "tool_arguments": {"path": "/root/work/large_file.txt"},
        }
    ]
    request = _req(tool_results=tool_results)
    call = SanitizedToolCall(name="read_text_file", arguments={"path": "/root/work/large_file.txt"})
    assert _emitted_call_failed_before(request, {}, call) is True


def test_guard_ignores_failed_call_with_different_name():
    request = _req(tool_results=_failed_read_result())
    call = SanitizedToolCall(name="write_file", arguments={"path": "/root/work/large_file.txt"})
    assert _emitted_call_failed_before(request, {}, call) is False


def test_guard_ignores_failed_call_with_different_arguments():
    request = _req(tool_results=_failed_read_result())
    call = SanitizedToolCall(name="read_text_file", arguments={"path": "/root/work/other.txt"})
    assert _emitted_call_failed_before(request, {}, call) is False


def test_guard_ignores_successful_call():
    tool_results = [
        {
            "tool_name": "read_text_file",
            "ok": True,
            "result": {"isError": False, "content": [{"text": "file body"}]},
            "tool_arguments": {"path": "/root/work/large_file.txt"},
        }
    ]
    request = _req(tool_results=tool_results)
    call = SanitizedToolCall(name="read_text_file", arguments={"path": "/root/work/large_file.txt"})
    assert _emitted_call_failed_before(request, {}, call) is False


def test_guard_ignores_empty_tool_results():
    request = _req(tool_results=[])
    call = SanitizedToolCall(name="read_text_file", arguments={"path": "/root/work/large_file.txt"})
    assert _emitted_call_failed_before(request, {}, call) is False


# ---------------------------------------------------------------------------
# Wiring: an unrecognized mcpmark task must leave the LLM path untouched.
# ---------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def chat(self, messages, tools=None, tool_choice="auto", limits=None):
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice, "limits": limits})
        return self.result


def test_unrecognized_mcpmark_request_uses_llm_final(monkeypatch):
    monkeypatch.setenv("COURSE_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    llm = FakeLLM(LLMResult(content="All done via LLM.", tool_calls=[]))
    agent = StudentAgent(llm_client=llm)

    response = agent.solve_mcpmark(_req())

    assert response.kind == "final"
    assert response.content == "All done via LLM."
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# Task 2: deterministic file_splitting solver
# ---------------------------------------------------------------------------

TASKS = Path(__file__).resolve().parents[2] / "third_party/mcpmark/tasks/filesystem"

SPLIT_INSTRUCTION = "You need to split large_file.txt into exactly 3 files with similar character counts."
LARGE_PATH = "/root/work/large_file.txt"
SPLIT_DIR = "/root/work/split"


def _desc(rel):
    p = TASKS / rel
    return p.read_text() if p.exists() else ""


def _read_result(*, path=LARGE_PATH, text="abcdefghij"):
    return {
        "tool_name": "read_text_file",
        "ok": True,
        "result": {"isError": False, "content": [{"text": text}]},
        "tool_arguments": {"path": path},
    }


def _create_dir_result(*, path=SPLIT_DIR, ok=True):
    return {
        "tool_name": "create_directory",
        "ok": ok,
        "result": {"isError": False, "content": [{"text": ""}]},
        "tool_arguments": {"path": path},
    }


def _write_result(*, path, content, ok=True):
    return {
        "tool_name": "write_file",
        "ok": ok,
        "result": {"isError": False, "content": [{"text": ""}]},
        "tool_arguments": {"path": path, "content": content},
    }


def test_split_three_preserves_content_and_balances():
    s = "abcdefghij"  # 10 chars
    parts = _split_three(s)
    assert "".join(parts) == s
    assert max(map(len, parts)) - min(map(len, parts)) <= 1
    assert len(parts) == 3


def test_recognition_uses_real_descriptions():
    split_desc = _desc("easy/file_context/file_splitting/description.md")
    pat_desc = _desc("easy/file_context/pattern_matching/description.md")
    assert "large_file.txt" in split_desc and "split" in split_desc
    assert "large_file.txt" in pat_desc and "split" not in pat_desc.lower()


def test_has_read_basename_true_for_read_result_by_basename():
    request = _req(tool_results=[_read_result()])
    assert _has_read_basename(request, "large_file.txt") is True


def test_has_read_basename_false_when_absent():
    assert _has_read_basename(_req(tool_results=[]), "large_file.txt") is False


def test_solver_emits_read_when_file_not_yet_read_and_no_failed_read():
    req = _req(instruction=SPLIT_INSTRUCTION)
    d = _file_splitting_solver(req, {})
    assert d is not None and len(d.calls) == 1 and d.calls[0].name == "read_text_file"
    assert d.calls[0].arguments == {"path": LARGE_PATH}


def test_solver_not_triggered_for_pattern_matching_instruction():
    pat_desc = _desc("easy/file_context/pattern_matching/description.md")
    req = _req(instruction=pat_desc or "read large_file.txt and find common substrings")
    assert _file_splitting_solver(req, {}) is None


def test_solver_emits_create_directory_after_read():
    req = _req(instruction=SPLIT_INSTRUCTION, tool_results=[_read_result()])
    d = _file_splitting_solver(req, {})
    assert d is not None and len(d.calls) == 1 and d.calls[0].name == "create_directory"
    assert d.calls[0].arguments == {"path": SPLIT_DIR}


def test_solver_emits_first_write_when_missing():
    parts = _split_three("abcdefghij")
    req = _req(instruction=SPLIT_INSTRUCTION, tool_results=[_read_result(), _create_dir_result()])
    d = _file_splitting_solver(req, {})
    assert d is not None and len(d.calls) == 1 and d.calls[0].name == "write_file"
    assert d.calls[0].arguments == {
        "path": os.path.join(SPLIT_DIR, "split_01.txt"),
        "content": parts[0],
    }


def test_solver_reeemits_write_when_recorded_content_differs():
    parts = _split_three("abcdefghij")
    wrong = [
        _write_result(path=os.path.join(SPLIT_DIR, "split_01.txt"), content="WRONG"),
    ]
    req = _req(
        instruction=SPLIT_INSTRUCTION,
        tool_results=[_read_result(), _create_dir_result()] + wrong,
    )
    d = _file_splitting_solver(req, {})
    assert d is not None and d.calls[0].name == "write_file"
    assert d.calls[0].arguments == {
        "path": os.path.join(SPLIT_DIR, "split_01.txt"),
        "content": parts[0],
    }


def test_solver_proceeds_to_second_write_when_first_matches():
    parts = _split_three("abcdefghij")
    done = [_write_result(path=os.path.join(SPLIT_DIR, "split_01.txt"), content=parts[0])]
    req = _req(
        instruction=SPLIT_INSTRUCTION,
        tool_results=[_read_result(), _create_dir_result()] + done,
    )
    d = _file_splitting_solver(req, {})
    assert d is not None and d.calls[0].name == "write_file"
    assert d.calls[0].arguments == {
        "path": os.path.join(SPLIT_DIR, "split_02.txt"),
        "content": parts[1],
    }


def test_solver_is_final_when_all_three_recorded_contents_match():
    parts = _split_three("abcdefghij")
    done = [
        _write_result(path=os.path.join(SPLIT_DIR, f"split_0{n}.txt"), content=parts[n - 1])
        for n in range(1, 4)
    ]
    req = _req(
        instruction=SPLIT_INSTRUCTION,
        tool_results=[_read_result(), _create_dir_result()] + done,
    )
    d = _file_splitting_solver(req, {})
    assert d is not None and d.is_final is True and d.calls == []


# ---------------------------------------------------------------------------
# Task 2 controller ruling: generalize to N files (3-file and 10-file variants).
# ---------------------------------------------------------------------------


def test_parse_split_count_from_exactly_n_files():
    assert _parse_split_count("split large_file.txt into exactly 3 files") == 3
    assert _parse_split_count("into exactly 10 files with equal character counts") == 10


def test_parse_split_count_from_enumeration():
    assert _parse_split_count("split_01.txt, split_02.txt, ..., split_10.txt") == 10


def test_parse_split_count_returns_none_when_absent():
    assert _parse_split_count("split large_file.txt into multiple smaller files") is None


def test_parse_split_count_returns_none_when_signals_disagree():
    assert _parse_split_count("exactly 3 files named split_01.txt ... split_10.txt") is None


def test_split_n_preserves_content_and_is_near_equal():
    s = "abcdefghij"  # 10 chars
    parts = _split_n(s, 10)
    assert "".join(parts) == s
    assert len(parts) == 10
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_solver_recognizes_standard_ten_file_variant():
    instr = _desc("standard/file_context/file_splitting/description.md")
    if not instr:
        instr = (
            "Split large_file.txt into exactly 10 files with equal character counts. "
            "Name the files as split_01.txt, split_02.txt, ..., split_10.txt."
        )
    assert _parse_split_count(instr) == 10

    # First turn: read the source file.
    d = _file_splitting_solver(_req(instruction=instr), {})
    assert d is not None and d.calls[0].name == "read_text_file"

    # After read + create_directory: the first write is split_01.txt.
    d = _file_splitting_solver(
        _req(instruction=instr, tool_results=[_read_result(), _create_dir_result()]), {}
    )
    assert d is not None and d.calls[0].name == "write_file"
    assert d.calls[0].arguments["path"] == os.path.join(SPLIT_DIR, "split_01.txt")


def test_solver_ten_file_is_final_when_all_ten_recorded_contents_match():
    instr = _desc("standard/file_context/file_splitting/description.md")
    if not instr:
        instr = "large_file.txt split into exactly 10 files split_01.txt ... split_10.txt"
    parts = _split_n("abcdefghij", 10)
    done = [
        _write_result(path=os.path.join(SPLIT_DIR, f"split_{n:02d}.txt"), content=parts[n - 1])
        for n in range(1, 11)
    ]
    req = _req(instruction=instr, tool_results=[_read_result(), _create_dir_result()] + done)
    d = _file_splitting_solver(req, {})
    assert d is not None and d.is_final is True and d.calls == []


def test_solver_returns_none_for_ambiguous_or_absent_count():
    instr = "Please split large_file.txt into smaller files in a split directory."
    assert _file_splitting_solver(_req(instruction=instr), {}) is None


# ---------------------------------------------------------------------------
# Task 2 extra requirement: end-to-end framework guard veto.
#
# When the solver would emit a call identical (name + equal tool_arguments) to
# one already present in tool_results with a failed result, the framework guard
# must veto it: plan_deterministic_mcpmark returns None and latches
# state["det_disabled"] = True. Task 1 deferred this test until a solver existed.
# ---------------------------------------------------------------------------


def test_guard_vetoes_solver_when_emitted_read_already_failed():
    failed_read = [
        {
            "tool_name": "read_text_file",
            "ok": False,
            "result": None,
            "tool_arguments": {"path": LARGE_PATH},
        }
    ]
    req = _req(instruction=SPLIT_INSTRUCTION, tool_results=failed_read)
    state: dict = {}
    assert plan_deterministic_mcpmark(req, state) is None
    assert state["det_disabled"] is True


# ---------------------------------------------------------------------------
# Phase B review fixes: ok:True + "Error"-prefixed results (postgres style) and
# successful-but-empty reads must not be mistaken for real content/writes and
# must not spin the solver to the turn limit. In every case the solver hands
# back to the LLM (returns None and latches state["det_disabled"]).
# ---------------------------------------------------------------------------

# A read that reports ok:true but whose extracted text begins with "Error"
# (the canonical postgres-style failure the shared predicate catches).
_ERROR_READ_RESULT = {
    "tool_name": "read_text_file",
    "ok": True,
    "result": {"isError": False, "content": [{"text": "Error: file not found"}]},
    "tool_arguments": {"path": LARGE_PATH},
}


def test_error_prefixed_read_is_not_treated_as_content_but_handed_back():
    req = _req(instruction=SPLIT_INSTRUCTION, tool_results=[_ERROR_READ_RESULT])
    state: dict = {}
    assert plan_deterministic_mcpmark(req, state) is None
    assert state["det_disabled"] is True


def test_error_prefixed_write_is_not_treated_as_done_but_handed_back():
    parts = _split_three("abcdefghij")
    error_write = {
        "tool_name": "write_file",
        "ok": True,
        "result": {"isError": False, "content": [{"text": "Error: disk full"}]},
        "tool_arguments": {
            "path": os.path.join(SPLIT_DIR, "split_01.txt"),
            "content": parts[0],
        },
    }
    req = _req(
        instruction=SPLIT_INSTRUCTION,
        tool_results=[_read_result(), _create_dir_result(), error_write],
    )
    state: dict = {}
    assert plan_deterministic_mcpmark(req, state) is None
    assert state["det_disabled"] is True


def test_successful_but_empty_read_hands_back_instead_of_reemitting():
    empty_read = {
        "tool_name": "read_text_file",
        "ok": True,
        "result": {"isError": False, "content": [{"text": ""}]},
        "tool_arguments": {"path": LARGE_PATH},
    }
    req = _req(instruction=SPLIT_INSTRUCTION, tool_results=[empty_read])
    state: dict = {}
    assert plan_deterministic_mcpmark(req, state) is None
    assert state["det_disabled"] is True


# ---------------------------------------------------------------------------
# Recognized file_splitting task driven through the REAL hook
# (StudentAgent.solve_mcpmark -> _react_step -> plan_deterministic_mcpmark).
# The direct-solver integration tests never exercise this wiring.
# ---------------------------------------------------------------------------


def _hook_request(*, instruction=SPLIT_INSTRUCTION, tool_results=None):
    return AgentRequest.from_dict(
        {
            "request_id": "r",
            "context_id": "c",
            "benchmark": "mcpmark",
            "case_id": "filesystem/easy/file_context/file_splitting",
            "step": 0,
            "text": instruction,
            "observation": {
                "test_directory": "/root/work",
                "instruction": instruction,
                "tool_results": tool_results or [],
            },
            "tools": [
                {"name": "read_text_file"},
                {"name": "write_file"},
                {"name": "create_directory"},
            ],
            "limits": {
                "max_tool_calls_per_turn": 4,
                "max_mutating_tool_calls_per_turn": 1,
            },
        }
    )


def test_hook_recognized_split_emits_read_text_file(monkeypatch):
    monkeypatch.setenv("COURSE_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    llm = FakeLLM(LLMResult(content="LLM would answer", tool_calls=[]))
    agent = StudentAgent(llm_client=llm)

    response = agent.solve_mcpmark(_hook_request())

    assert response.kind == "tool_call"
    assert response.tool_name == "read_text_file"
    assert response.tool_arguments == {"path": LARGE_PATH}
    assert llm.calls == []  # recognized path must not consult the LLM


def test_hook_recognized_split_returns_final_when_all_writes_recorded(monkeypatch):
    monkeypatch.setenv("COURSE_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    parts = _split_three("abcdefghij")
    results = [_read_result(), _create_dir_result()] + [
        _write_result(path=os.path.join(SPLIT_DIR, f"split_0{n}.txt"), content=parts[n - 1])
        for n in range(1, 4)
    ]
    llm = FakeLLM(LLMResult(content="LLM would answer", tool_calls=[]))
    agent = StudentAgent(llm_client=llm)

    response = agent.solve_mcpmark(_hook_request(tool_results=results))

    assert response.kind == "final"
