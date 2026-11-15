import pytest

from course_agent.protocol import AgentRequest
from course_agent.runtime import LLMResult
from course_agent.student_agent import StudentAgent
from course_agent.student_tools import SanitizedToolCall
from course_agent.student_tools.deterministic_mcpmark import (
    _emitted_call_failed_before,
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
