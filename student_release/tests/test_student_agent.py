import inspect
import time

import pytest

from course_agent.protocol import AgentRequest
from course_agent.runtime import LLMResult, LLMToolCall

from course_agent.student_agent import StudentAgent, _classify_llm_error
from course_agent.student_tools import (
    SanitizedToolCall,
    normalize_mcpmark_read_artifact,
    summarize_mcpmark_mutations,
)


class RateLimitError(Exception):
    """Stand-in for ``openai.RateLimitError`` (type-name detection)."""


class FakeLLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def chat(self, messages, tools=None, tool_choice="auto", limits=None):
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice, "limits": limits})
        return self.result


def _tau_request(tools):
    return AgentRequest.from_dict(
        {
            "request_id": "req-1",
            "context_id": "ctx-1",
            "benchmark": "tau2_airline",
            "case_id": "3",
            "step": 0,
            "text": "I need to change my flight.",
            "observation": {"message_type": "user", "content": "I need to change my flight."},
            "tools": tools,
            "limits": {"max_agent_turns": 100, "model_input_tokens": 64000},
        }
    )


def _deepplanning_request(tools, observation=None):
    return AgentRequest.from_dict(
        {
            "request_id": "req-dp",
            "context_id": "ctx-dp",
            "benchmark": "deepplanning",
            "case_id": "shopping/level_1/case_12",
            "step": 0,
            "text": "Find suitable products.",
            "observation": observation or {"query": "Find suitable products.", "cart": {"items": []}},
            "tools": tools,
            "limits": {"max_model_calls": 100, "model_input_tokens": 64000},
        }
    )


def _mcpmark_request(tools, observation=None):
    return AgentRequest.from_dict(
        {
            "request_id": "req-mcp",
            "context_id": "ctx-mcp",
            "benchmark": "mcpmark",
            "case_id": "filesystem/desktop/timeline_extraction",
            "step": 0,
            "text": "Read the files and answer the task.",
            "observation": observation or {"task": "Read the files and answer the task."},
            "tools": tools,
            "limits": {"max_model_calls": 100, "model_input_tokens": 64000},
        }
    )


def _clear_llm_env(monkeypatch):
    for name in (
        "COURSE_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_student_agent_template_exposes_locked_solver_contract():
    for method_name in ("solve_mcpmark", "solve_tau2_airline", "solve_deepplanning"):
        signature = inspect.signature(getattr(StudentAgent, method_name))
        assert list(signature.parameters) == ["self", "request"]


def test_student_agent_template_returns_valid_response_for_each_benchmark():
    agent = StudentAgent(llm_client=FakeLLM(LLMResult(content="ok", tool_calls=[])))

    responses = [
        agent.respond(_mcpmark_request([])),
        agent.respond(_tau_request([])),
        agent.respond(_deepplanning_request([])),
    ]

    assert [response.kind for response in responses] == ["final", "final", "final"]
    assert all(response.request_id for response in responses)


def test_student_agent_template_keeps_lazy_llm_client_hook():
    llm = FakeLLM(LLMResult(content="ok", tool_calls=[]))
    agent = StudentAgent(llm_client=llm)

    assert agent.llm_client is llm


def test_student_agent_template_has_clear_release_sections():
    source = inspect.getsource(StudentAgent)

    assert "##### LOCKED COURSE INTERFACE" in source
    assert "##### RESPONSE SCHEMA REQUIREMENTS" in source
    assert "##### OPTIONAL COURSE HELPERS" in source
    assert "##### STUDENT-DESIGNED BENCHMARK LOGIC" in source
    assert "##### ADD YOUR OWN HELPERS" in source


class RaisingLLM:
    """LLM stub that raises a queued sequence of errors, then returns ``result``."""

    def __init__(self, errors, result=None):
        self.errors = list(errors)
        self.result = result
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", limits=None):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def _wrapped_transient_wrapper() -> RuntimeError:
    """Mimic CourseLLMClient: generic RuntimeError chained from a real RateLimitError."""
    inner = RateLimitError("429 Too Many Requests")
    wrapper = RuntimeError("LLM chat failed after 2 attempt(s)")
    wrapper.__cause__ = inner
    return wrapper


def test_classify_llm_error_detects_hard_transient_and_unknown():
    assert _classify_llm_error(_wrapped_transient_wrapper()) == "transient"
    assert _classify_llm_error(RuntimeError("LLM call limit exceeded: 100")) == "hard"
    assert _classify_llm_error(RuntimeError("LLM input exceeds limit: estimated 70000 > 64000")) == "hard"
    assert _classify_llm_error(RuntimeError("something totally unexpected")) == "unknown"


def test_backoff_retries_transient_error_and_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda secs: sleeps.append(secs))

    result = LLMResult(content="ok", tool_calls=[])
    llm = RaisingLLM([_wrapped_transient_wrapper()], result=result)
    agent = StudentAgent(llm_client=llm)

    out = agent._call_llm_with_backoff(messages=[], tools=None, tool_choice="auto", limits={})

    assert out is result
    assert llm.calls == 2
    assert sleeps == [3.5]  # one back-off before the successful retry


def test_backoff_reraises_hard_cap_without_retrying(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    llm = RaisingLLM([RuntimeError("LLM call limit exceeded: 100")])
    agent = StudentAgent(llm_client=llm)

    with pytest.raises(RuntimeError, match="call limit exceeded"):
        agent._call_llm_with_backoff(messages=[], tools=None, tool_choice="auto", limits={})

    assert llm.calls == 1  # no retry for a hard local cap


def test_graceful_fallback_returns_final_on_hard_cap_for_all_benchmarks():
    agent = StudentAgent(llm_client=FakeLLM(LLMResult(content="ok", tool_calls=[])))

    for request in (_mcpmark_request([]), _tau_request([]), _deepplanning_request([])):
        response = agent._graceful_fallback_on_llm_error(
            request, {"messages": []}, RuntimeError("LLM call limit exceeded: 100")
        )
        assert response.kind == "final"


def test_graceful_fallback_keeps_transient_tau2_respond_to_user():
    agent = StudentAgent(llm_client=FakeLLM(LLMResult(content="ok", tool_calls=[])))

    response = agent._graceful_fallback_on_llm_error(
        _tau_request([{"name": "respond_to_user"}]), {"messages": []}, _wrapped_transient_wrapper()
    )

    assert response.kind == "tool_call"


def _read_multiple_files_result(*pairs: tuple[str, str]) -> list[dict]:
    # Reproduce the server framing exactly: `${path}:\n${content}\n` joined by `\n---\n`.
    text = "\n---\n".join(f"{path}:\n{content}\n" for path, content in pairs)
    return [{"tool_name": "read_multiple_files", "result": {"content": [{"text": text}]}}]


def test_normalize_mcpmark_read_artifact_strips_display_newline_from_uppercase_write():
    # The real recorded pristine content is 144 bytes and ends in "." with NO trailing newline.
    orig = (
        "This is a sample text file for testing purposes.\n"
        "It contains basic information about file operations.\n"
        "The content is simple and straightforward."
    )
    tool_results = _read_multiple_files_result(("/root/file_01.txt", orig), ("/root/file_02.txt", "other"))

    call = SanitizedToolCall(
        name="write_file",
        arguments={"path": "/root/uppercase/file_01.txt", "content": orig.upper() + "\n"},
    )
    corrected = normalize_mcpmark_read_artifact([call], tool_results)

    assert corrected == 1
    assert call.arguments["content"] == orig.upper()  # trailing artifact byte removed


def test_normalize_mcpmark_read_artifact_keeps_a_real_trailing_newline():
    # File genuinely ends with "\n"; the model wrote exactly that -> must NOT be altered.
    orig = "abc\n"
    tool_results = _read_multiple_files_result(("/root/a.txt", orig))

    call = SanitizedToolCall(name="write_file", arguments={"path": "/root/a.txt", "content": "abc\n"})
    corrected = normalize_mcpmark_read_artifact([call], tool_results)

    assert corrected == 0
    assert call.arguments["content"] == "abc\n"


def test_normalize_mcpmark_read_artifact_ignores_unrelated_content():
    tool_results = _read_multiple_files_result(("/root/a.txt", "hello world"))

    call = SanitizedToolCall(
        name="write_file",
        arguments={"path": "/root/summary.txt", "content": "a totally new summary\n"},
    )
    corrected = normalize_mcpmark_read_artifact([call], tool_results)

    assert corrected == 0
    assert call.arguments["content"] == "a totally new summary\n"


def test_normalize_mcpmark_read_artifact_handles_write_multiple_files():
    orig = "line one\nline two"
    tool_results = _read_multiple_files_result(("/root/x.txt", orig))

    call = SanitizedToolCall(
        name="write_multiple_files",
        arguments={"files": [{"path": "/root/up/x.txt", "content": orig.upper() + "\n"}]},
    )
    corrected = normalize_mcpmark_read_artifact([call], tool_results)

    assert corrected == 1
    assert call.arguments["files"][0]["content"] == orig.upper()


class SequencedLLM:
    """Returns one predefined LLMResult per ``chat`` call, in order."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def chat(self, messages, tools=None, tool_choice="auto", limits=None):
        self.calls.append({"messages": messages, "tools": tools, "limits": limits})
        if not self.results:
            raise AssertionError("SequencedLLM ran out of results")
        return self.results.pop(0)


def _mcpmark_recovery_request(*, tool_results, step=0, limits=None, tools=None):
    return AgentRequest.from_dict(
        {
            "request_id": "req-mcp-guard",
            "context_id": "ctx-mcp-guard",
            "benchmark": "mcpmark",
            "case_id": "filesystem/standard/file_property/time_classification",
            "step": step,
            "text": "Organize the files as requested.",
            "observation": {
                "test_directory": "/root/work",
                "instruction": "Organize the files as requested.",
                "tool_results": tool_results,
            },
            "tools": tools or [{"name": "write_file"}, {"name": "create_directory"}, {"name": "list_directory"}],
            "limits": limits
            or {"max_agent_turns": 20, "model_input_tokens": 64000, "case_timeout_sec": 300},
        }
    )


def _failed_write_result():
    return [
        {
            "tool_name": "write_file",
            "ok": False,
            "result": None,
            "tool_arguments": {"path": "/root/work/out.txt", "content": "x"},
        }
    ]


def test_summarize_mcpmark_mutations_counts_only_mutating_calls():
    tool_results = [
        {"tool_name": "read_multiple_files", "ok": True, "result": {"content": [{"text": "x"}]}},
        {"tool_name": "list_directory", "ok": True, "result": {"content": [{"text": "[FILE] a"}]}},
        {"tool_name": "write_file", "ok": True, "result": {"content": [{"text": "Written /root/a.txt"}]}},
    ]
    summary = summarize_mcpmark_mutations(tool_results)
    assert summary["attempts"] == 1
    assert summary["successes"] == 1


def test_summarize_mcpmark_mutations_flags_ok_false_failure():
    summary = summarize_mcpmark_mutations([{"tool_name": "create_directory", "ok": False, "result": None}])
    assert summary["attempts"] == 1
    assert summary["successes"] == 0
    assert summary["last_failure_tool"] == "create_directory"


def test_summarize_mcpmark_mutations_flags_postgres_error_text():
    # postgres reports SQL errors as ok=true with text starting "Error:" -> must be a failure.
    item = {
        "tool_name": "execute_sql",
        "ok": True,
        "result": {"isError": False, "content": [{"text": 'Error: column "s1.amount" must appear in the GROUP BY clause'}]},
    }
    summary = summarize_mcpmark_mutations([item])
    assert summary["attempts"] == 1
    assert summary["successes"] == 0


def test_summarize_mcpmark_mutations_accepts_normal_rows():
    item = {
        "tool_name": "execute_sql",
        "ok": True,
        "result": {"isError": False, "content": [{"text": "id | name\n1 | a"}]},
    }
    assert summarize_mcpmark_mutations([item])["successes"] == 1


def test_summarize_mcpmark_mutations_ignores_error_substring():
    # A legitimate column named error_count must not be read as a failure (leading-token only).
    item = {
        "tool_name": "execute_sql",
        "ok": True,
        "result": {"isError": False, "content": [{"text": "id | error_count\n1 | 3"}]},
    }
    assert summarize_mcpmark_mutations([item])["successes"] == 1


def test_summarize_mcpmark_mutations_treats_skipped_as_failure():
    item = {"tool_name": "write_file", "ok": False, "skipped": True, "result": None}
    summary = summarize_mcpmark_mutations([item])
    assert summary["attempts"] == 1
    assert summary["successes"] == 0


def test_failure_recovery_guard_refires_after_failed_mutation():
    llm = SequencedLLM(
        [
            LLMResult(content="Done. I created the output file.", tool_calls=[]),
            LLMResult(
                content="",
                tool_calls=[LLMToolCall(name="write_file", arguments={"path": "/root/work/out.txt", "content": "x"})],
            ),
        ]
    )
    agent = StudentAgent(llm_client=llm)

    response = agent.solve_mcpmark(_mcpmark_recovery_request(tool_results=_failed_write_result()))

    assert response.kind == "tool_call"
    assert len(llm.calls) == 2  # original call + exactly one recovery call


def test_failure_recovery_guard_falls_through_to_original_final():
    llm = SequencedLLM(
        [
            LLMResult(content="Done. I created the output file.", tool_calls=[]),
            LLMResult(content="", tool_calls=[]),
        ]
    )
    agent = StudentAgent(llm_client=llm)

    response = agent.solve_mcpmark(_mcpmark_recovery_request(tool_results=_failed_write_result()))

    assert response.kind == "final"
    assert response.content == "Done. I created the output file."
    assert len(llm.calls) == 2


def test_failure_recovery_guard_does_not_fire_when_a_mutation_succeeded():
    llm = SequencedLLM([LLMResult(content="Done.", tool_calls=[])])
    agent = StudentAgent(llm_client=llm)
    tool_results = [
        {"tool_name": "create_directory", "ok": False, "result": None},
        {"tool_name": "write_file", "ok": True, "result": {"content": [{"text": "Written"}]}},
    ]

    response = agent.solve_mcpmark(_mcpmark_recovery_request(tool_results=tool_results))

    assert response.kind == "final"
    assert len(llm.calls) == 1  # no recovery call


def test_failure_recovery_guard_does_not_fire_without_mutation_attempts():
    llm = SequencedLLM([LLMResult(content="The answer is 42.", tool_calls=[])])
    agent = StudentAgent(llm_client=llm)
    tool_results = [{"tool_name": "list_directory", "ok": True, "result": {"content": [{"text": "[FILE] a"}]}}]

    response = agent.solve_mcpmark(_mcpmark_recovery_request(tool_results=tool_results))

    assert response.kind == "final"
    assert len(llm.calls) == 1


def test_failure_recovery_guard_does_not_fire_when_turns_exhausted():
    llm = SequencedLLM([LLMResult(content="Done.", tool_calls=[])])
    agent = StudentAgent(llm_client=llm)
    request = _mcpmark_recovery_request(
        tool_results=_failed_write_result(),
        step=20,
        limits={"max_agent_turns": 20, "model_input_tokens": 64000, "case_timeout_sec": 300},
    )

    response = agent.solve_mcpmark(request)

    assert response.kind == "final"
    assert len(llm.calls) == 1


def test_failure_recovery_guard_does_not_fire_when_time_exhausted():
    llm = SequencedLLM([LLMResult(content="Done.", tool_calls=[])])
    agent = StudentAgent(llm_client=llm)
    # A timeout budget of 1s means elapsed (~0s) is already below timeout-15s -> guard is skipped.
    request = _mcpmark_recovery_request(
        tool_results=_failed_write_result(),
        limits={"max_agent_turns": 20, "model_input_tokens": 64000, "case_timeout_sec": 1},
    )

    response = agent.solve_mcpmark(request)

    assert response.kind == "final"
    assert len(llm.calls) == 1



# --- tau2 basic_economy upgrade gating (consent + budget) ---------------------

def _tau_request_at_step(tools, step):
    return AgentRequest.from_dict(
        {
            "request_id": "req-note",
            "context_id": "ctx-note",
            "benchmark": "tau2_airline",
            "case_id": "19",
            "step": step,
            "text": "I need to change my flight.",
            "observation": {"message_type": "user", "content": "I need to change my flight."},
            "tools": tools,
            "limits": {"max_agent_turns": 100, "model_input_tokens": 64000},
        }
    )


def _basic_economy_state(first_user_msg):
    res = {
        "cabin": "basic_economy",
        "flights": [{"flight_number": "AA1", "date": "2024-05-20"}],
        "payment_history": [{"payment_id": "credit_card_1"}],
    }
    return {
        "messages": [{"role": "user", "content": first_user_msg}],
        "tau2_initial_reservations_by_id": {"RES1": dict(res)},
        "tau2_reservations_by_id": {"RES1": dict(res)},
    }


def test_tau2_prompt_requires_customer_initiated_upgrade():
    from course_agent.student_tools.prompts import build_tau2_system_prompt

    prompt = build_tau2_system_prompt(_tau_request([{"name": "respond_to_user"}])).lower()
    # Customer-initiated-only rule must be present.
    assert "customer-initiated only" in prompt
    assert "do not solicit" in prompt
    assert "honour an explicit refusal" in prompt
    # The old solicitation instruction ("ASK first and get a clear yes") must be gone.
    assert "ask first and get a clear" not in prompt
    # And no unconditional "just use the 2-step path" instruction.
    assert "use the 2-step path: **step 1**" not in prompt


def test_tau2_note_does_not_solicit_upgrade():
    from course_agent.student_tools import build_ephemeral_turn_note

    req = _tau_request_at_step([{"name": "respond_to_user"}], step=2)
    note = build_ephemeral_turn_note(
        req, _basic_economy_state("I need to change my flight to a nonstop please.")
    )
    assert note is not None
    low = note.lower()
    # The note must instruct the agent NOT to initiate/offer the upgrade.
    assert "must not initiate or proactively offer" in low
    assert "customer themselves" in low
    # The old pushy line must be gone.
    assert "do not steer them to keep current flights" not in low
