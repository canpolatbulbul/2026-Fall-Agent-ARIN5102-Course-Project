"""Deterministic compute-and-emit hook for MCPMark.

This module lets the MCPMark solver bypass the LLM and emit *exact* tool
calls for tasks whose required environment interaction can be computed
deterministically from data already present in ``request.observation``.

Hard rules:
- All reads and writes to the benchmark environment go exclusively through
  ``request.tools`` calls. This module may only READ
  ``request.observation["tool_results"]`` (and other observation fields); it
  must never touch the filesystem, database, or environment directly.
- ``plan_deterministic_mcpmark`` is MCPMark-only and side-effect-free when it
  does not recognize a task (it returns ``None`` and mutates nothing).

The framework here is intentionally minimal: Task 1 provides the entry point,
the ``DeterministicDecision`` type, and a failure guard. Concrete solvers are
registered in ``_SOLVERS`` by later tasks.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from course_agent.protocol import AgentRequest
from course_agent.student_tools.data_helpers import (
    extract_mcp_text,  # noqa: F401  (consumed by later solvers)
    get_written_files_from_tool_results,
    mcpmark_item_failed,
    parse_read_files_from_tool_results,
)
from course_agent.student_tools.policy_guard import SanitizedToolCall


@dataclass
class DeterministicDecision:
    """A deterministic plan: exact tool calls to emit, or a final answer.

    ``is_final`` True means the agent is done and should return a final
    response (with empty content) instead of any tool calls.
    """

    calls: list[SanitizedToolCall] = field(default_factory=list)
    is_final: bool = False


def _instruction(request: AgentRequest) -> str:
    """Return the task instruction text from the richest available field."""
    obs = request.observation or {}
    return str(obs.get("instruction") or obs.get("task") or request.text or "")


def _has_tool(request: AgentRequest, name: str) -> bool:
    """Return True iff ``name`` is one of the tools offered this turn."""
    for tool in request.tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("name") == name:
            return True
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == name:
            return True
    return False


def _emitted_call_failed_before(
    request: AgentRequest,
    state: dict[str, Any],
    call: SanitizedToolCall,
) -> bool:
    """Return True iff ``call`` is identical to an already-failed tool_result.

    Matching is by tool name plus equal ``tool_arguments``. Whether a matched
    result failed is decided by the shared ``data_helpers.mcpmark_item_failed``
    predicate (the same one the mutation-recovery summary uses), so the two can
    never diverge.
    """
    obs = request.observation or {}
    tool_results = obs.get("tool_results")
    if not isinstance(tool_results, list):
        return False
    for item in tool_results:
        if not isinstance(item, dict):
            continue
        if item.get("tool_name") != call.name:
            continue
        if item.get("tool_arguments") != call.arguments:
            continue
        if mcpmark_item_failed(item)[0]:
            return True
    return False


# Solver callables: ``solver(request, state) -> DeterministicDecision | None``.
# Concrete solvers are registered in the ``_SOLVERS`` tuple defined below.


# ---------------------------------------------------------------------------
# file_splitting solver
# ---------------------------------------------------------------------------

# Recognition is a conservative conjunction of instruction markers and required
# tools. It must NOT hijack pattern_matching (which also references
# large_file.txt) — that task never contains the word "split". The file count N
# is parsed from the instruction; if it cannot be parsed unambiguously the solver
# declines (returns None) rather than guess.
_FILE_SPLITTING_REQUIRED_TOOLS = ("read_text_file", "write_file", "create_directory")

_SPLIT_COUNT_RE = re.compile(r"exactly\s+(\d+)\s+files?\b", re.IGNORECASE)
_SPLIT_NAME_RE = re.compile(r"split_(\d+)\.txt")


def _parse_split_count(instruction: str) -> int | None:
    """Parse the unambiguous number of split files N from the instruction.

    Accepts an explicit "exactly <N> files" count and/or an enumeration of
    ``split_NN.txt`` names (N taken as the largest index seen, which covers the
    ``split_01.txt, split_02.txt, ..., split_NN.txt`` form). If both signals are
    present and disagree, the instruction is ambiguous and ``None`` is returned.
    """
    count_match = _SPLIT_COUNT_RE.search(instruction)
    explicit = int(count_match.group(1)) if count_match else None
    indices = [int(value) for value in _SPLIT_NAME_RE.findall(instruction)]
    enumerated = max(indices) if indices else None

    if explicit is not None and enumerated is not None:
        return explicit if explicit == enumerated else None
    result = explicit if explicit is not None else enumerated
    if result is None or result < 1 or result > 1000:
        return None
    return result


def _split_n(content: str, n: int) -> list[str]:
    """Split ``content`` into ``n`` contiguous, near-equal parts.

    Part ``i`` is ``content[i*len//n:(i+1)*len//n]``; concatenation is exact and
    part lengths differ by at most one, satisfying both the 3-file "similar
    character counts" and 10-file "equal character counts" requirements.
    """
    length = len(content)
    return [content[i * length // n : (i + 1) * length // n] for i in range(n)]


def _split_three(content: str) -> list[str]:
    """Backwards-compatible 3-way split (see :func:`_split_n`)."""
    return _split_n(content, 3)


def _has_read_basename(request: AgentRequest, basename: str) -> bool:
    """Return True iff a successful read_text_file result exists for ``basename``.

    Lookup is by ``os.path.basename`` because the read index is keyed by full
    path only; a bare substring test would wrongly match e.g. ``large_file.txtx``.
    """
    obs = request.observation or {}
    tool_results = obs.get("tool_results")
    if not isinstance(tool_results, list):
        return False
    reads = parse_read_files_from_tool_results(tool_results)
    return any(os.path.basename(path) == basename for path in reads)


def _directory_created(tool_results: list[Any], path: str) -> bool:
    """Return True iff a non-failed ``create_directory`` for ``path`` succeeded."""
    target = os.path.normpath(path)
    for item in tool_results:
        if not isinstance(item, dict):
            continue
        if item.get("tool_name") != "create_directory":
            continue
        targs = item.get("tool_arguments") if isinstance(item.get("tool_arguments"), dict) else {}
        if os.path.normpath(str(targs.get("path") or "")) != target:
            continue
        if not mcpmark_item_failed(item)[0]:
            return True
    return False


def _file_splitting_solver(
    request: AgentRequest,
    state: dict[str, Any],
) -> DeterministicDecision | None:
    """Deterministically split ``large_file.txt`` into N balanced files.

    N is parsed from the instruction (3-file and 10-file variants share this
    solver). Re-entrant: the plan is derived from
    ``request.observation["tool_results"]`` on every call, so it resumes correctly
    after interruptions. Emits at most one mutating call per turn (reads before
    mutations).
    """
    instruction = _instruction(request)
    if "large_file.txt" not in instruction or "split" not in instruction:
        return None

    count = _parse_split_count(instruction)
    if count is None:
        return None

    obs = request.observation or {}
    test_dir = str(obs.get("test_directory") or "")
    if not test_dir:
        return None

    if not all(_has_tool(request, name) for name in _FILE_SPLITTING_REQUIRED_TOOLS):
        return None

    tool_results = obs.get("tool_results")
    if not isinstance(tool_results, list):
        tool_results = []

    large_path = os.path.join(test_dir, "large_file.txt")
    reads_by_basename = {
        os.path.basename(path): content
        for path, content in parse_read_files_from_tool_results(tool_results).items()
    }

    # 1. Read the source file if we have not successfully read it yet.
    if "large_file.txt" not in reads_by_basename:
        return DeterministicDecision(
            calls=[SanitizedToolCall(name="read_text_file", arguments={"path": large_path})]
        )

    parts = _split_n(reads_by_basename["large_file.txt"], count)

    # 2. Ensure the destination directory exists.
    split_dir = os.path.join(test_dir, "split")
    if not _directory_created(tool_results, split_dir):
        return DeterministicDecision(
            calls=[SanitizedToolCall(name="create_directory", arguments={"path": split_dir})]
        )

    # 3. Overwrite-on-mismatch: (re-)write the FIRST part whose recorded content
    # differs from the computed part. Re-writing with different args is safe
    # (write_file overwrites atomically) and is not vetoed by the failure guard.
    written = get_written_files_from_tool_results(tool_results)
    for index in range(1, count + 1):
        target = os.path.join(split_dir, f"split_{index:02d}.txt")
        if written.get(target) != parts[index - 1]:
            return DeterministicDecision(
                calls=[
                    SanitizedToolCall(
                        name="write_file",
                        arguments={"path": target, "content": parts[index - 1]},
                    )
                ]
            )

    # 4. All N recorded contents match: the task is complete.
    return DeterministicDecision(is_final=True)


# Registered solvers, tried in order by ``plan_deterministic_mcpmark``.
_SOLVERS: tuple[Callable[[AgentRequest, dict[str, Any]], DeterministicDecision | None], ...] = (
    _file_splitting_solver,
)


def plan_deterministic_mcpmark(
    request: AgentRequest,
    state: dict[str, Any],
) -> DeterministicDecision | None:
    """Return a deterministic decision for recognized MCPMark tasks, else ``None``.

    Side-effect-free on unrecognized tasks. Mutates ``state`` only to record the
    ``det_disabled`` failure-guard latch when a solver re-emits a call that has
    already failed in ``tool_results``.
    """
    if request.benchmark != "mcpmark" or state.get("det_disabled"):
        return None
    for solver in _SOLVERS:
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
