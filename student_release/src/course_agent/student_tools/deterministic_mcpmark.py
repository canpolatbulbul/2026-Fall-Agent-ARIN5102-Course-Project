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

from dataclasses import dataclass, field
from typing import Any, Callable

from course_agent.protocol import AgentRequest
from course_agent.student_tools.data_helpers import (
    extract_mcp_text,
    get_written_files_from_tool_results,  # noqa: F401  (consumed by later solvers)
    parse_read_files_from_tool_results,  # noqa: F401  (consumed by later solvers)
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

    Matching is by tool name plus equal ``tool_arguments``. A result is treated
    as failed iff any of the following holds:
      - ``ok`` is exactly ``False``,
      - ``skipped`` is truthy (the call never executed),
      - ``result.isError`` is exactly ``True``,
      - the extracted MCP text (leading whitespace stripped, lower-cased)
        starts with the ``error`` token.
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
        if (
            item.get("ok") is False
            or item.get("skipped")
            or (
                isinstance(item.get("result"), dict)
                and item["result"].get("isError") is True
            )
            or extract_mcp_text(item).lstrip().lower().startswith("error")
        ):
            return True
    return False


# Solver callables: ``solver(request, state) -> DeterministicDecision | None``.
# Concrete solvers are appended by later tasks (e.g. the file-splitting solver).
_SOLVERS: tuple[Callable[[AgentRequest, dict[str, Any]], DeterministicDecision | None], ...] = ()


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
