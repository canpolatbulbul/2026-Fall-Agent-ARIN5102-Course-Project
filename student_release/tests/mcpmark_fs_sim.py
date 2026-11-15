"""Faithful offline replay harness for deterministic MCPMark filesystem tasks.

This module simulates the MCPMark filesystem server's tool semantics against a
local temporary copy of the test environment, so a deterministic solver can be
driven end-to-end through :func:`plan_deterministic_mcpmark` and then validated
with the task's real ``verify.py``.

It is *test-only* code (a harness), so it may touch the temp copy directly --
the solver under test must never do so (it only reads ``tool_results``).

Replay fidelity: the simulator reproduces the server's exact text (the
production parsers are the contract). It does NOT normalize or absolutize paths
itself -- the solver already emits absolute paths -- and it returns the paths
exactly as passed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from course_agent.protocol import AgentRequest
from course_agent.student_tools.deterministic_mcpmark import plan_deterministic_mcpmark
from course_agent.student_tools.policy_guard import (
    SanitizedToolCall,
    classify_tool,
    sanitize_tool_calls,
)

_DEFAULT_LIMITS: dict[str, Any] = {
    "max_mutating_tool_calls_per_turn": 1,
    "max_tool_calls_per_turn": 4,
}


def make_request(
    instruction: str,
    tool_results: list[Any],
    *,
    test_dir: str | os.PathLike[str],
    tools: list[dict[str, Any]],
    benchmark: str = "mcpmark",
    case_id: str = "filesystem/file_context/duplicates_searching",
    limits: dict[str, Any] | None = None,
) -> AgentRequest:
    """Build an :class:`AgentRequest` shaped like the Phase A replay tests."""
    return AgentRequest.from_dict(
        {
            "request_id": "r",
            "context_id": "c",
            "benchmark": benchmark,
            "case_id": case_id,
            "step": 0,
            "text": instruction,
            "observation": {
                "test_directory": str(test_dir),
                "instruction": instruction,
                "tool_results": tool_results,
            },
            "tools": tools,
            "limits": dict(limits) if limits is not None else dict(_DEFAULT_LIMITS),
        }
    )


class SimEnv:
    """A minimal in-process stand-in for the MCPMark filesystem server."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _tree(self, path: Path) -> list[dict[str, Any]]:
        """List children of ``path``; keep ALL entries (including dotfiles)."""
        entries: list[dict[str, Any]] = []
        for child in sorted(path.iterdir(), key=lambda p: p.name):
            if child.is_dir():
                entries.append(
                    {"name": child.name, "type": "directory", "children": self._tree(child)}
                )
            else:
                entries.append({"name": child.name, "type": "file"})
        return entries

    def execute(self, call: SanitizedToolCall) -> dict[str, Any]:
        """Perform ``call`` against the temp copy and return a runner tool_result."""
        name = call.name
        args = dict(call.arguments)
        if name == "directory_tree":
            text = json.dumps(self._tree(Path(args["path"])), indent=2)
        elif name == "read_multiple_files":
            entries: list[str] = []
            for p in args["paths"]:
                try:
                    content = Path(p).read_text()
                    entries.append(f"{p}:\n{content}\n")
                except Exception as exc:  # noqa: BLE001 - mirror the server's inline error
                    entries.append(f"{p}: Error - {exc}")
            text = "\n---\n".join(entries)
        elif name == "read_text_file":
            text = Path(args["path"]).read_text()
        elif name == "write_file":
            Path(args["path"]).write_text(args["content"])
            text = f"Successfully wrote to {args['path']}"
        elif name == "create_directory":
            Path(args["path"]).mkdir(parents=True, exist_ok=True)
            text = f"Successfully created directory {args['path']}"
        elif name == "move_file":
            os.rename(args["source"], args["destination"])
            text = f"Successfully moved {args['source']} to {args['destination']}"
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected tool {name!r}")
        return {
            "tool_name": name,
            "tool_arguments": args,
            "ok": True,
            "result": {"isError": False, "content": [{"text": text}]},
        }


def _assert_batch_obeys_policy(
    calls: list[SanitizedToolCall],
    tools: list[dict[str, Any]],
    limits: dict[str, Any],
) -> None:
    """Assert ``calls`` is a policy-valid mcpmark batch (reads before mutations)."""
    schemas_by_name = {t.get("name"): t for t in tools}
    sanitized = sanitize_tool_calls(
        benchmark="mcpmark",
        raw_calls=[c.to_protocol_dict() for c in calls],
        # Pass the REAL schema (with readOnlyHint) so classify_tool matches
        # production; a bare {"name": ...} schema misclassifies directory_tree.
        available_tools=tools,
        limits=limits,
    )
    assert sanitized == calls, f"policy sanitizer altered the batch: {calls} -> {sanitized}"

    seen_mutating = False
    mutating = 0
    for call in calls:
        kind = classify_tool("mcpmark", call.name, schemas_by_name.get(call.name))
        if kind == "mutating":
            seen_mutating = True
            mutating += 1
        else:
            assert not seen_mutating, "read call emitted after a mutating call"
    assert mutating <= 1, f"more than one mutating call in a batch: {calls}"


def drive(
    instruction: str,
    case_id: str,
    test_dir: str | os.PathLike[str],
    tools: list[dict[str, Any]],
    *,
    limits: dict[str, Any] | None = None,
    max_turns: int = 80,
) -> int:
    """Drive the hook across turns until ``is_final``; return the final turn index.

    Feeds each turn's tool results back exactly as the runner would and asserts a
    non-``None`` decision every turn plus per-turn policy compliance. Fails the
    test if the hook returns ``None`` or never reaches ``is_final``.
    """
    test_dir = Path(test_dir)
    env = SimEnv(test_dir)
    active_limits = dict(limits) if limits is not None else dict(_DEFAULT_LIMITS)
    results: list[dict[str, Any]] = []
    state: dict[str, Any] = {}

    for step in range(max_turns + 1):
        request = make_request(
            instruction,
            results,
            test_dir=str(test_dir),
            tools=tools,
            benchmark="mcpmark",
            case_id=case_id,
            limits=active_limits,
        )
        decision = plan_deterministic_mcpmark(request, state)
        assert decision is not None, f"hook returned None at turn {step}"
        if decision.is_final:
            return step
        assert decision.calls, f"hook emitted an empty batch at turn {step}"
        _assert_batch_obeys_policy(decision.calls, tools, active_limits)
        for call in decision.calls:
            results.append(env.execute(call))

    raise AssertionError("hook never reached is_final")  # pragma: no cover - defensive


def run_verify(test_dir: str | os.PathLike[str], verify_path: str | os.PathLike[str]) -> subprocess.CompletedProcess:
    """Run the task's real ``verify.py`` against ``test_dir`` as a subprocess."""
    return subprocess.run(
        [sys.executable, str(verify_path)],
        env={**os.environ, "FILESYSTEM_TEST_DIR": str(test_dir)},
        capture_output=True,
        text=True,
    )
