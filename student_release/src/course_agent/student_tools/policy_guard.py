from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TAU2_AIRLINE_MUTATING_TOOLS = frozenset(
    {
        "book_reservation",
        "cancel_reservation",
        "respond_to_user",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }
)

TAU2_AIRLINE_TERMINAL_TOOLS = frozenset({"respond_to_user"})

DEEPLANNING_SHOPPING_MUTATING_TOOLS = frozenset(
    {
        "add_product_to_cart",
        "delete_product_from_cart",
        "add_coupon_to_cart",
        "delete_coupon_from_cart",
        "submit_plan",
    }
)

DEEPLANNING_SHOPPING_TERMINAL_TOOLS = frozenset({"submit_plan"})

READ_NAME_PREFIXES = (
    "read",
    "list",
    "search",
    "query",
    "select",
    "get",
    "describe",
    "filter",
    "sort",
    "calculate",
)


@dataclass(frozen=True)
class SanitizedToolCall:
    name: str
    arguments: dict[str, Any]

    def to_protocol_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.name,
            "tool_arguments": self.arguments,
        }


def tool_name_from_schema(tool: dict[str, Any]) -> str:
    if isinstance(tool.get("name"), str):
        return str(tool["name"])
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return str(function["name"])
    return ""


def normalize_tools_for_openai(raw_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert flat or OpenAI-style tool schemas into strict OpenAI function tool dicts."""
    normalized: list[dict[str, Any]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = dict(tool["function"])
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            parameters = fn.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            clean_fn: dict[str, Any] = {
                "name": name,
                "description": str(fn.get("description") or ""),
                "parameters": parameters,
            }
            normalized.append({"type": "function", "function": clean_fn})
            continue

        name = tool.get("name")
        if isinstance(name, str) and name:
            parameters = tool.get("parameters") or tool.get("inputSchema") or tool.get("input_schema")
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            clean_fn = {
                "name": name,
                "description": str(tool.get("description") or ""),
                "parameters": parameters,
            }
            normalized.append({"type": "function", "function": clean_fn})
    return normalized


def default_max_tool_calls_per_turn(benchmark: str) -> int:
    return {"tau2_airline": 3, "deepplanning": 5, "mcpmark": 4}.get(benchmark, 1)


def default_max_mutating_tool_calls_per_turn(benchmark: str) -> int:
    return {"tau2_airline": 1, "deepplanning": 3, "mcpmark": 1}.get(benchmark, 1)


def terminal_tool_names_for_benchmark(benchmark: str) -> set[str]:
    if benchmark == "tau2_airline":
        return set(TAU2_AIRLINE_TERMINAL_TOOLS)
    if benchmark == "deepplanning":
        return set(DEEPLANNING_SHOPPING_TERMINAL_TOOLS)
    return set()


def _read_only_hint(tool_schema: dict[str, Any] | None) -> bool:
    if not isinstance(tool_schema, dict):
        return False
    annotations = tool_schema.get("annotations")
    if isinstance(annotations, dict) and annotations.get("readOnlyHint") is True:
        return True
    function = tool_schema.get("function")
    if isinstance(function, dict):
        annotations = function.get("annotations")
        if isinstance(annotations, dict) and annotations.get("readOnlyHint") is True:
            return True
    return False


def classify_tool(benchmark: str, tool_name: str, tool_schema: dict[str, Any] | None = None) -> str:
    if benchmark == "tau2_airline":
        return "mutating" if tool_name in TAU2_AIRLINE_MUTATING_TOOLS else "read"
    if benchmark == "deepplanning":
        return "mutating" if tool_name in DEEPLANNING_SHOPPING_MUTATING_TOOLS else "read"
    if _read_only_hint(tool_schema):
        return "read"
    if tool_name.lower().startswith(READ_NAME_PREFIXES):
        return "read"
    return "mutating"


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def sanitize_tool_calls(
    *,
    benchmark: str,
    raw_calls: list[Any],
    available_tools: list[dict[str, Any]],
    limits: dict[str, Any],
) -> list[SanitizedToolCall]:
    """Sanitize LLM tool calls so they strictly satisfy runner tool_policy validation."""
    if not raw_calls:
        return []

    schemas_by_name = {
        tool_name_from_schema(tool): tool
        for tool in available_tools
        if tool_name_from_schema(tool)
    }
    terminal_names = terminal_tool_names_for_benchmark(benchmark)

    valid_calls: list[SanitizedToolCall] = []
    for item in raw_calls:
        if isinstance(item, dict):
            name = str(item.get("tool_name") or item.get("name") or "")
            args = item.get("tool_arguments") if "tool_arguments" in item else item.get("arguments")
        else:
            name = str(getattr(item, "name", "") or "")
            args = getattr(item, "arguments", {})
        if not name or name not in schemas_by_name:
            continue
        if not isinstance(args, dict):
            args = {}
        # Drop calls with unparsed raw arguments if JSON parsing failed
        if "_raw_arguments" in args and len(args) == 1:
            continue
        valid_calls.append(SanitizedToolCall(name=name, arguments=dict(args)))

    if not valid_calls:
        return []

    # Rule 1: Terminal tools cannot be combined with any other tool calls.
    # If non-terminal calls are present alongside a terminal call, execute the non-terminal
    # calls first and defer the terminal call to the next turn.
    non_terminal = [c for c in valid_calls if c.name not in terminal_names]
    terminal = [c for c in valid_calls if c.name in terminal_names]
    if non_terminal and terminal:
        valid_calls = non_terminal
    elif terminal:
        return [terminal[0]]

    # Rule 2: Read-only calls must precede mutating calls, and mutating calls <= max_mutating.
    max_mutating = _positive_int(
        limits.get("max_mutating_tool_calls_per_turn"),
        default_max_mutating_tool_calls_per_turn(benchmark),
    )
    max_total = _positive_int(
        limits.get("max_tool_calls_per_turn"),
        default_max_tool_calls_per_turn(benchmark),
    )
    if not bool(limits.get("allow_multi_tool_calls", True)):
        max_total = 1

    read_calls: list[SanitizedToolCall] = []
    mutating_calls: list[SanitizedToolCall] = []
    for call in valid_calls:
        kind = classify_tool(benchmark, call.name, schemas_by_name.get(call.name))
        if kind == "read":
            read_calls.append(call)
        else:
            mutating_calls.append(call)

    mutating_calls = mutating_calls[:max_mutating]

    # Combine read calls first, then mutating calls, respecting max_total.
    # If read_calls alone fill max_total, mutating_calls wait for next turn (or if no read calls, mutating runs).
    combined = (read_calls + mutating_calls)[:max_total]
    return combined
