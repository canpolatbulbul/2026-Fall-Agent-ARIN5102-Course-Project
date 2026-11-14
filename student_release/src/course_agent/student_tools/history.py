from __future__ import annotations

import copy
import json
import re
from typing import Any
import uuid

from course_agent.protocol import AgentRequest
from course_agent.student_tools.policy_guard import SanitizedToolCall


MAX_TOOL_RESULT_CHARS = 65000
SAFE_INPUT_TOKEN_BUDGET = 48000

_DP_STRIP_PRODUCT_FIELDS = (
    "washing_instructions",
    "material_composition",
    "details_craftsmanship",
    "review_summary",
    "collar_type",
    "elasticity",
    "thickness",
)


def _infer_dp_category(name: str) -> str:
    nl = (name or "").lower()
    if any(w in nl for w in ("shoe", "sneaker", "boot", "sandal", "loafer", "slipper", "cleat", "footwear", "trainer", "runner")):
        return "shoes"
    if any(w in nl for w in ("pant", "trouser", "jean", "short", "skirt", "legging", "tight", "jogger", "chino", "cargo", "sweatpant", "bottom", "capri", "slacks")):
        return "bottoms"
    return "tops"

_DP_PROVINCE_ALIASES: dict[str, list[str]] = {
    "beijing": ["beijing", "bj", "北京"],
    "shanghai": ["shanghai", "sh", "上海"],
    "tianjin": ["tianjin", "tj", "天津"],
    "chongqing": ["chongqing", "cq", "重庆"],
    "hebei": ["hebei", "ji", "河北"],
    "shanxi": ["shanxi", "jin", "山西"],
    "liaoning": ["liaoning", "liao", "辽宁"],
    "jilin": ["jilin", "ji_ln", "吉林"],
    "heilongjiang": ["heilongjiang", "hei", "黑龙江"],
    "jiangsu": ["jiangsu", "su", "江苏"],
    "zhejiang": ["zhejiang", "zhe", "浙江"],
    "anhui": ["anhui", "wan", "安徽"],
    "fujian": ["fujian", "min", "福建"],
    "jiangxi": ["jiangxi", "gan", "江西"],
    "shandong": ["shandong", "lu", "山东"],
    "henan": ["henan", "yu", "河南"],
    "hubei": ["hubei", "e", "湖北"],
    "hunan": ["hunan", "xiang", "湖南"],
    "guangdong": ["guangdong", "yue", "gd", "广东"],
    "hainan": ["hainan", "qiong", "海南"],
    "sichuan": ["sichuan", "chuan", "shu", "四川"],
    "guizhou": ["guizhou", "qian", "gui_gz", "贵州"],
    "yunnan": ["yunnan", "yun", "dian", "云南"],
    "shaanxi": ["shaanxi", "shan", "qin", "陕西"],
    "gansu": ["gansu", "gan_gs", "甘肃"],
    "qinghai": ["qinghai", "qing", "青海"],
    "inner mongolia": ["inner mongolia", "neimenggu", "meng", "内蒙古"],
    "guangxi": ["guangxi", "gui", "广西"],
    "tibet": ["tibet", "xizang", "zang", "西藏"],
    "ningxia": ["ningxia", "ning", "宁夏"],
    "xinjiang": ["xinjiang", "xin", "新疆"],
    "hongkong": ["hongkong", "hk", "xianggang", "香港"],
    "macau": ["macau", "mo", "aomen", "澳门"],
    "taiwan": ["taiwan", "tw", "台湾"],
}

_DP_PROVINCE_NORM: dict[str, str] = {
    alias: std_name for std_name, aliases in _DP_PROVINCE_ALIASES.items() for alias in aliases
}

_DP_REGION_MAP: dict[str, str] = {
    "beijing": "NC", "tianjin": "NC", "hebei": "NC", "shanxi": "NC", "inner mongolia": "NC",
    "liaoning": "NE", "jilin": "NE", "heilongjiang": "NE",
    "shanghai": "EC", "jiangsu": "EC", "zhejiang": "EC", "anhui": "EC", "fujian": "EC", "jiangxi": "EC", "shandong": "EC",
    "henan": "CC", "hubei": "CC", "hunan": "CC",
    "guangdong": "SC", "guangxi": "SC", "hainan": "SC", "hongkong": "SC", "macau": "SC", "taiwan": "SC",
    "sichuan": "SW", "chongqing": "SW", "guizhou": "SW", "yunnan": "SW", "tibet": "SW",
    "shaanxi": "NW", "gansu": "NW", "qinghai": "NW", "ningxia": "NW", "xinjiang": "NW",
}

_DP_BASE_REGION_TIME: dict[str, dict[str, int]] = {
    "NC": {"NC": 1, "NE": 2, "EC": 2, "CC": 2, "SC": 3, "SW": 3, "NW": 3},
    "NE": {"NC": 2, "NE": 1, "EC": 3, "CC": 3, "SC": 4, "SW": 4, "NW": 4},
    "EC": {"NC": 2, "NE": 3, "EC": 1, "CC": 2, "SC": 2, "SW": 3, "NW": 4},
    "CC": {"NC": 2, "NE": 3, "EC": 2, "CC": 1, "SC": 2, "SW": 2, "NW": 3},
    "SC": {"NC": 3, "NE": 4, "EC": 2, "CC": 2, "SC": 1, "SW": 3, "NW": 4},
    "SW": {"NC": 3, "NE": 4, "EC": 3, "CC": 2, "SC": 3, "SW": 1, "NW": 3},
    "NW": {"NC": 3, "NE": 4, "EC": 4, "CC": 3, "SC": 4, "SW": 3, "NW": 1},
}

_DP_PROVIDER_MODIFIERS: dict[str, int] = {
    "sf express": -2, "sf": -2,
    "jd logistics": -1, "jd": -1,
    "yto express": 1, "yto": 0,
    "zto express": 1, "zto": 0,
    "sto express": 1, "sto": 0,
    "yunda express": 1, "yunda": 0,
    "cainiao": 1,
    "china post": 2,
    "ems": 0,
    "deppon express": 0, "deppon": 0,
    "default": 0,
}


def _normalize_dp_province(address_str: str | None) -> str | None:
    if not address_str:
        return None
    processed = address_str.lower().replace(" ", "").replace("province", "").replace("city", "")
    if processed in _DP_PROVINCE_NORM:
        return _DP_PROVINCE_NORM[processed]
    for alias, std_name in _DP_PROVINCE_NORM.items():
        if alias in processed:
            return std_name
    return None


def _estimate_dp_delivery_days(origin_str: str | None, provider_str: str | None, dest_str: str | None) -> int | None:
    origin_prov = _normalize_dp_province(origin_str)
    dest_prov = _normalize_dp_province(dest_str)
    if not origin_prov or not dest_prov:
        return None
    origin_reg = _DP_REGION_MAP.get(origin_prov)
    dest_reg = _DP_REGION_MAP.get(dest_prov)
    if not origin_reg or not dest_reg:
        return None
    base_days = _DP_BASE_REGION_TIME[origin_reg][dest_reg]
    remote = {"tibet", "xinjiang", "qinghai", "inner mongolia"}
    if origin_prov in remote or dest_prov in remote:
        base_days += 2
    prov_key = (provider_str or "default").lower()
    mod = _DP_PROVIDER_MODIFIERS.get(prov_key, 0)
    return max(1, base_days + mod)


def _compact_and_enrich_tool_item(item: Any, state: dict[str, Any], benchmark: str) -> Any:
    """Strip unused bulky fields from DeepPlanning get_product_details and attach estimated_delivery_days."""
    if benchmark != "deepplanning" or not isinstance(item, dict):
        return item

    tname = str(item.get("tool_name") or "")
    res = item.get("result")

    if tname == "get_user_info" and isinstance(res, dict):
        user_data = res.get("user_info") if isinstance(res.get("user_info"), dict) else res
        if isinstance(user_data, dict) and ("coupons" in user_data or "shipping_addresses" in user_data):
            state["cached_user_info"] = user_data
        return item

    if tname.startswith("filter_by_") or tname == "sort_products":
        if isinstance(res, dict):
            ids = res.get("filtered_products_ids") or res.get("sorted_product_ids") or res.get("product_ids")
            if isinstance(ids, list) and ids:
                targs = item.get("tool_arguments") if isinstance(item.get("tool_arguments"), dict) else {}
                if (
                    tname == "filter_by_range"
                    and targs.get("condition_key") == "price"
                    and targs.get("operator") == ">"
                    and float(targs.get("value") or 0) == 0.0
                ):
                    state["dp_all_catalog_ids"] = [str(x) for x in ids]
                else:
                    batches = state.setdefault("filter_batches", [])
                    batches.append((tname, targs, list(ids), False))
        return item

    if tname == "add_product_to_cart":
        targs = item.get("tool_arguments") if isinstance(item.get("tool_arguments"), dict) else {}
        added_pid = str(targs.get("product_id") or "")
        if added_pid:
            state.setdefault("added_product_ids", []).append(added_pid)
        return item

    if tname != "get_product_details":
        return item
    if not isinstance(res, dict) or not isinstance(res.get("products"), list):
        return item

    inspected: set[str] = state.setdefault("inspected_product_ids", set())
    current_detail_ids: set[str] = set()
    user_info = state.get("cached_user_info")
    dest_prov = None
    expected_demo = None
    std_sizes: dict[str, str] = {}
    pref_fit = None
    if isinstance(user_info, dict):
        if isinstance(user_info.get("shipping_addresses"), dict):
            dest_prov = user_info["shipping_addresses"].get("province")
        if isinstance(user_info.get("demographics"), dict):
            gender = user_info["demographics"].get("gender")
            expected_demo = "Men" if gender == "Male" else ("Women" if gender == "Female" else None)
        if isinstance(user_info.get("body_profile"), dict):
            bp = user_info["body_profile"]
            if isinstance(bp.get("standard_sizes"), dict):
                std_sizes = {k: str(v) for k, v in bp["standard_sizes"].items()}
            pref_fit = bp.get("preference_fit")

    query_text = str(state.get("initial_user_text") or "")
    mentioned_sizes = set(re.findall(r"\b(XS|S|M|L|XL|XXL|2XL|3XL|\d{2})\b", query_text))
    is_full_catalog_call = len(res["products"]) > 40

    compact_products: list[Any] = []
    for prod in res["products"]:
        if not isinstance(prod, dict):
            compact_products.append(prod)
            continue
        pid = prod.get("product_id")
        if pid:
            inspected.add(str(pid))
            current_detail_ids.add(str(pid))

        cat = _infer_dp_category(str(prod.get("name") or ""))
        prod_demo = str(prod.get("target_demographic") or "")
        prod_size = str(prod.get("size") or "")
        prod_fit = str(prod.get("version_type") or "")
        default_size_ok = bool(
            (not expected_demo or prod_demo == expected_demo)
            and (not std_sizes or prod_size == std_sizes.get(cat))
        )
        default_fit_ok = bool(not pref_fit or prod_fit == pref_fit or prod_fit == "N/A")

        # When pre-fetching the entire catalog (>40 products), filter to products matching
        # the user's demographic and either their category standard_size or an explicitly requested size
        if is_full_catalog_call and expected_demo and std_sizes:
            if prod_demo != expected_demo:
                continue
            if prod_size != std_sizes.get(cat) and prod_size not in mentioned_sizes:
                continue

        raw_filtered = {k: v for k, v in prod.items() if k not in _DP_STRIP_PRODUCT_FIELDS}
        days = None
        if dest_prov and isinstance(raw_filtered.get("shipping_info"), dict):
            s_info = raw_filtered["shipping_info"]
            days = _estimate_dp_delivery_days(s_info.get("origin"), s_info.get("provider"), dest_prov)

        # Order keys so price, product_id, name, brand, size, demographic, season, and delivery days appear first
        priority_keys = (
            "price",
            "product_id",
            "name",
            "brand",
            "color",
            "size",
            "target_demographic",
            "suitable_season",
            "version_type",
        )
        p_copy: dict[str, Any] = {}
        for k in priority_keys:
            if k in raw_filtered:
                p_copy[k] = raw_filtered[k]
        p_copy["category"] = cat
        p_copy["default_size_ok"] = default_size_ok
        p_copy["default_fit_ok"] = default_fit_ok
        if days is not None:
            p_copy["transport_time"] = days
            raw_filtered.pop("shipping_info", None)
        for k, v in raw_filtered.items():
            if k not in p_copy:
                p_copy[k] = v
        compact_products.append(p_copy)

    # Mark any prior filter_batches that overlap with THIS get_product_details call as inspected_after_batch=True
    batches = state.get("filter_batches") or []
    for b_idx, entry in enumerate(batches):
        b_tname, b_targs, b_ids, b_inspected = entry
        if not b_inspected and current_detail_ids & set(str(x) for x in b_ids):
            batches[b_idx] = (b_tname, b_targs, b_ids, True)

    # Sort products by price ascending so the cheapest valid option is always presented first
    compact_products.sort(
        key=lambda x: float(x.get("price") or 0.0) if isinstance(x, dict) else 0.0
    )
    if is_full_catalog_call:
        state["dp_catalog_products"] = compact_products

    item_copy = copy.copy(item)
    res_copy = copy.copy(res)
    res_copy["_sorted_by_price_asc"] = True
    res_copy["_tie_breaker_rule"] = (
        "Products are sorted by price ASCENDING (cheapest first). Among all products satisfying a sub-request's "
        "constraints, ALWAYS pick the FIRST (lowest-price) valid product (unless a slightly higher-priced valid "
        "candidate unlocks a Level 2 minimum budget or a Level 3 coupon threshold that lowers final cost). "
        "NEVER pick a more expensive product just because its rating/review/sales count is closer to a '>' threshold!"
    )
    res_copy["products"] = compact_products
    item_copy["result"] = res_copy
    return item_copy


def estimate_tokens(payload: Any) -> int:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return max(1, (len(text) + 3) // 4)


def truncate_text(text: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return f"{text[:half]}\n... [TRUNCATED {omitted} chars] ...\n{text[-half:]}"


def serialize_tool_output(raw: Any, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    if isinstance(raw, str):
        text = raw
    else:
        try:
            text = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            text = str(raw)
    return truncate_text(text, max_chars=max_chars)


def _ingest_tau2_tool_output(state: dict[str, Any], call_info: dict[str, Any] | None, raw_content: Any) -> None:
    """Parse Tau2 tool outputs to track user reservations, reservation details, and cumulative flight price differences."""
    if not isinstance(call_info, dict):
        return
    tname = str(call_info.get("name") or "")
    targs = call_info.get("arguments") if isinstance(call_info.get("arguments"), dict) else {}
    parsed: Any = None
    if isinstance(raw_content, (dict, list)):
        parsed = raw_content
    elif isinstance(raw_content, str):
        try:
            parsed = json.loads(raw_content)
        except Exception:
            return
    if parsed is None:
        return

    if tname == "get_user_details" and isinstance(parsed, dict):
        state["tau2_user_details"] = parsed
        res_list = parsed.get("reservations")
        if isinstance(res_list, list):
            state["tau2_user_reservations"] = [str(r) for r in res_list if r]
    elif tname == "get_reservation_details" and isinstance(parsed, dict):
        rid = str(parsed.get("reservation_id") or targs.get("reservation_id") or "")
        if rid:
            state.setdefault("tau2_inspected_reservations", set()).add(rid)
            state.setdefault("tau2_reservations_by_id", {})[rid] = parsed
            state.setdefault("tau2_initial_reservations_by_id", {}).setdefault(rid, copy.deepcopy(parsed))
    elif tname == "search_direct_flight" and isinstance(parsed, list):
        fmap = state.setdefault("tau2_flights_by_num_date", {})
        search_date = str(targs.get("date") or "")
        for fl in parsed:
            if isinstance(fl, dict) and fl.get("flight_number"):
                fmap[(str(fl["flight_number"]), search_date)] = fl
    elif tname == "update_reservation_flights" and isinstance(parsed, dict):
        rid = str(parsed.get("reservation_id") or targs.get("reservation_id") or "")
        if rid:
            prev = state.get("tau2_reservations_by_id", {}).get(rid)
            if isinstance(prev, dict):
                old_pax = max(1, len(prev.get("passengers") or [1]))
                new_pax = max(1, len(parsed.get("passengers") or [1]))
                old_flight_total = sum(int(f.get("price") or 0) for f in (prev.get("flights") or []) if isinstance(f, dict)) * old_pax
                new_flight_total = sum(int(f.get("price") or 0) for f in (parsed.get("flights") or []) if isinstance(f, dict)) * new_pax
                diff = new_flight_total - old_flight_total
                diffs = state.setdefault("tau2_reservation_price_diffs", {})
                diffs[rid] = diffs.get(rid, 0) + diff
            state.setdefault("tau2_reservations_by_id", {})[rid] = parsed


def init_case_state(state: dict[str, Any], system_prompt: str, initial_user_text: str) -> None:
    if state.get("initialized"):
        # Always keep system prompt up-to-date (e.g., if domain_policy is present)
        messages = state.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_prompt
        return

    state["initialized"] = True
    state["initial_user_text"] = initial_user_text
    state["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_text},
    ]
    state["pending_tool_calls"] = []
    state["processed_tool_results"] = 0


def ingest_turn_observation(state: dict[str, Any], request: AgentRequest) -> None:
    """Append new tool results or user messages from request.observation into state['messages']."""
    messages: list[dict[str, Any]] = state["messages"]
    pending_calls: list[dict[str, Any]] = state.get("pending_tool_calls", [])
    obs = request.observation or {}

    if request.benchmark == "tau2_airline":
        msg_type = obs.get("message_type")
        if msg_type == "user":
            if pending_calls:
                for call in pending_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": "Message sent to user.",
                        }
                    )
                state["pending_tool_calls"] = []
            user_content = str(obs.get("content") or request.text or "")
            # Avoid duplicating the initial user message on step 0
            if request.step > 0 and user_content:
                messages.append({"role": "user", "content": user_content})
        elif msg_type == "tool_result":
            raw_c = obs.get("content", "")
            if pending_calls:
                _ingest_tau2_tool_output(state, pending_calls[0], raw_c)
            content_str = serialize_tool_output(raw_c)
            call_id = pending_calls[0]["id"] if pending_calls else str(obs.get("tool_call_id") or "call_0")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content_str,
                }
            )
            if len(pending_calls) > 1:
                for extra_call in pending_calls[1:]:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": extra_call["id"],
                            "content": "Executed.",
                        }
                    )
            state["pending_tool_calls"] = []
        elif msg_type == "tool_results":
            raw_results = obs.get("tool_results") or []
            for idx, item in enumerate(raw_results):
                raw_c = item.get("content", "")
                if idx < len(pending_calls):
                    _ingest_tau2_tool_output(state, pending_calls[idx], raw_c)
                    call_id = pending_calls[idx]["id"]
                else:
                    call_id = str(item.get("tool_call_id") or f"call_{idx}")
                content_str = serialize_tool_output(raw_c)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content_str,
                    }
                )
            if len(pending_calls) > len(raw_results):
                for extra_call in pending_calls[len(raw_results) :]:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": extra_call["id"],
                            "content": "Executed.",
                        }
                    )
            state["pending_tool_calls"] = []
        elif request.step > 0 and request.text:
            if pending_calls:
                for call in pending_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": "Message sent to user.",
                        }
                    )
                state["pending_tool_calls"] = []
            messages.append({"role": "user", "content": request.text})
        return

    # For mcpmark and deepplanning: observation["tool_results"] is a cumulative list
    all_tool_results = obs.get("tool_results")
    if isinstance(all_tool_results, list):
        processed = int(state.get("processed_tool_results", 0))
        new_results = all_tool_results[processed:]
        state["processed_tool_results"] = len(all_tool_results)

        for idx, item in enumerate(new_results):
            if idx < len(pending_calls):
                call_id = pending_calls[idx]["id"]
            else:
                call_id = f"call_{processed + idx}"
            enriched_item = _compact_and_enrich_tool_item(item, state, request.benchmark)
            content_str = serialize_tool_output(enriched_item)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content_str,
                }
            )
        # If for any reason pending_calls had more items than new_results, provide dummy tool responses
        # so OpenAI chat completion does not complain about unmatched tool_call_ids.
        if len(pending_calls) > len(new_results):
            for extra_call in pending_calls[len(new_results) :]:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": extra_call["id"],
                        "content": json.dumps({"ok": False, "skipped": True, "reason": "not_executed"}),
                    }
                )
        state["pending_tool_calls"] = []


def record_assistant_tool_calls(
    state: dict[str, Any],
    content: str,
    calls: list[SanitizedToolCall],
) -> None:
    """Record the assistant message with tool_calls so the next turn's tool outputs match their IDs."""
    pending: list[dict[str, Any]] = []
    openai_tool_calls: list[dict[str, Any]] = []
    for call in calls:
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        pending.append({"id": call_id, "name": call.name, "arguments": call.arguments})
        openai_tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
        )
    state["messages"].append(
        {
            "role": "assistant",
            "content": content or "",
            "tool_calls": openai_tool_calls,
        }
    )
    state["pending_tool_calls"] = pending


def record_assistant_final(state: dict[str, Any], content: str) -> None:
    state["messages"].append({"role": "assistant", "content": content or ""})
    state["pending_tool_calls"] = []


def _sanitize_openai_message_sequence(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure every assistant tool_calls block has matching tool messages and no orphaned tool messages exist."""
    cleaned: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            cleaned.append(msg)
            expected_ids = [
                str(tc.get("id"))
                for tc in (msg.get("tool_calls") or [])
                if isinstance(tc, dict) and tc.get("id")
            ]
            i += 1
            tool_map: dict[str, dict[str, Any]] = {}
            while i < n and messages[i].get("role") == "tool":
                tmsg = messages[i]
                tid = str(tmsg.get("tool_call_id") or "")
                if tid and tid not in tool_map:
                    tool_map[tid] = tmsg
                i += 1
            for tid in expected_ids:
                if tid in tool_map:
                    cleaned.append(tool_map[tid])
                else:
                    cleaned.append(
                        {
                            "role": "tool",
                            "tool_call_id": tid,
                            "content": "Executed.",
                        }
                    )
        elif role == "tool":
            # Drop orphaned tool messages not preceded by an assistant tool_calls message
            i += 1
        else:
            cleaned.append(msg)
            i += 1
    return cleaned


def build_messages_within_budget(
    state: dict[str, Any],
    openai_tools: list[dict[str, Any]],
    limits: dict[str, Any],
    ephemeral_user_note: str | None = None,
) -> list[dict[str, Any]]:
    """Return a copy of state['messages'] (plus optional ephemeral note) guaranteed to fit input token limits."""
    raw_limit = limits.get("model_input_tokens")
    try:
        max_tokens = int(raw_limit) if raw_limit else 64000
    except (TypeError, ValueError):
        max_tokens = 64000
    safe_budget = min(max_tokens - 4000, SAFE_INPUT_TOKEN_BUDGET)

    messages = _sanitize_openai_message_sequence([dict(m) for m in state["messages"]])
    if ephemeral_user_note:
        messages.append({"role": "user", "content": ephemeral_user_note})

    if estimate_tokens({"messages": messages, "tools": openai_tools}) <= safe_budget:
        return messages

    # Step 1: Aggressively truncate older tool message contents first
    for i in range(2, max(2, len(messages) - 4)):
        if messages[i].get("role") == "tool":
            content = str(messages[i].get("content", ""))
            if len(content) > 1200:
                messages[i] = {
                    **messages[i],
                    "content": truncate_text(content, max_chars=1200),
                }
        if estimate_tokens({"messages": messages, "tools": openai_tools}) <= safe_budget:
            return _sanitize_openai_message_sequence(messages)

    # Step 2: Drop oldest assistant+tool blocks (preserving system [0] and initial user [1])
    while len(messages) > 6 and estimate_tokens({"messages": messages, "tools": openai_tools}) > safe_budget:
        # Remove from index 2 until we hit the next assistant or user boundary so we don't orphan tool messages
        del messages[2]
        while len(messages) > 2 and messages[2].get("role") == "tool":
            del messages[2]

    return _sanitize_openai_message_sequence(messages)
