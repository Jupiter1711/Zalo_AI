"""Dựng prompt cho agent theo format chat của OpenAI/Qwen (messages + tools).

Dùng chung cho dữ liệu SFT (src/data/build_sft.py) và inference, để prompt lúc train và lúc chạy khớp nhau từng ký tự.
Mỗi bước agent gọi đúng một tool; agent dừng khi trả lời bằng text (không gọi tool).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Callable

ASK_TOOL = "sys.ask_missing_info"
FINAL_TEXT = "Đã xong."
OOS_TEXT = "Xin lỗi, mình không thực hiện được yêu cầu này."
WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
DAY_LABELS = [" - hôm nay", " - ngày mai", " - ngày kia"]

RULES = """Bạn là trợ lý LifeMate. Hãy thực hiện yêu cầu CUỐI CÙNG của người dùng bằng cách gọi tool, mỗi lượt gọi đúng một tool.
Quy tắc:
1. Chỉ dùng các tool được cung cấp. Nếu không có tool phù hợp, người dùng tự hủy yêu cầu, hoặc không cần hành động gì thì không gọi tool, chỉ trả lời ngắn.
2. Tham số là ID (thiết bị, liên hệ, hóa đơn, sự kiện...) phải lấy từ kết quả của tool đọc; không tự bịa ID. Nếu đã đủ thông tin thì gọi tool ghi luôn, không đọc thừa.
3. Nếu thiếu tham số bắt buộc, hoặc tìm thấy nhiều kết quả khớp mà không biết chọn cái nào, gọi sys.ask_missing_info với tool_name và danh sách params còn thiếu.
4. Với yêu cầu có điều kiện, đọc dữ liệu để kiểm tra điều kiện trước, chỉ làm khi điều kiện đúng.
5. Thời gian dùng ISO 8601 có múi giờ, ví dụ 2026-11-29T09:00:00+07:00; số tiền là số nguyên VND, ví dụ 500000.
6. Khi đã xong, trả lời ngắn và không gọi thêm tool."""


def installed_app_lines(installed_apps: dict, apps_info: dict[str, dict], drop_apps=()) -> list[str]:
    """Mỗi app đã cài một dòng "- id (tên): mô tả", theo thứ tự installed_apps; bỏ drop_apps và sys."""
    lines = []
    for app in installed_apps:
        if app in drop_apps or app == "sys":
            continue
        info = apps_info.get(app) or {}
        name = f" ({info['name_vi']})" if info.get("name_vi") else ""
        desc = f": {info['description']}" if info.get("description") else ""
        lines.append(f"- {app}{name}{desc}")
    return lines


def build_system_prompt(context: dict, apps_info: dict[str, dict], drop_apps=()) -> str:
    now = datetime.fromisoformat(context["current_time"])
    days = []
    for i in range(7):
        d = now + timedelta(days=i)
        days.append(f"- {d:%Y-%m-%d} ({WEEKDAYS[d.weekday()]}){DAY_LABELS[i] if i < len(DAY_LABELS) else ''}")
    profile = context.get("user_profile") or {}
    loc = context.get("location") or {}
    lines = [
        RULES,
        "",
        "Bối cảnh:",
        f"- Thời gian hiện tại: {context['current_time']} ({WEEKDAYS[now.weekday()]}), múi giờ {context.get('timezone', '')}",
        f"- Tên người dùng: {profile.get('name', '')}",
        f"- Địa chỉ nhà: {profile.get('home_address', '')}",
        f"- Nơi làm việc: {profile.get('work_address', '')}",
        f"- Thành phố hiện tại: {profile.get('city', '')}",
    ]
    if loc:
        lines.append(f"- Vị trí hiện tại: {loc.get('lat')}, {loc.get('lng')}")
    lines += ["", "Lịch 7 ngày tới:", *days]
    lines += ["", "Ứng dụng đã cài:", *installed_app_lines(context.get("installed_apps") or {}, apps_info, drop_apps)]
    return "\n".join(lines)


def tool_spec(tool: dict) -> dict:
    """Catalog tool -> function spec. Bỏ các khóa x-*: x-ref thành gợi ý trong description, x-one-of ghi vào mô tả tool."""
    params = tool["parameters"]
    props = {}
    for name, p in params.get("properties", {}).items():
        q = {k: v for k, v in p.items() if not k.startswith("x-")}
        if "x-ref" in p and not q.get("description"):
            q["description"] = f"ID lấy từ dữ liệu {p['x-ref']}"
        props[name] = q
    desc = tool["description"]
    if params.get("x-one-of"):
        desc += ". Cần ít nhất một trong các nhóm tham số: " + " | ".join(", ".join(g) for g in params["x-one-of"])
    if tool.get("returns"):
        desc += f". Trả về: {tool['returns']}"
    return {"type": "function", "function": {"name": tool["name"], "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": params.get("required", [])}}}


def build_tools(installed_apps: dict, catalog_tools: dict[str, dict], drop_apps=()) -> list[dict]:
    """Tool của các app đã cài (theo thứ tự trong installed_apps), bỏ drop_apps, luôn kèm sys.ask_missing_info ở cuối."""
    names = [t for app, a in installed_apps.items() if app not in drop_apps for t in a.get("tools", []) if t in catalog_tools]
    return [tool_spec(catalog_tools[n]) for n in names if n != ASK_TOOL] + [tool_spec(catalog_tools[ASK_TOOL])]


def format_result(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def tool_call_message(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


def tool_result_message(name: str, result: dict) -> dict:
    return {"role": "tool", "name": name, "content": format_result(result)}


def history_messages(history: list[dict], user_text: Callable[[dict], str]) -> list[dict]:
    """Các lượt trước: user (text từ user_text(turn), tức audio_text hoặc transcript ASR), assistant (tool call + kết quả, rồi text)."""
    msgs = []
    for turn in history or []:
        if turn["role"] == "user":
            msgs.append({"role": "user", "content": user_text(turn)})
            continue
        for call, result in zip(turn.get("tool_calls") or [], turn.get("tool_results") or []):
            msgs.append(tool_call_message(call["name"], call.get("arguments") or {}))
            msgs.append(tool_result_message(call["name"], result))
        msgs.append({"role": "assistant", "content": turn.get("text", "")})
    return msgs


def build_prompt_messages(sample: dict, user_text: Callable[[dict], str], apps_info: dict[str, dict],
                          drop_apps=()) -> list[dict]:
    """system + history + yêu cầu cuối. user_text nhận dict có "audio" (và "audio_text" ở train).
    apps_info lấy từ catalog của split đang chạy (load_catalog_apps); drop_apps phải trùng với build_tools."""
    final = {"audio": sample["audio"], "audio_text": sample.get("audio_text")}
    return ([{"role": "system", "content": build_system_prompt(sample["context"], apps_info, drop_apps)}]
            + history_messages(sample.get("history"), user_text)
            + [{"role": "user", "content": user_text(final)}])
