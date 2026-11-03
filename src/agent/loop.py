"""Vòng agent chạy trên ToolEnv của BTC: LLM sinh một tool call mỗi lượt -> env.call -> trả kết quả -> lặp đến khi LLM trả
lời bằng text. Dùng chung messages/tools với dữ liệu SFT (src/agent/prompt.py) để prompt lúc chạy khớp lúc train.

Không phụ thuộc torch: phần sinh text là một hàm `policy(episodes) -> list[str]` do notebook cung cấp (render chat template
+ model.generate theo batch). Nhờ vậy test được trên CPU bằng policy giả (xem src/agent/test_loop.py).

Chặn lỗi (call bị từ chối vẫn tính vào bài nộp và làm giảm precision):
- Trước khi gọi, kiểm tra bằng env.validate (không bị ghi log). Sai schema hoặc JSON hỏng -> không gọi env, trả lỗi cho model
  như một kết quả tool {"ok": false, "error": ...} để nó sửa; quá max_invalid lần thì dừng.
- Tool không có trong danh sách của mẫu -> coi như sai schema.
- Gọi lại y hệt một call đọc đã gọi, hoặc call ghi lần thứ 3 -> dừng (tránh lặp vô hạn).
- Quá max_steps call -> dừng.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from src.agent.prompt import build_prompt_messages, build_tools, tool_call_message, tool_result_message
from src.data.common import import_tool_env, load_catalog_apps, load_catalog_tools

MAX_SAME_WRITE = 2
TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


@dataclass
class Episode:
    sample: dict
    env: object
    messages: list[dict]
    tools: list[dict]
    allowed: set[str]
    calls: list[dict] = field(default_factory=list)  # call đã gửi vào env (đúng thứ tự)
    invalid: int = 0
    done: bool = False
    stop_reason: str = ""
    trace: list[dict] = field(default_factory=list)  # {"gen": text, "action": ...} từng lượt, để phân tích lỗi

    @property
    def id(self) -> str:
        return self.sample["id"]


def parse_generation(text: str):
    """-> ("text", None) nếu không gọi tool; ("call", {"name", "arguments"}); ("bad", lý do) nếu tool call hỏng."""
    m = TOOL_RE.search(text)
    if not m:
        return "text", None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return "bad", "tool call không phải JSON hợp lệ"
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return "bad", "tool call thiếu name"
    args = obj.get("arguments", {})
    if isinstance(args, str):  # một số model trả arguments dạng chuỗi JSON
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return "bad", "arguments không phải JSON hợp lệ"
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return "bad", "arguments phải là object"
    return "call", {"name": obj["name"], "arguments": args}


def _key(call: dict) -> str:
    return call["name"] + json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True)


class AgentRunner:
    def __init__(self, data_root, split: str, user_text: Callable[[dict], str], max_steps: int = 10, max_invalid: int = 2):
        self.data_root, self.split, self.user_text = str(data_root), split, user_text
        self.max_steps, self.max_invalid = max_steps, max_invalid
        self.ToolEnv = import_tool_env(data_root, split)
        self.catalog_tools = load_catalog_tools(data_root, split)
        self.apps_info = load_catalog_apps(data_root, split)

    def new_episode(self, sample: dict) -> Episode:
        """Tạo ToolEnv cho mẫu. Chỉ gọi MỘT lần cho mỗi mẫu (luật BTC: không tạo lại env để thử)."""
        tools = build_tools(sample["context"]["installed_apps"], self.catalog_tools)
        return Episode(
            sample=sample,
            env=self.ToolEnv(self.split, sample["id"], root=self.data_root),
            messages=build_prompt_messages(sample, self.user_text, self.apps_info),
            tools=tools,
            allowed={t["function"]["name"] for t in tools},
        )

    def _finish(self, ep: Episode, reason: str):
        ep.done, ep.stop_reason = True, reason

    def apply(self, ep: Episode, text: str):
        """Xử lý một lượt sinh của model cho ep."""
        kind, val = parse_generation(text)
        if kind == "text":
            ep.trace.append({"gen": text, "action": "stop"})
            return self._finish(ep, "text")
        if kind == "bad":
            return self._reject(ep, text, None, val)
        call = val
        n_same = sum(_key(c) == _key(call) for c in ep.calls)
        kind_of = (self.catalog_tools.get(call["name"]) or {}).get("kind")
        # gold có lặp y hệt nhưng hiếm: chỉ tool write, tối đa 2 lần (vd. loyalty.redeem 2 phần quà); tool đọc lặp = kẹt vòng
        if n_same >= (MAX_SAME_WRITE if kind_of == "write" else 1):
            ep.trace.append({"gen": text, "action": "stop: lặp call"})
            return self._finish(ep, "repeat")
        if call["name"] not in ep.allowed:
            return self._reject(ep, text, call, f"tool {call['name']} không có trong danh sách tool")
        v = ep.env.validate(call["name"], call["arguments"])
        if not v.get("ok"):
            return self._reject(ep, text, call, v.get("error") or "sai schema")
        result = ep.env.call(call["name"], call["arguments"])
        ep.calls.append(call)
        ep.trace.append({"gen": text, "action": "call", "ok": result.get("ok")})
        ep.messages.append(tool_call_message(call["name"], call["arguments"]))
        ep.messages.append(tool_result_message(call["name"], result))
        if len(ep.calls) >= self.max_steps:
            self._finish(ep, "max_steps")

    def _reject(self, ep: Episode, text: str, call: dict | None, error: str):
        """Call hỏng/sai schema: không gửi vào env (không bị tính), trả lỗi cho model như kết quả tool để nó sửa."""
        ep.invalid += 1
        ep.trace.append({"gen": text, "action": f"invalid: {error}"})
        if ep.invalid > self.max_invalid:
            return self._finish(ep, "invalid")
        name = call["name"] if call else "unknown"
        ep.messages.append(tool_call_message(name, call["arguments"] if call else {}))
        ep.messages.append(tool_result_message(name, {"ok": False, "error": error}))

    def run(self, samples: list[dict], policy: Callable[[list[Episode]], list[str]], log: Callable[[str], None] = print,
            ) -> list[Episode]:
        """Chạy tất cả mẫu theo từng vòng: mỗi vòng sinh một lượt cho mọi episode chưa xong (policy nhận cả batch)."""
        episodes = [self.new_episode(s) for s in samples]
        rnd = 0
        while True:
            active = [e for e in episodes if not e.done]
            if not active:
                break
            rnd += 1
            texts = policy(active)
            for ep, text in zip(active, texts):
                self.apply(ep, text)
            log(f"vòng {rnd}: {len(active)} mẫu đang chạy, còn {sum(not e.done for e in episodes)}")
        return episodes


def submission_row(ep: Episode) -> dict:
    """Dòng nộp bài: mọi call đã gọi (lấy từ env, kể cả call bị từ chối) + state_hash."""
    return {"id": ep.id, "tool_calls": ep.env.get_tool_calls(), "state_hash": ep.env.get_state_hash()}
