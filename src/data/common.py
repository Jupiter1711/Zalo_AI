"""Hàm dùng chung cho các script xử lý dữ liệu (chỉ thư viện chuẩn)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = Path("data/train+public_test/train+public_test") 
DEFAULT_SPLITS = Path("splits/splits.json")
ASK_TOOL = "sys.ask_missing_info"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_catalog(data_root: Path, split: str) -> dict:
    return json.load(open(data_root / split / f"{split}_tools_catalog.json", encoding="utf-8"))


def load_catalog_tools(data_root: Path, split: str) -> dict[str, dict]:
    """{tool name: tool spec} từ <split>_tools_catalog.json."""
    return {t["name"]: t for a in _load_catalog(data_root, split)["apps"] for t in a["tools"]}


def load_catalog_apps(data_root: Path, split: str) -> dict[str, dict]:
    """{app id: {name_vi, group, description}} từ <split>_tools_catalog.json."""
    return {a["id"]: {k: a.get(k, "") for k in ("name_vi", "group", "description")}
            for a in _load_catalog(data_root, split)["apps"]}


def import_tool_env(data_root: Path, split: str):
    """ToolEnv của đúng split (mỗi split có bản zaic_tools riêng)."""
    sys.path.insert(0, str(data_root / split))
    from zaic_tools import ToolEnv
    return ToolEnv


def output_type(sample: dict, kinds: dict[str, str]) -> str:
    """oos (gold rỗng) / ask (có sys.ask_missing_info) / read_only / write."""
    calls = sample["label"]["tool_calls"]
    if not calls:
        return "oos"
    if any(c["name"] == ASK_TOOL for c in calls):
        return "ask"
    if all(kinds.get(c["name"]) == "read" for c in calls):
        return "read_only"
    return "write"
