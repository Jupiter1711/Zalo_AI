"""Bước 1: chia train thành train / val_seen / val_unseen (chạy trên CPU, chỉ dùng thư viện chuẩn).

- val_unseen (leave-apps-out): mọi sample có đụng tới một app trong HELDOUT_APPS (trong gold, trong tool_name của
  sys.ask_missing_info, hoặc trong tool_calls của history). Mô phỏng việc public/private có app chưa gặp.
- val_seen: lấy ngẫu nhiên VAL_SEEN_FRAC phần còn lại, phân tầng theo (loại đầu ra, có history hay không).
- train: phần còn lại. Lưu ý cho bước 2: các sample train vẫn có thể *cài* app giữ ra (không dùng tới), khi dựng
  prompt SFT phải bỏ schema của các app đó để model không bao giờ thấy chúng.

Chạy từ thư mục gốc repo:
    python -m src.data.make_splits
    python -m src.data.make_splits --data-root "data/train+public_test/train+public_test" --out splits/splits.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from src.data.common import DEFAULT_DATA_ROOT, DEFAULT_SPLITS as DEFAULT_OUT, load_jsonl, output_type

SEED = 2026
VAL_SEEN_FRAC = 0.10
# app chỉ có trong train (không có trong public), mỗi app một nhóm, gần với các nhóm app mới của public_test
HELDOUT_APPS = ["bike_rental", "package_tracking", "loyalty", "fitness", "podcast_radio", "exchange_rate"]


def call_apps(calls: list[dict]) -> set[str]:
    """App của từng call; với sys.ask_missing_info thì lấy app của tool_name bị thiếu tham số."""
    apps = set()
    for c in calls or []:
        app = c["name"].split(".", 1)[0]
        if app == "sys":
            app = (c.get("arguments") or {}).get("tool_name", "").split(".", 1)[0]
        if app:
            apps.add(app)
    return apps


def touched_apps(sample: dict) -> set[str]:
    apps = call_apps(sample["label"]["tool_calls"])
    for turn in sample.get("history") or []:
        apps |= call_apps(turn.get("tool_calls"))
    return apps


def stratified_pick(ids_by_stratum: dict[tuple, list[str]], frac: float, rng: random.Random) -> set[str]:
    picked = set()
    for key in sorted(ids_by_stratum):
        ids = sorted(ids_by_stratum[key])
        rng.shuffle(ids)
        picked.update(ids[: round(len(ids) * frac)])
    return picked


def summarize(ids: list[str], info: dict[str, dict]) -> dict:
    return {
        "n": len(ids),
        "output_type": dict(collections.Counter(info[i]["type"] for i in ids).most_common()),
        "with_history": sum(info[i]["has_history"] for i in ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    samples = load_jsonl(args.data_root / "train" / "train.jsonl")
    catalog = json.load(open(args.data_root / "train" / "train_tools_catalog.json", encoding="utf-8"))
    kinds = {t["name"]: t["kind"] for a in catalog["apps"] for t in a["tools"]}
    unknown = set(HELDOUT_APPS) - {a["id"] for a in catalog["apps"]}
    if unknown:
        raise SystemExit(f"HELDOUT_APPS không có trong catalog train: {sorted(unknown)}")

    heldout = set(HELDOUT_APPS)



    
    info = {
        s["id"]: {"type": output_type(s, kinds), "has_history": bool(s.get("history")), "unseen": bool(touched_apps(s) & heldout)}
        for s in samples
    }

    val_unseen = sorted(i for i, x in info.items() if x["unseen"])
    rest = collections.defaultdict(list)
    for i, x in info.items():
        if not x["unseen"]:
            rest[(x["type"], x["has_history"])].append(i)
    val_seen = sorted(stratified_pick(rest, VAL_SEEN_FRAC, random.Random(SEED)))
    held = set(val_unseen) | set(val_seen)
    train = sorted(i for i in info if i not in held)

    # train samples còn *cài* app giữ ra (không dùng) -> bước 2 phải bỏ schema của chúng khỏi prompt
    installed_heldout = sum(bool(set(s["context"]["installed_apps"]) & heldout) for s in samples if s["id"] in set(train))

    out = {
        "seed": SEED,
        "val_seen_frac": VAL_SEEN_FRAC,
        "heldout_apps": HELDOUT_APPS,
        "stats": {
            "train": summarize(train, info),
            "val_seen": summarize(val_seen, info),
            "val_unseen": summarize(val_unseen, info),
            "train_with_heldout_installed": installed_heldout,
        },
        "train": train,
        "val_seen": val_seen,
        "val_unseen": val_unseen,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out["stats"], ensure_ascii=False, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
