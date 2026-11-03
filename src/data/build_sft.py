"""Bước 2: dựng dữ liệu SFT multi-turn cho LLM từ train + splits/splits.json (chạy trên CPU).

Mỗi dòng output: {"id", "split", "type", "tools", "messages", "loss_from"}
- tools/messages theo format chat OpenAI/Qwen, dựng bằng src/agent/prompt.py (cùng code với inference).
- Phần cần học là các message từ chỉ số loss_from trở đi: chuỗi gold tool call (mỗi call một lượt assistant, kèm kết quả
  thật từ zaic_tools) rồi một câu trả lời text để dừng. Các lượt assistant trong history là bối cảnh.
- Sample của split train bỏ schema các app giữ ra (heldout_apps) để val_unseen thật sự là app chưa gặp.
- Text user (mặc định --user-text itn): audio_text -> vi_numbers.asr_target -> llm_input, tức đúng dạng LLM nhận lúc
  inference (output ASR chữ thường, không dấu câu, số đã qua ITN). --user-text raw để dùng audio_text gốc.
- Tự kiểm tra: replay các call lấy từ messages trên một ToolEnv mới phải ra đúng final_state và state_hash của gold.

Chạy từ thư mục gốc repo:
    python -m src.data.build_sft
    python -m src.data.build_sft --transcripts data/processed/asr/train_transcripts.json   # dùng transcript ASR
    # v2: mỗi sample 2 bản (text sạch + text ASR k-fold), ghi ra thư mục khác
    python -m src.data.build_sft --mix --out-dir data/processed/sft_v2         --transcripts data/hf_outputs/outputs/asr_folds/n2_fold0.json data/hf_outputs/outputs/asr_folds/n2_fold1.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from src.agent.prompt import FINAL_TEXT, OOS_TEXT, build_prompt_messages, build_tools, tool_call_message, tool_result_message
from src.text.vi_numbers import asr_target, llm_input
from src.data.common import DEFAULT_DATA_ROOT, DEFAULT_SPLITS, import_tool_env, load_catalog_apps, load_catalog_tools, load_jsonl, output_type

DEFAULT_OUT_DIR = Path("data/processed/sft")
CHARS_PER_TOKEN = 3.0  # ước lượng thô cho tiếng Việt với tokenizer Qwen; đo chính xác bằng --tokenizer


def build_record(sample, split, catalog_tools, apps_info, drop_apps, user_text, ToolEnv, data_root, kinds) -> dict:
    messages = build_prompt_messages(sample, user_text, apps_info, drop_apps)
    loss_from = len(messages)
    env = ToolEnv("train", sample["id"], root=str(data_root))
    calls = sample["label"]["tool_calls"]
    for c in calls:
        messages.append(tool_call_message(c["name"], c["arguments"]))
        messages.append(tool_result_message(c["name"], env.call(c["name"], c["arguments"])))
    messages.append({"role": "assistant", "content": FINAL_TEXT if calls else OOS_TEXT})
    return {
        "id": sample["id"],
        "split": split,
        "type": output_type(sample, kinds),
        "tools": build_tools(sample["context"]["installed_apps"], catalog_tools, drop_apps),
        "messages": messages,
        "loss_from": loss_from,
        "_hash": env.get_state_hash(),
    }


def check_record(rec, sample, ToolEnv, data_root) -> str | None:
    """Replay call lấy từ messages[loss_from:] -> phải khớp gold (call, final_state, hash)."""
    calls = [tc["function"] for m in rec["messages"][rec["loss_from"]:] for tc in m.get("tool_calls") or []]
    calls = [{"name": c["name"], "arguments": c["arguments"]} for c in calls]
    if calls != sample["label"]["tool_calls"]:
        return "call trong messages khác gold"
    env = ToolEnv("train", sample["id"], root=str(data_root))
    for c in calls:
        env.call(c["name"], c["arguments"])
    state = env.get_state()
    if any(state.get(app) != st for app, st in sample["label"]["final_state"].items()):
        return "replay không ra final_state"
    if env.get_state_hash() != rec["_hash"]:
        return "state_hash không khớp"
    if rec["tools"][-1]["function"]["name"] != "sys.ask_missing_info":
        return "thiếu sys.ask_missing_info trong tools"
    names = {t["function"]["name"] for t in rec["tools"]}
    if any(c["name"] not in names for c in calls):
        return "gold gọi tool không có trong tools"
    system = rec["messages"][0]["content"]
    listed = {ln[2:].split(" ", 1)[0].rstrip(":") for ln in system.split("Ứng dụng đã cài:\n", 1)[1].splitlines()}
    if listed != {n.split(".", 1)[0] for n in names} - {"sys"}:
        return "app trong system prompt khác app trong tools"
    return None


def length_stats(values: list[int]) -> dict:
    v = sorted(values)
    return {"p50": v[len(v) // 2], "p90": v[int(len(v) * 0.9)], "p99": v[int(len(v) * 0.99)], "max": v[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--transcripts", type=Path, nargs="+",
                    help="một hoặc nhiều JSON {audio path: text} (hoặc {'transcripts': {...}} của train_asr k-fold)")
    ap.add_argument("--mix", action="store_true",
                    help="mỗi sample 2 record: text sạch (itn) và text ASR (--transcripts); field text_src = clean/asr")
    ap.add_argument("--user-text", choices=["itn", "raw"], default="itn",
                    help="itn (mặc định): text user giống output ASR sau ITN, khớp lúc inference; raw: audio_text gốc")
    ap.add_argument("--tokenizer", help="tên/đường dẫn tokenizer HF để đếm token chính xác (cần transformers)")
    args = ap.parse_args()

    samples = {s["id"]: s for s in load_jsonl(args.data_root / "train" / "train.jsonl")}
    splits = json.load(open(args.splits, encoding="utf-8"))
    catalog_tools = load_catalog_tools(args.data_root, "train")
    apps_info = load_catalog_apps(args.data_root, "train")
    kinds = {n: t["kind"] for n, t in catalog_tools.items()}
    ToolEnv = import_tool_env(args.data_root, "train")

    transcripts = {}
    for path in args.transcripts or []:
        data = json.load(open(path, encoding="utf-8"))
        transcripts.update(data.get("transcripts", data))
    if args.mix and not transcripts:
        ap.error("--mix cần --transcripts")
    # output ASR đi qua đúng đường như lúc inference (asr_target -> llm_input, giống notebooks/infer.py)
    asr_text = lambda turn: llm_input(asr_target(transcripts[turn["audio"]]))  # noqa: E731
    if args.user_text == "itn":  # transcript gold giả làm output ASR hoàn hảo: số -> chữ -> ITN
        clean_text = lambda turn: llm_input(asr_target(turn["audio_text"]))  # noqa: E731
    else:
        clean_text = lambda turn: turn["audio_text"]  # noqa: E731
    if args.mix:
        variants = [("clean", clean_text), ("asr", asr_text)]
    elif transcripts:
        variants = [("asr", asr_text)]
    else:
        variants = [("clean", clean_text)]
    print("biến thể text user:", [v for v, _ in variants], f"| {len(transcripts)} transcript ASR")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for split in ("train", "val_seen", "val_unseen"):
        drop_apps = set(splits["heldout_apps"]) if split == "train" else set()
        sizes, types, n_tools = [], collections.Counter(), []
        with open(args.out_dir / f"{split}.jsonl", "w", encoding="utf-8") as fout:
            for sid, (src, user_text) in ((i, v) for i in splits[split] for v in variants):
                rec = build_record(samples[sid], split, catalog_tools, apps_info, drop_apps, user_text, ToolEnv,
                                   args.data_root, kinds)
                rec["text_src"] = src
                err = check_record(rec, samples[sid], ToolEnv, args.data_root)
                if err:
                    errors.append((sid, err))
                del rec["_hash"]
                if tokenizer is not None:
                    sizes.append(len(tokenizer.apply_chat_template(rec["messages"], tools=rec["tools"], tokenize=True)))
                else:
                    sizes.append(round(len(json.dumps([rec["tools"], rec["messages"]], ensure_ascii=False)) / CHARS_PER_TOKEN))
                types[rec["type"]] += 1
                n_tools.append(len(rec["tools"]))
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        unit = "token" if tokenizer is not None else f"token ước lượng (ký tự/{CHARS_PER_TOKEN:g})"
        print(f"{split}: {len(sizes)} mẫu {dict(types)} | số tool/mẫu {length_stats(n_tools)} | {unit} {length_stats(sizes)}"
              f" | >16384: {sum(s > 16384 for s in sizes)}")
    print(f"lỗi tự kiểm tra: {len(errors)}", errors[:10])
    print(f"-> {args.out_dir}/{{train,val_seen,val_unseen}}.jsonl")


if __name__ == "__main__":
    main()
