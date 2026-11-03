"""Bước 3: manifest ASR (wav, transcript) theo đúng split của splits/splits.json.

Mỗi sample train có 1 audio lệnh cuối + audio của các lượt user trong history; tất cả audio của một sample
nằm chung một split, để audio của val không lọt vào train ASR. Thêm public_test.jsonl (không có text) để chạy
ASR hàng loạt lúc inference.

Mỗi dòng: {id, sample_id, turn, is_final, audio, duration, text, text_asr}
- audio: đường dẫn tương đối so với data root, vd. "train/audio/train_0006/train_0006.wav"
  (trên Kaggle: os.path.join(<thư mục dataset>, audio)).
- text: audio_text gốc (NFC), giữ hoa thường, dấu câu và chữ số như BTC ghi.
- text_asr: target cho ASR = vi_numbers.asr_target(text): chữ số đọc thành chữ ("50 nghìn" -> "năm mươi nghìn",
  SĐT đọc từng số), chữ thường, bỏ dấu câu, "-" thành khoảng trắng. Mã chữ+số (VTV6, RON95, RV219) giữ nguyên.
  Lúc inference: output ASR -> vi_numbers.llm_input() -> LLM.
- turn: "final" cho lệnh cuối, "u1", "u2"... theo thứ tự lượt user trong history.

Chạy: python -m src.data.build_asr_manifest   (chỉ thư viện chuẩn, CPU)
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
import wave
from pathlib import Path

from src.data.common import DEFAULT_DATA_ROOT, DEFAULT_SPLITS, load_jsonl
from src.text.vi_numbers import asr_target

DEFAULT_OUT_DIR = Path("data/processed/asr")
EXPECTED_FORMAT = (16000, 1, 2)  # sample rate, số kênh, số byte mỗi mẫu
DIGIT_RE = re.compile(r"\d")


def wav_info(path: Path) -> tuple[tuple[int, int, int], float]:
    with wave.open(str(path)) as w:
        return (w.getframerate(), w.getnchannels(), w.getsampwidth()), w.getnframes() / w.getframerate()


def sample_rows(sample: dict, split_dir: str) -> list[dict]:
    """Các dòng manifest của một sample: lượt user trong history trước, lệnh cuối sau."""
    turns = [t for t in sample.get("history") or [] if t["role"] == "user"]
    items = [(f"u{i}", t, False) for i, t in enumerate(turns, 1)] + [("final", sample, True)]
    rows = []
    for turn, t, is_final in items:
        text = t.get("audio_text")
        rows.append({
            "id": Path(t["audio"]).stem,
            "sample_id": sample["id"],
            "turn": turn,
            "is_final": is_final,
            "audio": f"{split_dir}/{t['audio']}",
            "text": None if text is None else unicodedata.normalize("NFC", text),
            "text_asr": None if text is None else asr_target(text),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    train = {s["id"]: s for s in load_jsonl(args.data_root / "train" / "train.jsonl")}
    public = load_jsonl(args.data_root / "public_test" / "public_test.jsonl")
    splits = json.load(open(args.splits, encoding="utf-8"))
    jobs = [(name, [train[i] for i in splits[name]], "train") for name in ("train", "val_seen", "val_unseen")]
    jobs.append(("public_test", public, "public_test"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    errors, seen_ids, seen_audio = [], set(), set()
    for name, samples, split_dir in jobs:
        rows = [r for s in samples for r in sample_rows(s, split_dir)]
        n_digit = n_digit_asr = 0
        with open(args.out_dir / f"{name}.jsonl", "w", encoding="utf-8") as fout:
            for r in rows:
                path = args.data_root / r["audio"]
                if not path.exists():
                    errors.append((r["id"], "thiếu file wav"))
                    continue
                fmt, r["duration"] = wav_info(path)
                r["duration"] = round(r["duration"], 3)
                if fmt != EXPECTED_FORMAT:
                    errors.append((r["id"], f"sai định dạng wav {fmt}"))
                if split_dir == "train" and not r["text_asr"]:
                    errors.append((r["id"], "transcript rỗng"))
                if r["id"] in seen_ids or r["audio"] in seen_audio:
                    errors.append((r["id"], "audio trùng giữa các dòng"))
                seen_ids.add(r["id"])
                seen_audio.add(r["audio"])
                n_digit += bool(r["text"] and DIGIT_RE.search(r["text"]))
                n_digit_asr += bool(r["text_asr"] and DIGIT_RE.search(r["text_asr"]))
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        durs = sorted(r["duration"] for r in rows if "duration" in r)
        n_final = sum(r["is_final"] for r in rows)
        print(f"{name}: {len(samples)} sample, {len(rows)} audio ({n_final} lệnh cuối, {len(rows) - n_final} lượt history)"
              f" | {sum(durs) / 3600:.2f} giờ | dài nhất {durs[-1]:.1f}s"
              + (f" | có chữ số: text {n_digit}, text_asr {n_digit_asr} (chỉ còn mã như vtv6)" if split_dir == "train" else ""))
    print(f"lỗi tự kiểm tra: {len(errors)}", errors[:10])
    print(f"-> {args.out_dir}/{{train,val_seen,val_unseen,public_test}}.jsonl")


if __name__ == "__main__":
    main()
