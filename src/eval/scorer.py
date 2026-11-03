"""Bước 5: local scorer, mô phỏng cách BTC chấm, chạy trên các split có gold (train / val_seen / val_unseen).

Cách chấm (README của zaic_tools + bản tóm tắt đề; công thức gốc trong đề bị mất nên phần F1 là GIẢ ĐỊNH):
- Cổng state: replay tool_calls của bài nộp trên state ban đầu bằng zaic_tools, so state_hash với state_hash của gold
  (replay gold). Sai -> sample 0 điểm. state_hash đã bỏ _meta, app thông tin và trường text tự do.
- Khớp call: cùng tên tool; tham số fuzzy (x-match: fuzzy trong catalog) có token-F1 > 0.6; tham số khác bằng nhau sau
  chuẩn hóa như zaic_tools (chuỗi chữ thường + gộp khoảng trắng, "700 cm" == "700cm", list scalar so như tập hợp).
  Không xét thứ tự; mỗi call khớp tối đa một call gold (ghép cặp cực đại). Giá trị rỗng (None, "", []) coi như không truyền.
- Mọi call trong bài nộp đều tính, kể cả call bị từ chối vì sai schema.
- P = M/|pred|, R = M/|gold|, F1 = 2PR/(P+R). Gold rỗng: 1 điểm nếu bài nộp cũng rỗng, ngược lại 0.
- Điểm sample = [hash đúng] × F1; điểm cuối = trung bình. Tie-break của BTC: F1 bỏ qua hash ("f1_no_hash").

Dùng như thư viện:  Scorer(data_root).score_sample(sample_id, pred_calls) -> dict
Dòng lệnh:
    python -m src.eval.scorer --submission result/val_unseen.jsonl --subset val_unseen
    python -m src.eval.scorer --oracle --subset val_unseen      # tự kiểm tra: nộp đúng gold phải được 1.0
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from src.data.common import DEFAULT_DATA_ROOT, DEFAULT_SPLITS, import_tool_env, load_jsonl, output_type

FUZZY_THRESHOLD = 0.6
EMPTY = (None, "", [], {})


class Scorer:
    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT, split: str = "train"):
        self.data_root, self.split = Path(data_root), split
        self.ToolEnv = import_tool_env(self.data_root, split)
        from zaic_tools.utils import _canon_scalar, _tokens  # cùng chuẩn hóa với BTC
        self._canon_scalar, self._tokens = _canon_scalar, _tokens
        catalog = json.load(open(self.data_root / split / f"{split}_tools_catalog.json", encoding="utf-8"))
        self.fuzzy = {(t["name"], p) for a in catalog["apps"] for t in a["tools"]
                      for p, s in t["parameters"].get("properties", {}).items() if s.get("x-match") == "fuzzy"}
        self.kinds = {t["name"]: t["kind"] for a in catalog["apps"] for t in a["tools"]}
        self.samples = {s["id"]: s for s in load_jsonl(self.data_root / split / f"{split}.jsonl")}
        self._gold_hash: dict[str, str] = {}

    # ------------------------------------------------------------------ replay
    def replay(self, sample_id: str, calls: list[dict]) -> tuple[str, list[dict]]:
        """state_hash và kết quả sau khi chạy calls trên một ToolEnv mới (như BTC replay bài nộp)."""
        env = self.ToolEnv(self.split, sample_id, root=str(self.data_root))
        results = [env.call(c.get("name", ""), c.get("arguments") or {}) for c in calls]
        return env.get_state_hash(), results

    def gold_hash(self, sample_id: str) -> str:
        if sample_id not in self._gold_hash:
            self._gold_hash[sample_id] = self.replay(sample_id, self.samples[sample_id]["label"]["tool_calls"])[0]
        return self._gold_hash[sample_id]

    # ------------------------------------------------------------------ so khớp
    def token_f1(self, a, b) -> float:
        ta, tb = self._tokens(a), self._tokens(b)
        if not ta and not tb:
            return 1.0
        common = len(ta & tb)
        if not common:
            return 0.0
        p, r = common / len(ta), common / len(tb)
        return 2 * p * r / (p + r)

    def _canon(self, v):
        if isinstance(v, dict):
            return {k: self._canon(x) for k, x in v.items() if x not in EMPTY}
        if isinstance(v, list):
            out = [self._canon(x) for x in v]
            if all(not isinstance(x, (dict, list)) for x in out):
                out = sorted(out, key=lambda x: (type(x).__name__, str(x)))
            return out
        return self._canon_scalar(v)

    def arg_mismatch(self, name: str, pred: dict, gold: dict) -> str | None:
        """None nếu hai call cùng tên khớp nhau, ngược lại tên tham số lệch đầu tiên."""
        pred = {k: v for k, v in (pred or {}).items() if v not in EMPTY}
        gold = {k: v for k, v in (gold or {}).items() if v not in EMPTY}
        for k in sorted(set(pred) | set(gold)):
            if k not in pred or k not in gold:
                return k
            if (name, k) in self.fuzzy:
                if self.token_f1(pred[k], gold[k]) <= FUZZY_THRESHOLD:
                    return k
            elif type(pred[k]) is not type(gold[k]) and not (isinstance(pred[k], (int, float)) and isinstance(gold[k], (int, float))
                                                          and not isinstance(pred[k], bool) and not isinstance(gold[k], bool)):
                return k
            elif self._canon(pred[k]) != self._canon(gold[k]):
                return k
        return None

    def match(self, pred: list[dict], gold: list[dict]) -> list[int | None]:
        """Ghép cặp cực đại pred <-> gold (thuật toán Kuhn). Trả về với mỗi pred: chỉ số gold được ghép hoặc None."""
        ok = [[p.get("name") == g.get("name") and self.arg_mismatch(g["name"], p.get("arguments"), g.get("arguments")) is None
               for g in gold] for p in pred]
        gold_of: list[int | None] = [None] * len(pred)
        pred_of: list[int | None] = [None] * len(gold)

        def augment(i: int, seen: set) -> bool:
            for j in range(len(gold)):
                if ok[i][j] and j not in seen:
                    seen.add(j)
                    if pred_of[j] is None or augment(pred_of[j], seen):
                        pred_of[j], gold_of[i] = i, j
                        return True
            return False

        for i in range(len(pred)):
            augment(i, set())
        return gold_of

    # ------------------------------------------------------------------ chấm một sample
    def score_sample(self, sample_id: str, pred_calls: list[dict], submitted_hash: str | None = None) -> dict:
        sample = self.samples[sample_id]
        gold = sample["label"]["tool_calls"]
        pred_calls = pred_calls or []
        pred_hash, results = self.replay(sample_id, pred_calls)
        hash_ok = pred_hash == self.gold_hash(sample_id)
        gold_of = self.match(pred_calls, gold)
        m = sum(j is not None for j in gold_of)
        if not gold:
            f1 = 1.0 if not pred_calls else 0.0
        elif m == 0:
            f1 = 0.0
        else:
            p, r = m / len(pred_calls), m / len(gold)
            f1 = 2 * p * r / (p + r)
        matched_gold = {j for j in gold_of if j is not None}
        extra = [c for c, j in zip(pred_calls, gold_of) if j is None]
        missed = [g for j, g in enumerate(gold) if j not in matched_gold]
        return {
            "id": sample_id,
            "type": output_type(sample, self.kinds),
            "score": f1 if hash_ok else 0.0,
            "f1_no_hash": f1,
            "hash_ok": hash_ok,
            "n_pred": len(pred_calls),
            "n_gold": len(gold),
            "n_match": m,
            "n_rejected": sum(r.get("ok") is False for r in results),
            "submitted_hash_ok": None if submitted_hash is None else submitted_hash == pred_hash,
            "extra": [self._explain(c, gold) for c in extra],
            "missed": [g["name"] for g in missed],
        }

    def _explain(self, call: dict, gold: list[dict]) -> dict:
        """Call thừa: vì sao không khớp (sai tên tool, hay lệch tham số nào so với gold cùng tên)."""
        same = [g for g in gold if g["name"] == call.get("name")]
        if not same:
            return {"name": call.get("name"), "why": "không có trong gold"}
        bad = sorted({self.arg_mismatch(g["name"], call.get("arguments"), g.get("arguments")) for g in same} - {None})
        return {"name": call.get("name"), "why": "lệch tham số " + ", ".join(bad)}


def summarize(rows: list[dict]) -> dict:
    def agg(rs):
        n = len(rs)
        return {"n": n, "score": sum(r["score"] for r in rs) / n, "f1_no_hash": sum(r["f1_no_hash"] for r in rs) / n,
                "hash_acc": sum(r["hash_ok"] for r in rs) / n} if n else {"n": 0}
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    # "không có trong gold" (sai/thừa tool) | "lệch <tham số>" (đúng tool, sai tham số)
    extra_why = collections.Counter(e["why"].replace("lệch tham số ", "lệch ") for r in rows for e in r["extra"])
    return {
        "all": agg(rows),
        "by_type": {t: agg(rs) for t, rs in sorted(by_type.items())},
        "calls": {"pred": sum(r["n_pred"] for r in rows), "gold": sum(r["n_gold"] for r in rows),
                  "match": sum(r["n_match"] for r in rows), "rejected_schema": sum(r["n_rejected"] for r in rows)},
        "top_extra_reasons": extra_why.most_common(10),
        "top_missed_tools": collections.Counter(t for r in rows for t in r["missed"]).most_common(10),
        "submitted_hash_mismatch": sum(r["submitted_hash_ok"] is False for r in rows),
    }


def print_summary(s: dict, missing: int):
    a = s["all"]
    print(f"Điểm: {a['score']:.4f} | F1 bỏ qua hash: {a['f1_no_hash']:.4f} | hash đúng: {a['hash_acc']:.1%} | {a['n']} sample")
    for t, x in s["by_type"].items():
        print(f"  {t:<10} n={x['n']:<4} điểm {x['score']:.4f}  F1 {x['f1_no_hash']:.4f}  hash {x['hash_acc']:.1%}")
    c = s["calls"]
    print(f"Call: nộp {c['pred']}, gold {c['gold']}, khớp {c['match']}, bị từ chối vì sai schema {c['rejected_schema']}")
    if s["top_extra_reasons"]:
        print("Call thừa/không khớp:", s["top_extra_reasons"])
    if s["top_missed_tools"]:
        print("Tool gold bị thiếu:", s["top_missed_tools"])
    if s["submitted_hash_mismatch"]:
        print(f"CẢNH BÁO: {s['submitted_hash_mismatch']} dòng có state_hash nộp khác hash khi replay tool_calls")
    if missing:
        print(f"CẢNH BÁO: {missing} sample không có trong bài nộp (tính 0 điểm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, help="jsonl {id, tool_calls, state_hash}")
    ap.add_argument("--oracle", action="store_true", help="chấm chính gold (tự kiểm tra scorer, phải được 1.0)")
    ap.add_argument("--subset", default="val_unseen", help="tên phần trong splits.json (train/val_seen/val_unseen) hoặc 'all'")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--details", type=Path, help="ghi kết quả từng sample ra jsonl để phân tích lỗi")
    args = ap.parse_args()
    if not args.submission and not args.oracle:
        ap.error("cần --submission hoặc --oracle")

    scorer = Scorer(args.data_root, "train")
    if args.subset == "all":
        ids = list(scorer.samples)
    else:
        ids = json.load(open(args.splits, encoding="utf-8"))[args.subset]

    if args.oracle:
        sub = {i: {"tool_calls": scorer.samples[i]["label"]["tool_calls"], "state_hash": None} for i in ids}
    else:
        sub = {r["id"]: r for r in load_jsonl(args.submission)}
        extra_ids = set(sub) - set(ids)
        if extra_ids:
            print(f"bỏ qua {len(extra_ids)} dòng không thuộc subset {args.subset}", file=sys.stderr)

    rows, missing = [], 0
    for i in ids:
        if i not in sub:
            missing += 1
        r = sub.get(i) or {"tool_calls": None, "state_hash": None}
        row = scorer.score_sample(i, r.get("tool_calls"), r.get("state_hash"))
        if i not in sub:  # thiếu dòng: 0 điểm kể cả khi gold rỗng
            row.update(score=0.0, f1_no_hash=0.0, hash_ok=False)
        rows.append(row)

    print_summary(summarize(rows), missing)
    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        with open(args.details, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"-> {args.details}")


if __name__ == "__main__":
    main()
