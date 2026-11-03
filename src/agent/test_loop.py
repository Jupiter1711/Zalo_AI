"""Test vòng agent bằng policy giả (không cần GPU). Chạy từ gốc repo: python -m unittest src.agent.test_loop -v"""
import json
import unittest
import warnings

from src.agent.loop import AgentRunner, parse_generation, submission_row
from src.data.common import DEFAULT_DATA_ROOT, DEFAULT_SPLITS
from src.eval.scorer import Scorer
from src.text.vi_numbers import asr_target, llm_input

warnings.simplefilter("ignore", ResourceWarning)
SCORER = Scorer()
IDS = json.load(open(DEFAULT_SPLITS, encoding="utf-8"))["val_unseen"]
SAMPLES = [SCORER.samples[i] for i in IDS]


def runner(**kw):
    return AgentRunner(DEFAULT_DATA_ROOT, "train", lambda t: llm_input(asr_target(t["audio_text"])), **kw)


def fmt(call):
    return "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"


def oracle(episodes):
    """Sinh lần lượt các gold call (đúng format Qwen), hết thì trả lời text."""
    out = []
    for ep in episodes:
        gold = ep.sample["label"]["tool_calls"]
        k = len(ep.calls)
        out.append(fmt(gold[k]) if k < len(gold) else "Đã xong.")
    return out


class TestLoop(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)

    def test_parse(self):
        self.assertEqual(parse_generation("Đã xong."), ("text", None))
        self.assertEqual(parse_generation(fmt({"name": "a.b", "arguments": {"x": 1}}))[1], {"name": "a.b", "arguments": {"x": 1}})
        self.assertEqual(parse_generation('<tool_call>\n{"name": "a.b", "arguments": "{\\"x\\": 1}"}\n</tool_call>')[1]["arguments"], {"x": 1})
        self.assertEqual(parse_generation('<tool_call>\n{"name": "a.b"}')[1], {"name": "a.b", "arguments": {}})  # thiếu thẻ đóng
        self.assertEqual(parse_generation("<tool_call>\n{hỏng\n</tool_call>")[0], "bad")

    def test_oracle_policy_scores_one(self):
        eps = runner().run(SAMPLES, oracle, log=lambda s: None)
        rows = [SCORER.score_sample(ep.id, r["tool_calls"], r["state_hash"]) for ep in eps for r in [submission_row(ep)]]
        bad = [r["id"] for r in rows if r["score"] != 1.0 or not r["submitted_hash_ok"]]
        self.assertEqual(bad, [])

    def test_invalid_call_not_submitted(self):
        s = next(x for x in SAMPLES if x["label"]["tool_calls"])
        r = runner(max_invalid=1)
        ep = r.new_episode(s)
        r.apply(ep, fmt({"name": "khong.ton_tai", "arguments": {}}))
        self.assertFalse(ep.done)
        self.assertEqual(ep.messages[-1]["role"], "tool")
        r.apply(ep, fmt({"name": s["label"]["tool_calls"][0]["name"], "arguments": {"__sai__": 1}}))
        self.assertTrue(ep.done)
        self.assertEqual(ep.stop_reason, "invalid")
        self.assertEqual(submission_row(ep)["tool_calls"], [])  # không call nào lọt vào env

    def test_repeat_stops(self):
        r = runner()
        s = next(x for x in SAMPLES if x["label"]["tool_calls"] and r.catalog_tools[x["label"]["tool_calls"][0]["name"]]["kind"] == "read")
        ep = r.new_episode(s)
        g = s["label"]["tool_calls"][0]
        r.apply(ep, fmt(g))
        r.apply(ep, fmt(g))
        self.assertEqual((ep.done, ep.stop_reason, len(ep.calls)), (True, "repeat", 1))
        w = next(x for x in SAMPLES if any(r.catalog_tools[c["name"]]["kind"] == "write" for c in x["label"]["tool_calls"]))
        ep = r.new_episode(w)
        g = next(c for c in w["label"]["tool_calls"] if r.catalog_tools[c["name"]]["kind"] == "write")
        for _ in range(3):
            r.apply(ep, fmt(g))
        self.assertEqual((ep.stop_reason, len(ep.calls)), ("repeat", 2))

    def test_max_steps(self):
        s = next(x for x in SAMPLES if len(x["label"]["tool_calls"]) >= 2)
        eps = runner(max_steps=1).run([s], oracle, log=lambda m: None)
        self.assertEqual((eps[0].stop_reason, len(eps[0].calls)), ("max_steps", 1))


if __name__ == "__main__":
    unittest.main()
