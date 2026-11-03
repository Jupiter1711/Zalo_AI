"""Test scorer trên dữ liệu train thật. Chạy từ gốc repo: python -m unittest src.eval.test_scorer -v"""
import copy
import random
import unittest
import warnings

from src.eval.scorer import Scorer

S = Scorer()
GOLD = {i: s["label"]["tool_calls"] for i, s in S.samples.items()}


def first(pred):
    return next(i for i, calls in GOLD.items() if pred(calls))


def has_write(calls):
    return any(S.kinds.get(c["name"]) == "write" for c in calls)


class TestScorer(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)  # zaic_tools của BTC mở state.json không đóng

    def test_oracle_all_samples(self):
        bad = [i for i, calls in GOLD.items() if S.score_sample(i, calls)["score"] != 1.0]
        self.assertEqual(bad, [])

    def test_empty_submission(self):
        oos = first(lambda c: not c)
        self.assertEqual(S.score_sample(oos, [])["score"], 1.0)
        self.assertEqual(S.score_sample(oos, [{"name": "weather.get_forecast", "arguments": {}}])["score"], 0.0)
        w = first(has_write)
        self.assertEqual(S.score_sample(w, [])["score"], 0.0)

    def test_order_does_not_matter(self):
        i = first(lambda c: len(c) >= 3 and not has_write(c))  # chỉ đọc: đảo thứ tự không đổi state
        calls = list(reversed(GOLD[i]))
        self.assertEqual(S.score_sample(i, calls)["score"], 1.0)

    def test_extra_call_lowers_precision(self):
        i = first(lambda c: len(c) == 2 and has_write(c))
        r = S.score_sample(i, GOLD[i] + [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "x.y", "params": ["a"]}}])
        self.assertTrue(r["hash_ok"])
        self.assertAlmostEqual(r["score"], 2 * (2 / 3) * 1 / (2 / 3 + 1))  # P=2/3, R=1
        self.assertEqual(len(r["extra"]), 1)

    def test_duplicate_call_matches_once(self):
        i = first(lambda c: len(c) == 1 and not has_write(c))
        r = S.score_sample(i, GOLD[i] * 2)
        self.assertEqual(r["n_match"], 1)
        self.assertAlmostEqual(r["score"], 2 * 0.5 * 1 / 1.5)

    def test_missing_write_fails_hash(self):
        i = first(lambda c: len(c) >= 2 and S.kinds.get(c[-1]["name"]) == "write")
        r = S.score_sample(i, GOLD[i][:-1])
        self.assertFalse(r["hash_ok"])
        self.assertEqual(r["score"], 0.0)
        self.assertGreater(r["f1_no_hash"], 0.0)

    def test_schema_error_counts_as_extra(self):
        i = first(lambda c: len(c) == 1)
        r = S.score_sample(i, GOLD[i] + [{"name": "khong.ton_tai", "arguments": {}}])
        self.assertEqual(r["n_rejected"], 1)
        self.assertAlmostEqual(r["f1_no_hash"], 2 * 0.5 * 1 / 1.5)

    def test_case_and_space_insensitive_exact(self):
        i = first(lambda c: any(c2["name"] == "contacts.search" for c2 in c))
        calls = copy.deepcopy(GOLD[i])
        for c in calls:
            if c["name"] == "contacts.search":
                c["arguments"]["query"] = "  " + c["arguments"]["query"].upper() + " "
        self.assertEqual(S.score_sample(i, calls)["score"], 1.0)

    def test_exact_param_mismatch(self):
        i = first(lambda c: any(c2["name"] == "contacts.search" for c2 in c))
        calls = copy.deepcopy(GOLD[i])
        for c in calls:
            if c["name"] == "contacts.search":
                c["arguments"]["query"] += " xyz"
        r = S.score_sample(i, calls)
        self.assertLess(r["f1_no_hash"], 1.0)
        self.assertIn("query", r["extra"][0]["why"])

    def test_fuzzy_param(self):
        # một call có tham số fuzzy dài (>= 6 token)
        def pick(c):
            return any((x["name"], k) in S.fuzzy and isinstance(v, str) and len(v.split()) >= 6
                       for x in c for k, v in x["arguments"].items())
        i = first(pick)
        j, k = next((n, k) for n, x in enumerate(GOLD[i]) for k, v in x["arguments"].items()
                    if (x["name"], k) in S.fuzzy and isinstance(v, str) and len(v.split()) >= 6)
        near = copy.deepcopy(GOLD[i])
        near[j]["arguments"][k] = " ".join(near[j]["arguments"][k].split()[:-1])  # bỏ 1 từ: vẫn khớp
        self.assertEqual(S.score_sample(i, near)["score"], 1.0)
        far = copy.deepcopy(GOLD[i])
        far[j]["arguments"][k] = "hoàn toàn khác"
        r = S.score_sample(i, far)
        self.assertTrue(r["hash_ok"])  # trường fuzzy không nằm trong hash
        self.assertLess(r["score"], 1.0)

    def test_ask_params_as_set(self):
        i = first(lambda c: any(x["name"] == "sys.ask_missing_info" for x in c))
        calls = copy.deepcopy(GOLD[i])
        for c in calls:
            if c["name"] == "sys.ask_missing_info":
                c["arguments"]["params"] = c["arguments"]["params"] + ["khac"]
        self.assertLess(S.score_sample(i, calls)["score"], 1.0)

    def test_int_vs_float_and_bool(self):
        self.assertIsNone(S.arg_mismatch("smart_home.climate_set", {"temperature": 26}, {"temperature": 26.0}))
        self.assertIsNotNone(S.arg_mismatch("door_lock.set", {"locked": 1}, {"locked": True}))
        self.assertIsNotNone(S.arg_mismatch("x.y", {"n": "26"}, {"n": 26}))

    def test_random_shuffle_of_reads(self):
        rnd = random.Random(0)
        for i in rnd.sample([i for i, c in GOLD.items() if c and not has_write(c)], 20):
            calls = GOLD[i][:]
            rnd.shuffle(calls)
            self.assertEqual(S.score_sample(i, calls)["score"], 1.0, i)


if __name__ == "__main__":
    unittest.main()
