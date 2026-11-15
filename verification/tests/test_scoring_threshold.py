import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from datamate.abstention import CONFIDENCE_SOURCE, read_policy
from datamate.inference import token_confidence
from datamate.score import approx_btc_score
from threshold import RUN_CONFIG_KEYS, ranking_quality, sweep, write_jsonl


class BtcScoring(unittest.TestCase):
    def score(self, kind, candidate, gold, aliases=()):
        return approx_btc_score({"type": kind, "answer": candidate},
                                {"type": kind, "answer": gold, "alias": aliases})

    def test_numbers_dates_and_unanswerable(self):
        self.assertEqual(self.score("percent", 12.4, 12.40), 1)
        self.assertEqual(self.score("percent", "12.4%", 12.40), -1)
        self.assertEqual(self.score("percent", 0.124, 12.40), -1)
        self.assertEqual(self.score("number", 10000000000001, 10000000000000), -1)
        self.assertEqual(self.score("boolean", 1, True), -1)
        self.assertEqual(self.score("date", "2026-09", "2026-09"), 1)
        gold = {"type": "unanswerable", "answer": None}
        self.assertEqual(approx_btc_score({"type": "unanswerable", "reason": "Thiếu dữ liệu"}, gold), 1)
        self.assertEqual(approx_btc_score({"type": "abstain"}, gold), 0)

    def test_ordered_list_and_table_aliases(self):
        self.assertEqual(self.score("list", ["KH1", "Khánh"], ["Đức", "Khánh"], [["KH1"], []]), 1)
        self.assertEqual(self.score("list", ["Khánh", "KH1"], ["Đức", "Khánh"], [["KH1"], []]), -1)
        gold = {"columns": ["khach_hang", "so_tien_no"], "rows": [["Đức", 7], ["Khánh", 8]]}
        pred = {"columns": ["so_tien_no", "khach_hang"], "rows": [[8, "KH2"], [7, "ĐỨC"]]}
        self.assertEqual(self.score("table", pred, gold, [["KH1"], ["KH2"]]), 1)
        pred["rows"][0][0] = "8"
        self.assertEqual(self.score("table", pred, gold, [["KH1"], ["KH2"]]), -1)

    def test_token_logprobs(self):
        self.assertAlmostEqual(token_confidence({"content": [{"logprob": 0}, {"logprob": -2}]})["confidence"],
                               0.36787944117)
        self.assertIsNone(token_confidence({"content": [{"prob": 0.7}]}))
        self.assertIsNone(token_confidence({"content": [{"logprob": float("nan")}]}))

    def test_disjoint_threshold_selection_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = {"questions.jsonl": [], "answers.jsonl": []}
            for phase, biz, values in [("cal", "biz_a", [(0.2, 2), (0.9, 1)]),
                                       ("val", "biz_b", [(0.1, 1), (0.9, 2)])]:
                directory = root / phase
                (directory / "logs").mkdir(parents=True)
                config = dict.fromkeys(RUN_CONFIG_KEYS, "fixture")
                config["confidence_source"] = CONFIDENCE_SOURCE
                (directory / "run_config.json").write_text(json.dumps(config))
                questions, rows, logs = [], [], []
                for i, (confidence, answer) in enumerate(values):
                    qid = f"{phase}_{i}"
                    questions.append({"id": qid, "business_id": biz, "question": "Bao nhiêu?", "as_of_date": "2026-11-01"})
                    data["answers.jsonl"].append({"id": qid, "type": "number", "answer": 1})
                    candidate = {"type": "number", "answer": answer}
                    rows.append({"id": qid, **candidate})
                    logs.append({"id": qid, "answer": candidate, "candidate_answer": candidate,
                                 "elapsed_ms": 20, "thinking": None, "code": "SELECT 1", "output": [[answer]],
                                 "confidence": confidence, "confidence_source": CONFIDENCE_SOURCE})
                data["questions.jsonl"].extend(questions)
                write_jsonl(directory / "questions.jsonl", questions)
                write_jsonl(directory / "submission.jsonl", rows)
                write_jsonl(directory / "logs/questions.jsonl", logs)
            dataset = SimpleNamespace(members=data, jsonl=data.__getitem__)
            output = root / "out"
            metrics = sweep(dataset, root / "cal", root / "val", output)
            self.assertEqual(metrics["calibration_pa"], 0.5)
            self.assertEqual(metrics["calibration_ranking"]["auc"], 1)
            self.assertEqual(metrics["validation_ranking"]["auc"], 0)
            self.assertIsNone(read_policy(output / "confidence_policy.json")["threshold"])
            self.assertEqual(metrics["pa"], -0.5)
            self.assertEqual(metrics["coverage"], 0.5)
            self.assertEqual(metrics["avg_latency_ms"], 20)
            answer = json.loads((output / "submission.jsonl").read_text().splitlines()[0])
            self.assertEqual(answer, {"id": "val_0", "type": "abstain"})
            with (output / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["predicted_answer"], "1")
            self.assertEqual(row["final_type"], "abstain")
            self.assertIn("confidence_below_threshold", row["abstain_reason"])
            for gold in data["answers.jsonl"]:
                if gold["id"].startswith("val"):
                    gold["answer"] = 2
            policy_file = root / "build" / "confidence_policy.json"
            passed = sweep(dataset, root / "cal", root / "val", root / "passed", policy_file)
            self.assertEqual(passed["validation_pa_gain"], 0.5)
            self.assertAlmostEqual(read_policy(policy_file)["threshold"], passed["optimal_threshold"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                sweep(dataset, root / "cal", root / "cal", root / "bad")

    def test_auc_missing_class_and_score_ties(self):
        predictions = {"q1": {"type": "number", "answer": 1}, "q2": {"type": "number", "answer": 2}}
        logs = {qid: {"confidence_source": CONFIDENCE_SOURCE, "confidence": 0.8} for qid in predictions}
        gold = {qid: {"type": "number", "answer": 1} for qid in predictions}
        self.assertEqual(ranking_quality(predictions, logs, gold)["auc"], 0.5)
        logs["q2"]["confidence"] = None
        self.assertIsNone(ranking_quality(predictions, logs, gold)["auc"])
