import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from report import error_category, export_reports


class Reports(unittest.TestCase):
    def test_exports_joined_rows_and_local_pa(self):
        questions = [{"id": f"q{i}", "business_id": "biz_001", "question": "=Bao nhiêu?",
                      "as_of_date": "2026-01-01"} for i in range(5)]
        gold = [{"id": q["id"], "type": "number", "answer": 7} for q in questions]
        answers = [
            {"id": "q0", "type": "number", "answer": 7},
            {"id": "q1", "type": "number", "answer": 8},
            {"id": "q2", "type": "string", "answer": "bảy"},
            {"id": "q3", "type": "abstain"},
            {"id": "q4", "type": "number", "answer": 7},
        ]
        logs = [
            {"id": row["id"], "answer": {k: v for k, v in row.items() if k != "id"}, "elapsed_ms": 17,
             "thinking": None, "code": "SELECT 7" if row["type"] != "abstain" else None,
             "output": [[7]] if row["type"] != "abstain" else None}
            for row in answers
        ]
        logs[3]["attempts"] = [{"code": "SELECT bad.col", "error": "no such column"}]
        data = {"questions.jsonl": questions, "answers.jsonl": gold}
        dataset = SimpleNamespace(members=data, jsonl=data.__getitem__)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "logs").mkdir()
            for name, rows in (("submission.jsonl", answers), ("logs/questions.jsonl", logs),
                               ("questions.jsonl", list(reversed(questions)))):
                (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = export_reports(dataset, root)
            self.assertEqual(result["local_pa"], 0)
            self.assertEqual((result["correct"], result["wrong"], result["abstain"]), (2, 2, 1))
            with (root / "evaluation.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r["id"] for r in rows], ["q4", "q3", "q2", "q1", "q0"])
            self.assertEqual([r["error_category"] for r in rows],
                             ["", "execution", "formatting", "undetermined", ""])
            self.assertEqual(sum(float(r["pa_contribution"]) for r in rows), 0)
            with (root / "predictions.csv").open(encoding="utf-8-sig", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["sql"], "SELECT 7")
            self.assertEqual(row["question"], "'=Bao nhiêu?")
            self.assertNotIn("gold_answer", row)
            del data["answers.jsonl"]
            self.assertEqual(export_reports(dataset, root), {})
            self.assertFalse((root / "evaluation.csv").exists())

    def test_answer_and_transport_errors_are_separate(self):
        pred = {"type": "abstain"}
        gold = {"type": "number"}
        self.assertEqual(error_category(pred, gold, 0, {"attempts": [{"model_error": "timeout"}]}),
                         ("execution", "timeout"))
        self.assertEqual(error_category(pred, gold, 0, {"attempts": [{"code": "SELECT 1",
                         "output": [[1]], "error": "schema mismatch"}]}), ("formatting", "schema mismatch"))
