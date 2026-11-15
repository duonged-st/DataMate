"""Export readable DataMate reports from existing submission and question logs."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from datamate.answer import validate_btc_submission
from datamate.data import Dataset
from datamate.abstention import CONFIDENCE_SOURCE, log_confidence
from datamate.score import approx_btc_score, valid_value
from verify_outputs import load_jsonl, verify


PREDICTION_COLUMNS = (
    "id", "business_id", "as_of_date", "question", "predicted_type", "predicted_answer",
    "final_type", "final_answer", "final_reason", "sql", "sql_result", "sql_attempts",
    "confidence", "confidence_source", "abstain_reason", "elapsed_ms", "attempt_count", "error_details",
)
EVALUATION_COLUMNS = (
    *PREDICTION_COLUMNS, "gold_type", "gold_answer", "gold_aliases", "is_correct",
    "status", "score", "pa_contribution", "error_category", "error_detail",
)


def json_cell(value):
    return "" if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def csv_text(value):
    # Excel can evaluate a cell as a formula, even when the CSV field is quoted.
    if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def save_csv(path, columns, rows):
    pending = path.with_suffix(".csv.tmp")
    with pending.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def logged_errors(log):
    errors = [log[k] for k in ("loader_error", "runtime_error") if log.get(k)]
    for attempt in log.get("attempts", []):
        errors.extend(attempt[k] for k in ("model_error", "error") if attempt.get(k))
    return errors


def error_category(pred, gold, points, log):
    if points == 1:
        return "", ""
    reason = log.get("abstain_reason")
    if points == 0:
        if isinstance(reason, dict):
            return reason["category"], reason["detail"]
        if log.get("loader_error") or log.get("runtime_error"):
            return "execution", log.get("loader_error") or log["runtime_error"]
        attempts = log.get("attempts", [])
        last = attempts[-1] if attempts else {}
        if last.get("model_error"):
            return "execution", last["model_error"]
        if last.get("error"):
            if "output" in last:
                return "formatting", last["error"]
            if "code" in last:
                return "execution", last["error"]
            return "sql_generation", last["error"]
        return "abstain", "No diagnostic reason recorded"
    actual, expected = pred["type"], gold["type"]
    if actual == "unanswerable" or expected == "unanswerable":
        return "business_rule", "false_unanswerable" if actual == "unanswerable" else "missed_unanswerable"
    if actual != expected:
        return "formatting", f"type_mismatch: {actual} vs {expected}"
    if not valid_value(pred.get("answer"), actual):
        return "formatting", "invalid_answer_format"
    return "undetermined", "value_mismatch; inspect SQL and business rule"


def score_summary(counts):
    total = sum(counts.values())
    points = counts[1] - counts[-1]
    return {"questions": total, "correct": counts[1], "wrong": counts[-1], "abstain": counts[0],
            "score": points, "accuracy": counts[1] / total, "local_pa": points / total,
            "coverage": (counts[1] + counts[-1]) / total}


def run_summary(counts, logs):
    summary = score_summary(counts)
    latencies = [log["elapsed_ms"] for log in logs.values() if "elapsed_ms" in log]
    return {**summary, "total": summary["questions"], "pa": summary["local_pa"],
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "confidence_method": CONFIDENCE_SOURCE if any(log_confidence(log) is not None
                                                          for log in logs.values()) else None,
            "optimal_threshold": None}


def export_reports(dataset, folder, allow_missing_logs=False):
    questions = {row["id"]: row for row in dataset.jsonl("questions.jsonl")}
    manifest = folder / "questions.jsonl"
    submitted = load_jsonl(folder / "submission.jsonl")
    ids = [r["id"] for r in load_jsonl(manifest)] if manifest.is_file() else [r["id"] for r in submitted]
    if not ids or len(ids) != len(set(ids)) or set(ids) - questions.keys():
        raise ValueError("Empty list, duplicate IDs, or question IDs absent from the dataset")
    log_path = folder / "logs/questions.jsonl"
    if allow_missing_logs and not log_path.is_file():
        validate_btc_submission(submitted, set(ids))
        predictions = {row["id"]: row for row in submitted}
        logs = {}
    else:
        predictions = verify(folder, set(ids))
        logs = {row["id"]: row for row in load_jsonl(log_path)}
    gold = {}
    if "answers.jsonl" in dataset.members:
        gold = {row["id"]: row for row in dataset.jsonl("answers.jsonl")}
        if set(ids) - gold.keys():
            raise ValueError("Dataset has a gold file but some predicted IDs are missing from it")

    prediction_rows = []
    evaluation_rows = []
    counts = Counter()
    categories = Counter()
    by_business = defaultdict(Counter)
    for identifier in ids:
        question, pred, log = questions[identifier], predictions[identifier], logs.get(identifier, {})
        attempts = log.get("attempts", [])
        candidate = log.get("candidate_answer", log.get("candidate"))
        if candidate is None and pred["type"] != "abstain":
            candidate = pred
        candidate = candidate or {}
        abstain_reason = log.get("abstain_reason") if pred["type"] == "abstain" else None
        if pred["type"] == "abstain" and not abstain_reason:
            error_kind, detail = error_category(pred, {}, 0, log)
            abstain_reason = {"category": error_kind, "detail": detail}
        row = {
            "id": identifier, "business_id": question["business_id"],
            "as_of_date": question["as_of_date"], "question": csv_text(question["question"]),
            "predicted_type": candidate.get("type"),
            "predicted_answer": json_cell(candidate.get("answer")),
            "final_type": pred["type"], "final_answer": json_cell(pred.get("answer")),
            "final_reason": csv_text(pred.get("reason", "")),
            "sql": csv_text(log.get("code") or ""),
            "sql_result": json_cell(log.get("output")),
            "sql_attempts": json_cell([a for a in attempts if "code" in a]),
            "confidence": log_confidence(log),
            "confidence_source": CONFIDENCE_SOURCE if log_confidence(log) is not None else None,
            "abstain_reason": json_cell(abstain_reason), "elapsed_ms": log.get("elapsed_ms"),
            "attempt_count": len(attempts) if log else None,
            "error_details": csv_text(" | ".join(logged_errors(log))),
        }
        prediction_rows.append(row)
        if gold:
            expected = gold[identifier]
            points = approx_btc_score(pred, expected)
            counts[points] += 1
            by_business[question["business_id"]][points] += 1
            category, detail = error_category(pred, expected, points, log)
            if category:
                categories[category] += 1
            evaluation_rows.append({
                **row, "gold_type": expected["type"], "gold_answer": json_cell(expected.get("answer")),
                "gold_aliases": json_cell(expected.get("alias")),
                "status": {1: "correct", -1: "wrong", 0: "abstain"}[points],
                "is_correct": points == 1,
                "score": points, "pa_contribution": points / len(ids),
                "error_category": category, "error_detail": csv_text(detail),
            })

    save_csv(folder / "predictions.csv", PREDICTION_COLUMNS, prediction_rows)
    if not gold:
        for name in ("evaluation.csv", "by_business.csv", "confidence_sweep.csv"):
            (folder / name).unlink(missing_ok=True)
        return {}
    save_csv(folder / "evaluation.csv", EVALUATION_COLUMNS, evaluation_rows)
    summaries = [{"business_id": biz, **score_summary(bucket)}
                 for biz, bucket in sorted(by_business.items())]
    save_csv(folder / "by_business.csv", tuple(summaries[0]), summaries)
    return {**run_summary(counts, logs), "error_categories": dict(sorted(categories.items())),
            "by_business": summaries, "confidence_scored": sum(log_confidence(log) is not None
                                                              for log in logs.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Original dataset ZIP or extracted directory")
    parser.add_argument("--output", required=True, type=Path, help="Existing run directory")
    parser.add_argument("--allow-missing-logs", action="store_true", help="Score existing submissions without fabricating logs")
    args = parser.parse_args()
    with Dataset(args.data) as dataset:
        metrics = export_reports(dataset, args.output, args.allow_missing_logs)
    if metrics:
        serialized = json.dumps(metrics, ensure_ascii=False, indent=2)
        (args.output / "metrics.json").write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
    else:
        predictions = load_jsonl(args.output / "submission.jsonl")
        log_path = args.output / "logs/questions.jsonl"
        logs = load_jsonl(log_path) if log_path.is_file() else []
        total = len(predictions)
        abstain = sum(row["type"] == "abstain" for row in predictions)
        summary = {"total": total, "abstain": abstain, "coverage": (total - abstain) / total,
                   "avg_latency_ms": sum(log["elapsed_ms"] for log in logs) / len(logs) if logs else None,
                   "confidence_method": CONFIDENCE_SOURCE if any(log_confidence(log) is not None
                                                                  for log in logs) else None,
                   "optimal_threshold": None}
        (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
