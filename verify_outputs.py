"""Check the BTC output contract and optionally compare two offline runs."""

import argparse
import json
from pathlib import Path

from datamate.answer import validate_btc_submission
from datamate.data import Dataset


def load_jsonl(path):
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify(folder, ids):
    submission = load_jsonl(folder / "submission.jsonl")
    validate_btc_submission(submission, ids)
    logs = load_jsonl(folder / "logs" / "questions.jsonl")
    by_id = {item["id"]: item for item in submission}
    logged_ids = {item["id"] for item in logs}
    if len(logs) != len(ids) or logged_ids != ids:
        raise ValueError("Logs must cover each expected question exactly once")
    for item in logs:
        answer = {k: v for k, v in by_id[item["id"]].items() if k != "id"}
        if item.get("answer") != answer:
            raise ValueError(f"Log and submission differ for {item['id']}")
        ms = item.get("elapsed_ms")
        if type(ms) is not int or ms < 0 or "thinking" not in item:
            raise ValueError(f"Invalid elapsed_ms or missing thinking for {item['id']}")
        if item.get("code") is not None and "output" not in item:
            raise ValueError(f"SQL log missing output for {item['id']}")
    return by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--questions-file", type=Path, help="Optional smoke JSONL manifest")
    args = parser.parse_args()
    with Dataset(args.data) as dataset:
        ids = {q["id"] for q in dataset.jsonl("questions.jsonl")}
    if args.questions_file:
        sample = {q["id"] for q in load_jsonl(args.questions_file)}
        if not sample or sample - ids:
            parser.error("The manifest is empty or contains IDs absent from the dataset")
        ids = sample
    first = verify(args.output, ids)
    print(f"Valid output and logs for {len(first)} questions.")
    if args.compare:
        second = verify(args.compare, ids)
        changed = [q for q in sorted(ids) if first[q] != second[q]]
        print(f"Different answers: {len(changed)}/{len(ids)}")
        if changed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
