"""CLI shared by Kaggle and the submission container."""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from time import monotonic

from .abstention import filter_answer, read_policy
from .data import Dataset, build_business
from .inference import LlamaClient, infer
from .prompt import SplitContext
from .server import LocalServer


def write_line(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    handle.flush()


def select_questions(rows, businesses, limit, per_business, seed):
    if businesses:
        names = set(businesses)
        missing = names - {r["business_id"] for r in rows}
        if missing:
            raise ValueError(f"Businesses absent from this split: {sorted(missing)}")
        rows = [r for r in rows if r["business_id"] in names]
    if per_business:
        rng = random.Random(seed)
        groups = defaultdict(list)
        for q in rows:
            groups[q["business_id"]].append(q)
        chosen = {r["id"] for values in groups.values() for r in rng.sample(values, min(len(values), per_business))}
        rows = [r for r in rows if r["id"] in chosen]
    return rows[:limit] if limit else rows


def audit(dataset, questions, args):
    summary = {}
    ids = sorted({q["business_id"] for q in questions})
    for biz in ids:
        started = monotonic()
        try:
            _, catalog, _ = build_business(dataset, biz, args.work)
            summary[biz] = catalog
            rows = sum(t["rows"] for t in catalog["tables"])
            print(f"{biz}: {len(catalog['tables'])} tables, {rows:,} rows, {monotonic()-started:.1f}s", flush=True)
        except (ValueError, OSError) as exc:
            summary[biz] = {"error": str(exc)}
            print(f"{biz}: {exc}", file=sys.stderr, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "loader_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(any("error" in result or any("Header not detected" == w.get("error") for w in result.get("warnings", []))
                   for result in summary.values()))


def predict(dataset, questions, args):
    deadline = monotonic() + args.run_time_limit if args.run_time_limit else float("inf")
    policy = read_policy(args.threshold_config) if args.threshold_config else None
    threshold = policy["threshold"] if policy else None
    if threshold is not None:
        print(f"Using confidence threshold {threshold!r} from {args.threshold_config}", flush=True)
    context = SplitContext(dataset)
    root = args.output
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    server = None
    if args.model_file:
        command = ["bash", str(Path(__file__).resolve().parents[1] / "serve.sh"),
                   str(args.model_file.resolve())]
        env = None
        if args.server_bin:
            binary = args.server_bin.resolve()
            libs = [str(binary.parent), os.environ.get("LD_LIBRARY_PATH", "")]
            env = {"LLAMA_SERVER": str(binary), "LD_LIBRARY_PATH": ":".join(filter(None, libs))}
        server = LocalServer(command, args.endpoint, log_dir / "llama-server.log", env=env)
    client = LlamaClient(args.endpoint, server)
    groups = {}
    counts = Counter()
    with server.running() if server else nullcontext(), \
         (root / "submission.jsonl").open("w", encoding="utf-8") as submission, \
         (root / "logs" / "questions.jsonl").open("w", encoding="utf-8") as logs:
        for index, q in enumerate(questions, 1):
            start = monotonic()
            biz = q["business_id"]
            if start < deadline and biz not in groups:
                try:
                    groups[biz] = build_business(dataset, biz, args.work)
                except (OSError, ValueError) as exc:
                    groups[biz] = str(exc)
                    print(f"Loader {biz}: {exc}", file=sys.stderr)
            data = groups.get(biz)
            remaining = deadline - monotonic()
            if remaining <= 0 or isinstance(data, str):
                row = {"id": q["id"], "type": "abstain"}
                log = {"id": q["id"], "answer": {"type": "abstain"}, "thinking": None,
                       "code": None, "output": None}
                if remaining <= 0:
                    log["runtime_error"] = "run budget expired"
                else:
                    log["loader_error"] = data
            else:
                row, log = infer(q, *data, context, client, question_timeout=min(120, remaining),
                                 log_confidence=args.log_confidence or threshold is not None)
            if "candidate_answer" not in log:
                log.update(candidate_answer=None, confidence=None, confidence_source=None,
                           abstain_reason={"category": "execution", "detail": log.get("loader_error") or log["runtime_error"]})
            row, log = filter_answer(row, log, threshold)
            log["elapsed_ms"] = int((monotonic() - start) * 1000)
            write_line(submission, row)
            write_line(logs, log)
            counts[row["type"]] += 1
            if index == 1 or index == len(questions) or index % 20 == 0:
                print(f"{index}/{len(questions)} | {dict(counts)}", flush=True)
    return 0


def score(dataset, path):
    from .score import approx_btc_score

    gold = {r["id"]: r for r in dataset.jsonl("answers.jsonl")}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({r["id"] for r in rows}) != len(rows) or any(r["id"] not in gold for r in rows):
        raise ValueError("Submission has duplicate IDs or IDs not present in Training")
    counts = Counter(approx_btc_score(row, gold[row["id"]]) for row in rows)
    print(json.dumps({"scored": len(rows), "correct": counts[1], "wrong": counts[-1],
                      "abstain": counts[0], "score": counts[1] - counts[-1]}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "predict", "score"])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("result"))
    parser.add_argument("--work", type=Path, default=Path(".datamate_cache"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--model-file", type=Path, help="Start and manage the bundled local server")
    parser.add_argument("--server-bin", type=Path, help="Explicit executable for managed llama-server")
    parser.add_argument("--run-time-limit", type=int, help="Seconds reserved for answering before writing remaining abstentions")
    parser.add_argument("--threshold-config", type=Path, help="Apply a frozen confidence policy to predictions")
    parser.add_argument("--log-confidence", action="store_true", help="Request optional llama.cpp token log probabilities")
    parser.add_argument("--business", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-business", type=int)
    parser.add_argument("--seed", default=20261112, type=int)
    parser.add_argument("--questions-file", type=Path,
                        help="Optional dev manifest in CSV or JSONL with an id column")
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args(argv)
    if any(x is not None and x <= 0 for x in (args.limit, args.per_business, args.run_time_limit)):
        parser.error("--limit, --per-business and --run-time-limit must be positive")
    if args.model_file and (not args.model_file.is_file() or args.endpoint != "http://127.0.0.1:8080"):
        parser.error("--model-file requires an existing GGUF and the default local endpoint")
    if args.server_bin and (not args.model_file or not args.server_bin.is_file()):
        parser.error("--server-bin requires --model-file and an existing executable")
    with Dataset(args.data) as dataset:
        if args.mode == "score":
            return score(dataset, args.submission or args.output / "submission.jsonl")
        questions = dataset.jsonl("questions.jsonl")
        if args.questions_file:
            import csv
            with args.questions_file.open(encoding="utf-8-sig") as f:
                chosen = ([json.loads(line) for line in f if line.strip()]
                          if args.questions_file.suffix.lower() == ".jsonl" else csv.DictReader(f))
                ids = [r["id"] for r in chosen]
            by_id = {q["id"]: q for q in questions}
            if not ids or len(set(ids)) != len(ids) or set(ids) - by_id.keys():
                parser.error("Manifest is empty or contains duplicate or unknown question IDs")
            questions = [by_id[identifier] for identifier in ids]
        questions = select_questions(questions, args.business, args.limit, args.per_business, args.seed)
        return audit(dataset, questions, args) if args.mode == "audit" else predict(dataset, questions, args)


if __name__ == "__main__":
    raise SystemExit(main())
