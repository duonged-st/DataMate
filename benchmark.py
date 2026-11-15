"""Run and record reproducible Qwen benchmarks on the published Training split."""

import argparse
from collections import Counter
import json
import platform
import subprocess
import sys
import os
from pathlib import Path
from time import monotonic

from datamate.__main__ import select_questions
from datamate.data import Dataset
from prepare_assets import SHA256, digest
from report import export_reports
from datamate.abstention import CONFIDENCE_SOURCE
from verify_outputs import load_jsonl

ROOT = Path(__file__).resolve().parent
SEED = 20261112
RUNS = {"smoke": 10, "regression": 80}


def call(log, *args, timeout=30):
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout)
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        log.append(f"{args[0]}: {exc}")
        return None


def run_logged(output, *args):
    with output.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "datamate", *map(str, args)],
                                   cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        with process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        if process.wait():
            raise RuntimeError(f"DataMate exited {process.returncode}; see {output}")


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "datamate").glob("*.py")) + [ROOT / "serve.sh", ROOT / "report.py", Path(__file__).resolve()]
    with Dataset(args.data) as dataset:
        questions = dataset.jsonl("questions.jsonl")
    by_id = {q["id"]: q for q in questions}
    dev = [json.loads(s)["id"] for s in (ROOT / "training_code/dev_80.jsonl").read_text().splitlines()]
    if args.phase in RUNS:
        selected = [by_id[q] for q in dev[:RUNS[args.phase]]]
    else:
        if not args.business:
            raise ValueError("Choose businesses your team has not used to tune the pipeline.")
        if set(args.business) & {by_id[q]["business_id"] for q in dev}:
            raise ValueError("Holdout businesses must exclude the regression businesses.")
        selected = select_questions(questions, args.business, None, args.per_business, SEED)

    manifest = out / "questions.jsonl"
    manifest.write_text("".join(json.dumps({"id": q["id"], "business_id": q["business_id"]}) + "\n"
                                for q in selected), encoding="utf-8")
    model_sha = digest(args.model)
    if model_sha != SHA256:
        raise ValueError("GGUF differs from the pinned Qwen3.5-9B Q4_K_M model")

    environment_errors = []
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, (str(args.server.resolve().parent), env.get("LD_LIBRARY_PATH"))))
    try:
        version_result = subprocess.run([str(args.server.resolve()), "--version"], env=env, check=True,
                                        capture_output=True, text=True, timeout=20)
        version = (version_result.stdout + version_result.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot run selected llama-server: {exc}") from exc
    meta = {"phase": args.phase, "questions": len(selected), "businesses": sorted({q["business_id"] for q in selected}),
            "seed": SEED, "temperature": 0, "max_tokens": 900, "context_length": 16384, "parallel_slots": 1,
            "model_sha256": model_sha, "server_sha256": digest(args.server), "server_version": version,
            "python": platform.python_version(), "platform": platform.platform(),
            "gpu": call(environment_errors, "nvidia-smi", "-L"),
            "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
            "confidence_source": CONFIDENCE_SOURCE if args.log_confidence else None,
            "environment_errors": environment_errors}
    (out / "run_config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    started = monotonic()
    run_logged(out / "run.log", "predict", "--data", args.data.resolve(), "--output", out,
               "--work", args.work.resolve(), "--questions-file", manifest, "--model-file",
               args.model.resolve(), "--server-bin", args.server.resolve(),
               *(["--log-confidence"] if args.log_confidence else []))
    with Dataset(args.data) as dataset:
        scores = export_reports(dataset, out)
    if not scores:
        raise ValueError("Training benchmark requires answers.jsonl")
    logs = load_jsonl(out / "logs/questions.jsonl")
    qwen_questions = sum(any("response" in attempt for attempt in row.get("attempts", [])) for row in logs)
    errors = sum(any("model_error" in attempt for attempt in row.get("attempts", [])) for row in logs)
    rule_questions = sum(any(a.get("source") == "rulebook_sql" and "error" not in a
                             for a in row.get("attempts", [])) for row in logs)
    gap_questions = sum(any(a.get("source") == "catalog_evidence" for a in row.get("attempts", []))
                        for row in logs)
    confidence_diagnostics = Counter(attempt["confidence_evidence"].get("status", "valid")
                                     for row in logs for attempt in row.get("attempts", [])
                                     if "confidence_evidence" in attempt)
    metrics = {"phase": args.phase, **scores, "qwen_questions": qwen_questions,
               "rulebook_sql_questions": rule_questions, "catalog_evidence_questions": gap_questions,
               "confidence_diagnostics": dict(confidence_diagnostics),
               "questions_with_model_error": errors, "wall_seconds": round(monotonic() - started, 3)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    if not (qwen_questions or rule_questions or gap_questions):
        raise RuntimeError("No Qwen response or rule-based result; see logs/llama-server.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=[*RUNS, "holdout"])
    for name in ("data", "model", "server", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--work", type=Path, default=Path(".datamate_cache"))
    parser.add_argument("--log-confidence", action="store_true")
    parser.add_argument("--business", action="append")
    parser.add_argument("--per-business", type=int, default=20)
    args = parser.parse_args()
    if args.per_business <= 0:
        parser.error("--per-business must be positive")
    run(args)


if __name__ == "__main__":
    main()
