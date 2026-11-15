"""Stage the original Training archive and the pinned GGUF before Docker build."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from datamate.abstention import read_policy
from datamate.data import Dataset
from datamate.score import approx_btc_score
from verify_outputs import load_jsonl, verify

ROOT = Path(__file__).resolve().parent
REV = "3885219b6810b007914f3a7950a8d1b469d598a5"
MODEL = "Qwen3.5-9B-Q4_K_M.gguf"
SHA256 = "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"
URL = f"https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/{REV}/{MODEL}"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_assets(existing=None):
    target = ROOT / "models" / MODEL
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        pending = target.with_suffix(".part")
        if existing:
            shutil.copy2(existing, pending)
        else:
            print(f"Downloading {MODEL} (5.68 GB)", flush=True)
            with urllib.request.urlopen(URL, timeout=120) as response, pending.open("wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
        if digest(pending) != SHA256:
            pending.unlink()
            raise ValueError("GGUF SHA-256 mismatch")
        pending.replace(target)
    elif digest(target) != SHA256:
        raise ValueError(f"GGUF SHA-256 mismatch: {target}")
    return target


def benchmark_evidence(source, questions, answers):
    if source is None:
        return {}
    gold = {row["id"]: row for row in answers}
    by_id = {row["id"]: row for row in questions}
    dev = [row["id"] for row in load_jsonl(ROOT / "training_code/dev_80.jsonl")]
    files = ("questions.jsonl", "submission.jsonl", "metrics.json", "run_config.json",
             "run.log", "predictions.csv", "evaluation.csv", "logs/questions.jsonl", "logs/llama-server.log")
    evidence = {}
    for phase in ("smoke", "regression", "holdout"):
        run = source / phase
        ids = [row["id"] for row in load_jsonl(run / "questions.jsonl")]
        expected = {"smoke": dev[:10], "regression": dev}.get(phase)
        if expected is not None and ids != expected:
            raise ValueError(f"{phase}: question IDs differ from the expected manifest")
        if not ids or len(set(ids)) != len(ids) or set(ids) - gold.keys():
            raise ValueError(f"{phase}: invalid question IDs")
        if phase == "holdout" and {by_id[q]["business_id"] for q in ids} & {by_id[q]["business_id"] for q in dev}:
            raise ValueError("Holdout includes a regression business")
        result = verify(run, set(ids))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        scores = [approx_btc_score(result[q], gold[q]) for q in ids]
        counts = {"correct": scores.count(1), "wrong": scores.count(-1), "abstain": scores.count(0),
                  "score": sum(scores), "accuracy": scores.count(1) / len(scores),
                  "local_pa": sum(scores) / len(scores)}
        logs = load_jsonl(run / "logs/questions.jsonl")
        responses = sum(any("response" in attempt for attempt in row.get("attempts", [])) for row in logs)
        rules = sum(any(a.get("source") == "rulebook_sql" and "output" in a and "error" not in a
                        for a in row.get("attempts", [])) for row in logs)
        gaps = sum(any(a.get("source") == "catalog_evidence" for a in row.get("attempts", [])) for row in logs)
        if (not (responses or rules or gaps) or metrics.get("qwen_questions") != responses
                or metrics.get("rulebook_sql_questions", 0) != rules
                or metrics.get("catalog_evidence_questions", 0) != gaps or config.get("model_sha256") != SHA256):
            raise ValueError(f"{phase}: run evidence does not match logs or pinned model")
        if any(metrics.get(k) != v for k, v in counts.items()):
            raise ValueError(f"{phase}: metrics do not match the recorded predictions")
        sources = sorted((ROOT / "datamate").glob("*.py")) + [ROOT / "serve.sh", ROOT / "report.py", ROOT / "benchmark.py"]
        expected_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
        if config.get("source_sha256") != expected_hashes:
            raise ValueError(f"{phase}: log source hashes differ from this V5 code")
        for name in files:
            (run / name).stat()
        extras = tuple(name for name in ("by_business.csv", "confidence_sweep.csv") if (run / name).is_file())
        evidence[phase] = {"root": run, "files": files + extras, "metrics": metrics}
    return evidence


def stage_training(source, benchmarks=None):
    with Dataset(source) as split:
        questions = split.jsonl("questions.jsonl")
        answers = split.jsonl("answers.jsonl")
    if len(questions) != 3000 or {q["id"] for q in questions} != {a["id"] for a in answers}:
        raise ValueError("Expected the published Training ZIP with 3,000 matching questions and answers")
    folder = ROOT / "training_code"
    prior_manifest = folder / "data_manifest.json"
    if benchmarks is None and prior_manifest.is_file():
        prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
        if prior.get("benchmarks"):
            benchmarks = ROOT / "verification" / "benchmarks"
    evidence = benchmark_evidence(benchmarks, questions, answers)
    target = folder / "data" / "train.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target:
        shutil.copy2(source, target)
    artifact = {"path": "data/train.zip", "source": "DataMate BTC Training ZIP",
                "bytes": target.stat().st_size, "sha256": digest(target), "questions": len(questions),
                "answers": len(answers), "businesses": len({q["business_id"] for q in questions})}
    dev = folder / "dev_80.jsonl"
    manifest = {"train": artifact, "fine_tuning": False, "synthetic_data": False,
                "validation": {"source": "Same 80 Training question IDs as V4.2",
                               "manifest": dev.name, "questions": len(dev.read_text().splitlines()),
                               "sha256": digest(dev), "measured_accuracy": None}}
    if evidence:
        manifest["benchmarks"] = {}
        for phase, run in evidence.items():
            artifacts = []
            for name in run["files"]:
                output = ROOT / "verification" / "benchmarks" / phase / name
                output.parent.mkdir(parents=True, exist_ok=True)
                input_file = run["root"] / name
                if input_file.resolve() != output:
                    shutil.copy2(input_file, output)
                artifacts.append({"path": output.relative_to(ROOT).as_posix(), "bytes": output.stat().st_size,
                                  "sha256": digest(output)})
            manifest["benchmarks"][phase] = {"metrics": run["metrics"], "artifacts": artifacts}
        manifest["validation"]["measured_accuracy"] = evidence["regression"]["metrics"]["accuracy"]
    (folder / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target



def stage_policy(source):
    policy = read_policy(source)
    if policy["threshold"] is not None and policy.get("model_sha256") != SHA256:
        raise ValueError("The confidence policy was not evaluated with the pinned GGUF")
    target = ROOT / "confidence_policy.json"
    pending = target.with_suffix(".tmp")
    pending.write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--model", type=Path, help="Reuse an existing GGUF with the pinned SHA-256")
    parser.add_argument("--benchmarks", type=Path, help="Directory with Kaggle smoke, regression and holdout results")
    parser.add_argument("--confidence-policy", type=Path, help="Freeze the evaluated Docker inference policy")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if args.model_only and args.train_only:
        parser.error("Choose at most one of --model-only and --train-only")
    if not args.model_only:
        if not args.train:
            parser.error("--train is required unless using --model-only")
        print(stage_training(args.train, args.benchmarks))
    if not args.train_only:
        print(model_assets(args.model))
    if args.confidence_policy:
        print(stage_policy(args.confidence_policy))


if __name__ == "__main__":
    main()
