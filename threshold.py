"""Select a threshold on calibration businesses and evaluate it on disjoint validation businesses."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from datamate.answer import validate_btc_submission
from datamate.data import Dataset
from datamate.abstention import CONFIDENCE_SOURCE, MISSING_CONFIDENCE_POLICY, filter_answer, log_confidence
from datamate.score import approx_btc_score
from report import export_reports, score_summary, save_csv
from verify_outputs import load_jsonl, verify

RUN_CONFIG_KEYS = ("source_sha256", "server_sha256", "model_sha256", "seed", "temperature",
                   "max_tokens", "context_length", "parallel_slots", "confidence_source")


def load_run(folder, questions):
    manifest = load_jsonl(folder / "questions.jsonl")
    ids = [row["id"] for row in manifest]
    if (not ids or len(ids) != len(set(ids)) or
            any(questions.get(row["id"]) != row["business_id"] for row in manifest)):
        raise ValueError(f"Invalid manifest: {folder}")
    predictions = verify(folder, set(ids))
    logs = {row["id"]: row for row in load_jsonl(folder / "logs/questions.jsonl")}
    for qid, prediction in predictions.items():
        if prediction["type"] == "abstain":
            reason = logs[qid].get("abstain_reason")
            if isinstance(reason, dict) and reason.get("detail") == "confidence_below_threshold":
                raise ValueError(f"Expected an unfiltered run: {folder}")
            continue
        expected = {k: v for k, v in prediction.items() if k != "id"}
        if logs[qid].get("candidate_answer") != expected:
            raise ValueError(f"Expected an unfiltered run with recorded candidates: {folder}")
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    if config.get("confidence_source") != CONFIDENCE_SOURCE or any(k not in config for k in RUN_CONFIG_KEYS):
        raise ValueError(f"Expected a benchmark with token logprob enabled: {folder}")
    if not any(log_confidence(log) is not None for log in logs.values()):
        raise ValueError(f"No valid token logprobs found in {folder}; use a compatible llama-server")
    return manifest, predictions, logs, {key: config[key] for key in RUN_CONFIG_KEYS}


def ranking_quality(predictions, logs, gold):
    positives, negatives = [], []
    for qid, prediction in predictions.items():
        confidence = log_confidence(logs[qid])
        points = approx_btc_score(prediction, gold[qid])
        if confidence is not None and points:
            (positives if points == 1 else negatives).append(confidence)
    pairs = len(positives) * len(negatives)
    auc = (sum(1 if p > n else 0.5 if p == n else 0 for p in positives for n in negatives) / pairs
           if pairs else None)
    bins = []
    for i in range(5):
        lower, upper = i / 5, (i + 1) / 5
        correct = sum(lower <= s and (s < upper or (i == 4 and s == 1)) for s in positives)
        wrong = sum(lower <= s and (s < upper or (i == 4 and s == 1)) for s in negatives)
        count = correct + wrong
        bins.append({"lower": lower, "upper": upper, "count": count, "correct": correct, "wrong": wrong,
                     "observed_accuracy": correct / count if count else None})
    return {"auc": auc, "scored_correct": len(positives), "scored_wrong": len(negatives),
            "confidence_bins": bins}


def count_scores(predictions, logs, threshold, gold):
    return score_summary(Counter(approx_btc_score(filter_answer(row, logs[qid], threshold)[0], gold[qid])
                                 for qid, row in predictions.items()))


def write_jsonl(path, rows):
    pending = path.with_suffix(".tmp")
    with pending.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


def sweep(dataset, calibration_dir, validation_dir, output, policy_output=None):
    questions = {row["id"]: row["business_id"] for row in dataset.jsonl("questions.jsonl")}
    gold = {row["id"]: row for row in dataset.jsonl("answers.jsonl")}
    c_manifest, calibration, c_logs, c_config = load_run(calibration_dir, questions)
    v_manifest, validation, v_logs, v_config = load_run(validation_dir, questions)
    c_biz = {row["business_id"] for row in c_manifest}
    v_biz = {row["business_id"] for row in v_manifest}
    if c_biz & v_biz:
        raise ValueError(f"Calibration and validation overlap by business: {sorted(c_biz & v_biz)}")
    if c_config != v_config:
        raise ValueError("Calibration and validation used different model, runtime or code settings")
    if output.resolve() in {calibration_dir.resolve(), validation_dir.resolve()}:
        raise ValueError("Use a new output directory for threshold results")

    observed = {score for log in c_logs.values() if (score := log_confidence(log)) is not None}
    # The last boundary evaluates withholding every scored calibration candidate.
    thresholds = [None, *sorted({*(i / 100 for i in range(0, 101, 5)), *observed,
                                math.nextafter(max(observed), math.inf)})]
    c_results = [count_scores(calibration, c_logs, t, gold) for t in thresholds]
    # Ties favor higher coverage, then the first/lower threshold including the unfiltered baseline.
    selected = max(range(len(thresholds)),
                   key=lambda i: (c_results[i]["local_pa"], c_results[i]["coverage"]))
    proposed = thresholds[selected]
    calibration_quality = ranking_quality(calibration, c_logs, gold)
    auc = calibration_quality["auc"]
    calibration_gain = c_results[selected]["local_pa"] - c_results[0]["local_pa"]
    recommended_index = selected
    if auc is None or auc <= 0.5 or calibration_gain <= 0:
        recommended_index = 0
    chosen = thresholds[recommended_index]
    results = []
    for index, threshold in enumerate(thresholds):
        v_metrics = count_scores(validation, v_logs, threshold, gold)
        results.append({"policy": "baseline" if threshold is None else "threshold", "threshold": threshold,
                        "selected_on_calibration": index == selected, "used_for_validation_submission": index == recommended_index,
                        **{f"calibration_{k}": c_results[index][k] for k in ("local_pa", "coverage", "correct", "wrong", "abstain")},
                        **{f"validation_{k}": v_metrics[k] for k in ("local_pa", "coverage", "correct", "wrong", "abstain")}})
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    rows = [filter_answer(row, v_logs[qid], chosen) for qid, row in validation.items()]
    validate_btc_submission([row for row, _ in rows], set(validation))
    write_jsonl(output / "questions.jsonl", v_manifest)
    write_jsonl(output / "submission.jsonl", (row for row, _ in rows))
    write_jsonl(output / "logs/questions.jsonl", (log for _, log in rows))
    metrics = export_reports(dataset, output)
    validation_gain = metrics["local_pa"] - results[0]["validation_local_pa"]
    frozen_threshold = chosen if chosen is not None and validation_gain > 0 else None
    policy = {"format_version": 1, "confidence_source": CONFIDENCE_SOURCE,
              "missing_confidence_policy": MISSING_CONFIDENCE_POLICY, "threshold": frozen_threshold,
              "proposed_threshold": proposed, "calibration_auc": auc,
              "calibration_pa_gain": calibration_gain, "validation_pa_gain": validation_gain,
              "calibration_businesses": sorted(c_biz), "validation_businesses": sorted(v_biz),
              "model_sha256": c_config["model_sha256"], "calibration_server_sha256": c_config["server_sha256"]}
    policy_text = json.dumps(policy, indent=2, allow_nan=False) + "\n"
    (output / "confidence_policy.json").write_text(policy_text, encoding="utf-8")
    if policy_output is not None:
        policy_output.parent.mkdir(parents=True, exist_ok=True)
        pending = policy_output.with_suffix(".tmp")
        pending.write_text(policy_text, encoding="utf-8")
        pending.replace(policy_output)
    metrics.update(optimal_threshold=chosen, proposed_threshold=proposed, deployment_threshold=frozen_threshold,
                   calibration_ranking=calibration_quality,
                   validation_ranking=ranking_quality(validation, v_logs, gold), calibration_pa_gain=calibration_gain,
                   validation_pa_gain=validation_gain, calibration_businesses=sorted(c_biz),
                   validation_businesses=sorted(v_biz), calibration_pa=c_results[recommended_index]["local_pa"],
                   validation_baseline_pa=results[0]["validation_local_pa"],
                   validation_baseline_coverage=results[0]["validation_coverage"],
                   calibration_confidence_scored=sum(log_confidence(log) is not None for log in c_logs.values()),
                   missing_confidence_policy=MISSING_CONFIDENCE_POLICY)
    save_csv(output / "confidence_sweep.csv", tuple(results[0]), results)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "calibration", "validation", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--policy-output", type=Path, help="Also freeze the resulting policy in the Docker build directory")
    args = parser.parse_args()
    with Dataset(args.data) as dataset:
        print(json.dumps(sweep(dataset, args.calibration, args.validation, args.output, args.policy_output), indent=2))


if __name__ == "__main__":
    main()
