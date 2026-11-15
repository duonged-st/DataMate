"""Use the same confidence policy in evaluation and submission inference."""

import json
import math

CONFIDENCE_SOURCE = "llama_cpp_token_logprob_geometric_mean"
MISSING_CONFIDENCE_POLICY = "keep_original"


def log_confidence(log):
    value = log.get("confidence")
    if log.get("confidence_source") == CONFIDENCE_SOURCE and type(value) in (int, float) and 0 <= value <= 1:
        return value
    return None


def read_policy(path):
    policy = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(policy, dict) or type(policy.get("format_version")) is not int or
            policy["format_version"] != 1 or policy.get("confidence_source") != CONFIDENCE_SOURCE or
            policy.get("missing_confidence_policy") != MISSING_CONFIDENCE_POLICY):
        raise ValueError("Unsupported abstention policy")
    if "threshold" not in policy:
        raise ValueError("Abstention policy is missing threshold")
    threshold = policy["threshold"]
    # nextafter(1, inf) represents withholding all scored candidates, including score 1.
    if threshold is not None and (type(threshold) not in (int, float) or
                                  not 0 <= threshold <= math.nextafter(1.0, math.inf)):
        raise ValueError("Invalid confidence threshold")
    return policy


def filter_answer(prediction, log, threshold):
    if threshold is None or prediction["type"] == "abstain":
        return prediction, log
    confidence = log_confidence(log)
    if confidence is None or confidence >= threshold:
        return prediction, log
    candidate = {k: v for k, v in prediction.items() if k != "id"}
    return {"id": prediction["id"], "type": "abstain"}, {
        **log, "answer": {"type": "abstain"}, "candidate_answer": candidate,
        "abstain_reason": {"category": "abstain", "detail": "confidence_below_threshold",
                           "confidence": confidence, "threshold": threshold},
    }
