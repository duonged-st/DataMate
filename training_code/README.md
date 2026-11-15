# Reproducibility

This solution uses pretrained Qwen3.5-9B weights quantized by Unsloth. No fine-tuning or synthetic data generation was performed. Inference and preprocessing code lives in `/code/datamate`, with pinned Python dependencies in `/code/requirements.txt`.

`data_manifest.json` records the original 3,000-question BTC Training ZIP and its SHA-256. Public and Private Test questions are runtime inputs.

Exact preprocessing and verification commands from `/code` with a running local llama-server:

```bash
python3 -m datamate audit --data training_code/data/train.zip --output /tmp/audit
python3 -m datamate predict --data training_code/data/train.zip --questions-file training_code/dev_80.jsonl --output /tmp/dev80
python3 -m datamate score --data training_code/data/train.zip --output /tmp/dev80
```

The dev manifest replicates V4.2's 40-question random sample from each of two previously inspected Training businesses. Holdout runs should use Training businesses selected before inspecting gold answers, with `--business` and `--per-business`. Inference uses one GPU, greedy sampling and seed 20261112. Log duration can vary by machine. The offline runtime requires a CUDA 12.8-compatible NVIDIA driver and the Docker build includes CUDA compute capability 12.0.

`/code/verification` contains loader summaries, source-level audits and the CPU smoke report. To reproduce CPU tests from `/code`:

```bash
python -m unittest discover -s verification/tests -v
python cpu_smoke.py --data /path/to/public_test.zip --work /tmp/datamate-audit --output /tmp/cpu-smoke.json
```

On Kaggle, use the notebook to run smoke (10), regression (80) and a team-confirmed holdout set. Download `datamate_benchmarks.zip` and extract it. When staging assets, use `prepare_assets.py --train path/to/train.zip --benchmarks path/to/extracted_results`. This verifies the regression IDs and scores, checks source fingerprints, packages the configurations and logs under `/code/verification/benchmarks`, and records the measured regression accuracy in the manifest. CPU fixture results do not populate that metric.

A threshold experiment exports `confidence_policy.json`. Pass its path to
`prepare_assets.py --confidence-policy` when staging the image. The config includes the
calibration and validation businesses, model checksum, rank AUC and PA gains. The Docker
entry point reads the packaged config and does not read gold. Recheck the exported policy
using the same llama-server build as the final image, as a different build can emit
different token logprobs or omit them.
