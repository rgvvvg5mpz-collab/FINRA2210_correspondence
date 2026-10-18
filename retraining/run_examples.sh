#!/usr/bin/env bash
# Run every retraining stage on the FAKE sample data, end to end, with the project venv (.venv).
#   retraining/run_examples.sh
# DPO / GRPO training itself needs an NVIDIA GPU and requirements-train.txt, so it is not run here;
# their data preparation (pairs, rewards) is.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
R=retraining

echo "== 1. reviewer feedback -> corrections, preference pairs, KTO examples"
$PY $R/01_feedback/feedback_to_data.py --log $R/01_feedback/sample/feedback_log.sample.jsonl --out $R/01_feedback/sample/out

echo "== 2. audit: select disagreements, then apply the (sample) auditor verdicts"
$PY $R/02_audit/select_disagreements.py --gold $R/02_audit/sample/train.sample.jsonl \
    --preds $R/02_audit/sample/preds.sample.jsonl --out $R/02_audit/sample/tasks --per-chunk 2 --split-name train
$PY $R/02_audit/apply_audit.py --tasks $R/02_audit/sample/tasks --verdicts $R/02_audit/sample/verdicts --out $R/02_audit/sample/out

echo "== 3. SFT round: apply label fixes, add feedback samples, de-duplicate, incremental split"
$PY $R/03_sft/build_round.py --clean $R/03_sft/sample/in_domain.clean.sample.jsonl \
    --label-fixes $R/02_audit/sample/out/train_label_fixes.jsonl --add $R/01_feedback/sample/out/corrections.jsonl \
    --keep-test $R/03_sft/sample/test.ids --keep-train $R/03_sft/sample/train.ids --out $R/03_sft/sample/out

echo "== 4. DPO pairs from the model's mistakes (on the corrected round-2 split)"
$PY $R/04_dpo/make_pairs.py --split $R/03_sft/sample/out/splits/train.jsonl --preds $R/02_audit/sample/preds.sample.jsonl \
    --skip-ids $R/02_audit/sample/out/both_ok_ids.txt --extra $R/01_feedback/sample/out/preferences.jsonl \
    --out $R/04_dpo/sample/out/dpo_pairs.jsonl

echo "== 5. GRPO reward on sample answers"
$PY $R/05_grpo/reward.py --demo
$PY $R/05_grpo/reward.py --split $R/03_sft/sample/out/splits/train.jsonl --completions $R/05_grpo/sample/completions.sample.jsonl

echo "== 6. release gate: v0.1.0 fine-tuned vs zero-shot base (PASS), then the reverse (FAIL)"
$PY $R/06_release_gate/gate.py --candidate 'results/metrics/fine_tuned_{split}.json' \
    --baseline 'results/metrics/base_zero_shot_{split}.json' --report $R/06_release_gate/sample/gate_pass.md
if $PY $R/06_release_gate/gate.py --candidate 'results/metrics/base_zero_shot_{split}.json' \
    --baseline 'results/metrics/fine_tuned_{split}.json' --report $R/06_release_gate/sample/gate_fail.md > /dev/null; then
  echo "unexpected PASS"; exit 1
else
  echo "FAIL as expected (report: $R/06_release_gate/sample/gate_fail.md)"
fi
echo "all retraining examples ran"
