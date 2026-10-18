#!/usr/bin/env bash
# LoRA-train Phi-4-mini-instruct with mlx-lm, save the loss curve + summary, then evaluate it and
# the zero-shot base on test + OOD.
#   caffeinate -i scripts/run_mlx_pipeline.sh            # all stages
#   STAGES="eval_tuned eval_base" scripts/run_mlx_pipeline.sh
# Evals use --resume, so rerunning after a crash continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-mlx/bin/python
RUN=runs/phi4mini-2210-mlx
BASE_RUN=runs/base-mlx
BASE=mlx-community/Phi-4-mini-instruct-4bit
STAGES=${STAGES:-"data train report eval_tuned eval_base"}
export PYTHONPATH=src
mkdir -p "$RUN" "$BASE_RUN"

# evaluate MODEL SPLIT OUTDIR: resumable; on failure (e.g. Metal out-of-memory when something
# else is using the GPU) retry from the saved predictions with a smaller batch.
evaluate() {
  local model=$1 split=$2 dir=$3 batch
  for batch in ${EVAL_BATCHES:-16 8 4}; do
    echo "--- eval $model on $split, batch $batch $(date -u +%FT%TZ)"
    if $PY -m finra2210.evaluate --backend mlx --model "$model" --data data/splits/$split.jsonl \
         --batch "$batch" --resume --out "$dir/eval_$split.json" >> "$dir/eval_$split.log" 2>&1; then
      return 0
    fi
    echo "--- eval failed at batch $batch; see $dir/eval_$split.log"
  done
  return 1
}

for stage in $STAGES; do
  echo "=== $stage $(date -u +%FT%TZ)"
  case $stage in
    data) $PY -m finra2210.mlx_data --train data/splits/train.jsonl --out data/mlx --val-frac 0.05 ;;
    train) $PY -m mlx_lm lora -c configs/phi4mini_lora_mlx.yaml 2>&1 | tee "$RUN/train.log" ;;
    report) $PY -m finra2210.loss_report --log "$RUN/train.log" ;;  # loss.json, loss_curve.png, training_summary.md
    eval_tuned) for split in test ood; do evaluate "$RUN" $split "$RUN"; done ;;
    eval_base) for split in test ood; do evaluate "$BASE" $split "$BASE_RUN"; done ;;
    *) echo "unknown stage $stage"; exit 2 ;;
  esac
done
echo "=== done $(date -u +%FT%TZ)"
