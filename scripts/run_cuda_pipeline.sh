#!/usr/bin/env bash
# LoRA-train Phi-4-mini-instruct on an NVIDIA GPU (PyTorch / TRL), save the loss curve + summary,
# then evaluate it and the zero-shot base on test + OOD. CUDA counterpart of run_mlx_pipeline.sh.
#   nohup scripts/run_cuda_pipeline.sh > runs/pipeline_cuda.log 2>&1 &
#   STAGES="eval_tuned eval_base" scripts/run_cuda_pipeline.sh
#   TRAIN_ARGS="--qlora" scripts/run_cuda_pipeline.sh                  # 12-16 GB GPUs
#   NGPU=2 scripts/run_cuda_pipeline.sh                                # multi-GPU training (DDP)
# Evals use --resume, so rerunning after a crash continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv-cuda/bin/python}
RUN=${RUN:-runs/phi4mini-2210}
BASE_RUN=${BASE_RUN:-runs/base-hf}
BASE=microsoft/Phi-4-mini-instruct
STAGES=${STAGES:-"train report eval_tuned eval_base"}
NGPU=${NGPU:-1}
EVAL_BATCH=${EVAL_BATCH:-16}
export PYTHONPATH=src
mkdir -p "$RUN" "$BASE_RUN"

for stage in $STAGES; do
  echo "=== $stage $(date -u +%FT%TZ)"
  case $stage in
    train)
      if [ "$NGPU" -gt 1 ]; then
        $PY -m accelerate.commands.launch --num_processes "$NGPU" -m finra2210.train_phi4 \
          --train data/splits/train.jsonl --out "$RUN" ${TRAIN_ARGS:-} 2>&1 | tee "$RUN/train.log"
      else
        $PY -m finra2210.train_phi4 --train data/splits/train.jsonl --out "$RUN" ${TRAIN_ARGS:-} 2>&1 | tee "$RUN/train.log"
      fi ;;
    report) $PY -m finra2210.loss_report --hf-state "$RUN/trainer_state.json" ;;  # loss.json, loss_curve.png, training_summary.md
    eval_tuned)
      for split in test ood; do
        $PY -m finra2210.evaluate --backend hf --model "$RUN" --data data/splits/$split.jsonl \
          --batch "$EVAL_BATCH" --resume --out "$RUN/eval_$split.json" > "$RUN/eval_$split.log" 2>&1
      done ;;
    eval_base)
      for split in test ood; do
        $PY -m finra2210.evaluate --backend hf --model "$BASE" --data data/splits/$split.jsonl \
          --batch "$EVAL_BATCH" --resume --out "$BASE_RUN/eval_$split.json" > "$BASE_RUN/eval_$split.log" 2>&1
      done ;;
    *) echo "unknown stage $stage"; exit 2 ;;
  esac
done
echo "=== done $(date -u +%FT%TZ)"
