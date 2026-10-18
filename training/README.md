# Training

LoRA fine-tuning of **microsoft/Phi-4-mini-instruct** (3.8B) on `data/splits/train.jsonl`, on Apple
Silicon (mlx-lm) or an NVIDIA GPU (PyTorch + TRL). Both backends use the same recipe, so their
results are comparable. v0.1.0 was trained with mlx-lm on an M5 Pro (24 GB).

| File | What it is |
|---|---|
| [`run_mlx_pipeline.sh`](run_mlx_pipeline.sh) | Apple Silicon: data → train → loss report → eval fine-tuned → eval zero-shot base |
| [`configs/phi4mini_lora_mlx.yaml`](configs/phi4mini_lora_mlx.yaml) | mlx-lm LoRA config (the v0.1.0 recipe) |
| [`run_cuda_pipeline.sh`](run_cuda_pipeline.sh) | NVIDIA: train → loss report → eval fine-tuned → eval zero-shot base |
| [`TRAINING_NVIDIA.md`](TRAINING_NVIDIA.md) | step-by-step NVIDIA guide: setup, copying the data, commands, troubleshooting |
| `src/finra2210/mlx_data.py` | builds `data/mlx/{train,valid}.jsonl` from `train.jsonl` |
| `src/finra2210/train_phi4.py` | PyTorch / TRL trainer (NVIDIA) |
| `src/finra2210/loss_report.py` | `loss.json`, `loss_curve.png`, `training_summary.md` from a training log |

## Recipe

| Setting | Value |
|---|---|
| Base | `microsoft/Phi-4-mini-instruct` (NVIDIA, bf16) / `mlx-community/Phi-4-mini-instruct-4bit` (mlx, QLoRA) |
| LoRA | rank 32, alpha 64, dropout 0.05 on `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` in all 32 blocks |
| Optimizer | AdamW, lr 2e-4, 50 warmup steps (mlx) / 3% (NVIDIA), cosine decay to 2e-6 |
| Batch | 4 × gradient accumulation 4 = effective 16 |
| Epochs | 2 (6,636 iterations of 4 examples on mlx; ~1,660 optimizer steps) |
| Loss | reviewer (assistant) turns only; each 5-turn conversation becomes two examples, one per assistant turn |
| Validation | stratified 5% of `train.jsonl` (539 conversations → 705 examples); `test.jsonl` is never seen in training |
| Max length | 2,048 tokens (longest training example: 1,913) |

## Apple Silicon (mlx-lm)

```bash
python3 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-lm matplotlib markdown-it-py
```

```bash
caffeinate -i training/run_mlx_pipeline.sh
```

Stages (`STAGES="data train report eval_tuned eval_base"` selects a subset):
- **`data`:** `finra2210.mlx_data` holds out the 5% validation set (ids in `data/mlx/valid.ids`) and
  expands conversations: 13,271 train / 705 validation examples in `data/mlx/`.
- **`train`:** `mlx_lm lora -c training/configs/phi4mini_lora_mlx.yaml` writes the adapter to
  `runs/phi4mini-2210-mlx/`, with a checkpoint and a full validation pass every 400 iterations.
  Gradient checkpointing is required on 24 GB; it runs at ~7 s per iteration, ~13 h for 2 epochs.
- **`report`:** `finra2210.loss_report` writes `loss.json`, `loss_curve.png` and
  `training_summary.md` next to the adapter. It can run mid-training too.
- **`eval_tuned` / `eval_base`:** `finra2210.evaluate --backend mlx` on all of test and OOD, for the
  adapter and for the zero-shot 4-bit base (`runs/base-mlx/`). About 1.4 s per item at batch 16;
  the script retries at batch 8 and 4 if the GPU runs out of memory, and `--resume` continues an
  interrupted eval.

v0.1.0 result: best validation loss 0.287 at the last iteration (from 2.185), no overfitting. Loss
curve and summary: [`results/training/`](../results/training/).

## NVIDIA GPU (PyTorch + TRL)

```bash
nohup training/run_cuda_pipeline.sh > runs/pipeline_cuda.log 2>&1 &
```

Environment variables: `STAGES`, `TRAIN_ARGS` (for example `--qlora` or `--batch 2 --grad-accum 8`),
`NGPU` (DDP), `EVAL_BATCH`, `PY`. Outputs go to `runs/phi4mini-2210/` and `runs/base-hf/`. The
full guide, including GPU memory settings and the expected time, is
[`TRAINING_NVIDIA.md`](TRAINING_NVIDIA.md). The NVIDIA path is tested end to end on CPU with a tiny
model but has not been run on a GPU yet.

## After training

- Build `results/` (metrics, plots, gzipped predictions):
  `PYTHONPATH=src .venv-mlx/bin/python evaluation/make_results.py` (see
  [`evaluation/README.md`](../evaluation/README.md)).
- To retrain with corrections, feedback or new data, see [`retraining/`](../retraining/).
