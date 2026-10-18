# Training and evaluating on an NVIDIA GPU

How to LoRA-fine-tune `microsoft/Phi-4-mini-instruct` as the FINRA 2210 reviewer on an NVIDIA
GPU with PyTorch + TRL, and evaluate it. This is the CUDA counterpart of the Apple Silicon
(mlx-lm) run described in the README. The recipe is the same, so the two runs can be compared.

| Piece | File |
|---|---|
| Training | [src/finra2210/train_phi4.py](src/finra2210/train_phi4.py) |
| Evaluation | [src/finra2210/evaluate.py](src/finra2210/evaluate.py) (`--backend hf`) |
| Loss curve + summary | [src/finra2210/loss_report.py](src/finra2210/loss_report.py) (`--hf-state`) |
| Everything in one go | [scripts/run_cuda_pipeline.sh](scripts/run_cuda_pipeline.sh) |
| Python packages | [requirements-train.txt](requirements-train.txt) |

Status: tested end to end on CPU with a tiny random Phi-3-architecture model and the real
Phi-4-mini tokenizer (torch 2.14.1, transformers 5.20.1, TRL 1.16.0, PEFT 0.21.2). It has not
yet been run on an NVIDIA GPU, so the timings below are estimates.

## 1. Hardware

The base model has 3.8B parameters, ~7.7 GB of weights in bf16.

| GPU memory | Examples | Settings |
|---|---|---|
| 24 GB or more | RTX 3090 / 4090, A10, L4, A100, H100 | defaults (bf16, batch 4 × grad-accum 4) |
| 16 GB | RTX 4080, A4000, T4 (fp16) | `--batch 2 --grad-accum 8` |
| 12–16 GB | RTX 3060 12 GB, 4070 | `--qlora` (4-bit base, needs bitsandbytes on Linux) |

- **Precision:** Ampere and newer GPUs train in bf16. Older ones (T4, V100) switch to fp16
  automatically.
- **Time (rough estimate):** about 1,660 optimizer steps; roughly 1.5–2 h on one A100 and
  3–5 h on a 24 GB consumer or L4 card.
- **Evaluation is the larger cost:** test + OOD is 6,986 rows per model, and the zero-shot base
  model is evaluated too.

## 2. Code and data

```bash
git clone https://github.com/rgvvvg5mpz-collab/FINRA2210_correspondence.git && cd FINRA2210_correspondence
```

`data/` is gitignored, so the splits must be copied from the machine that built them. You need
these three files (about 116 MB in total) in `data/splits/`:

| File | Rows | MD5 |
|---|---|---|
| `train.jsonl` | 10,710 | `f859399efbc7abc2c0e04aa2311d2984` |
| `test.jsonl` | 1,186 | `36569ce553eed440dd09074cb0bac327` |
| `ood.jsonl` | 5,800 | `b7bcc715295d2f991afb9801912bf207` |

Copy them from the Mac (adjust user, host and path):

```bash
mkdir -p data/splits && rsync -av you@mac-host:~/Desktop/ClaudeCode_GIT/FINRA2210_correspondence/data/splits/ data/splits/
```

Then check them:

```bash
md5sum data/splits/train.jsonl data/splits/test.jsonl data/splits/ood.jsonl
```

If the checksums differ, the splits were regenerated (for example after a top-up), and results
won't be comparable with the mlx run. The MD5 values above are the splits used for the mlx run
of 2026-10-17.

## 3. Python environment

Use Python 3.11 or 3.12 on Linux, or Windows with WSL2, with a recent NVIDIA driver. First
install PyTorch from the index that matches your driver; `nvidia-smi` shows the highest CUDA
version it supports.

```bash
python3.12 -m venv .venv-cuda && .venv-cuda/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv-cuda/bin/pip install -r requirements-train.txt
```

Optional extras:
- `bitsandbytes`, only needed for `--qlora`:
  ```bash
  .venv-cuda/bin/pip install bitsandbytes
  ```
- `flash-attn`, only needed for `--attn flash_attention_2` (Ampere or newer). The default
  `sdpa` attention is fine without it.

Check that the GPU is visible:

```bash
.venv-cuda/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

The base model is public (MIT licence) and downloads from Hugging Face (~7.7 GB) on first use.
Set `HF_HOME` to put the cache on a large disk.

## 4. Smoke test (a few minutes)

This trains on 40 conversations for one epoch, to catch environment problems before the
full run:

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.train_phi4 --train data/splits/train.jsonl --out runs/smoke --limit 40 --epochs 1 --eval-steps 4
```

It should print `device=cuda dtype=torch.bfloat16` (or `float16`), then the example counts and
falling loss lines, and leave an adapter in `runs/smoke/`. Delete `runs/smoke` afterwards.

## 5. Full run: train → loss report → evaluate tuned → evaluate base

```bash
nohup scripts/run_cuda_pipeline.sh > runs/pipeline_cuda.log 2>&1 &
```

Follow it with `tail -f runs/pipeline_cuda.log` (training lines also go to
`runs/phi4mini-2210/train.log`). The script stops at the first failure. Rerunning it with
`STAGES=...` continues, and the evals resume from their saved predictions.

| Variable | Default | Use |
|---|---|---|
| `STAGES` | `train report eval_tuned eval_base` | run a subset, e.g. `STAGES="eval_tuned eval_base"` |
| `TRAIN_ARGS` | empty | extra training flags, e.g. `"--qlora"` or `"--batch 2 --grad-accum 8"` |
| `NGPU` | 1 | train on N GPUs with DDP (see section 7) |
| `EVAL_BATCH` | 16 | generation batch for evals; lower it if evals run out of memory |
| `PY` | `.venv-cuda/bin/python` | Python interpreter |
| `RUN` / `BASE_RUN` | `runs/phi4mini-2210` / `runs/base-hf` | output directories |

## 6. The same steps by hand

Train:

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.train_phi4 --train data/splits/train.jsonl --out runs/phi4mini-2210
```

Loss curve and training summary:

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.loss_report --hf-state runs/phi4mini-2210/trainer_state.json
```

Evaluate the fine-tuned adapter on test, then on OOD:

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.evaluate --backend hf --model runs/phi4mini-2210 --data data/splits/test.jsonl --batch 16 --resume --out runs/phi4mini-2210/eval_test.json
```

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.evaluate --backend hf --model runs/phi4mini-2210 --data data/splits/ood.jsonl --batch 16 --resume --out runs/phi4mini-2210/eval_ood.json
```

Zero-shot base model as the baseline (same commands, base model id, its own output directory):

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.evaluate --backend hf --model microsoft/Phi-4-mini-instruct --data data/splits/test.jsonl --batch 16 --resume --out runs/base-hf/eval_test.json
```

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.evaluate --backend hf --model microsoft/Phi-4-mini-instruct --data data/splits/ood.jsonl --batch 16 --resume --out runs/base-hf/eval_ood.json
```

## 7. Multi-GPU

DDP keeps one full model copy per GPU. The effective batch is batch × grad-accum × number of
GPUs, so lower `--grad-accum` to keep it at 16:

```bash
PYTHONPATH=src .venv-cuda/bin/accelerate launch --num_processes 2 -m finra2210.train_phi4 --train data/splits/train.jsonl --out runs/phi4mini-2210 --grad-accum 2
```

With the pipeline script: `NGPU=2 TRAIN_ARGS="--grad-accum 2" scripts/run_cuda_pipeline.sh`.
Evaluation runs on one GPU.

## 8. What the training does

Same as `configs/phi4mini_lora_mlx.yaml`:

| Setting | Value |
|---|---|
| Base | `microsoft/Phi-4-mini-instruct`, bf16 (fp16 on pre-Ampere; 4-bit NF4 with `--qlora`) |
| LoRA | rank 32, alpha 64, dropout 0.05, on `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` in all 32 blocks |
| Optimizer | AdamW, lr 2e-4, 3% linear warmup, cosine decay |
| Batch | 4 per GPU × 4 gradient-accumulation steps = 16 |
| Epochs | 2 |
| Max length | 2,048 tokens (the longest example is ~1.9k) |
| Loss | reviewer (assistant) turns only. Each conversation becomes one prompt/completion example per assistant turn: 13,271 train examples |
| Validation | the same stratified 5% holdout of `train.jsonl` as the mlx run (539 conversations, 705 examples; ids in `<out>/valid.ids`). `test.jsonl` is never used during training |
| Checkpoints | validation and a save every 100 optimizer steps (`--eval-steps`); the final adapter is the lowest-validation-loss checkpoint |

## 9. Outputs

`runs/phi4mini-2210/`:
- `adapter_model.safetensors` and `adapter_config.json`: the LoRA adapter, which
  `evaluate.py --model runs/phi4mini-2210` loads directly.
- `train_args.json`, `valid.ids`, `trainer_state.json` and `checkpoint-*/`.
- `loss.json`, `loss_curve.png` and `training_summary.md`, from the report stage.
- `eval_test.json` and `eval_ood.json`, plus `*.preds.jsonl` (one line per scored row).

Each eval report has overall metrics and breakdowns by difficulty, communication type,
channel and channel group. The channel groups are `chat` (SMS / IM / text blast), `chat_app`
(OOD WhatsApp / LinkedIn / X / Slack-Teams DMs) and `non_chat`. The OOD report also breaks
down by shift axis.

`--merge` also writes a standalone merged model to `runs/phi4mini-2210/merged/`, for serving
with vLLM or TGI. It is skipped with `--qlora`: merge the adapter onto a bf16 base instead.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` in training | `--batch 2 --grad-accum 8`, then `--batch 1 --grad-accum 16`, or `--qlora` |
| `CUDA out of memory` in eval | `EVAL_BATCH=8` (or `--batch 8`) and rerun; `--resume` keeps finished rows |
| `torch.cuda.is_available()` is False | the PyTorch wheel doesn't match the driver; reinstall from the right `--index-url` (`cu121`, `cu124`, `cu128`, …) |
| `--qlora needs an NVIDIA GPU` / bitsandbytes import error | bitsandbytes needs Linux + CUDA; `pip install bitsandbytes` |
| Very slow training on a T4 / V100 | expected: fp16 and no bf16 tensor cores; prefer an Ampere+ GPU |
| `flash_attention_2` import error | drop `--attn flash_attention_2` (sdpa is the default) or install `flash-attn` |
| `torch_dtype is deprecated` warning in eval | harmless |
