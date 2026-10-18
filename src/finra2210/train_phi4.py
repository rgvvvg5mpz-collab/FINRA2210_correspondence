"""LoRA fine-tune of microsoft/Phi-4-mini-instruct on the FINRA 2210 review task (PyTorch: NVIDIA CUDA).

  python -m finra2210.train_phi4 --train data/splits/train.jsonl --out runs/phi4mini-2210
  python -m finra2210.train_phi4 --train data/splits/train.jsonl --out runs/phi4mini-2210-qlora --qlora   # 12-16 GB GPUs
  accelerate launch --num_processes 2 -m finra2210.train_phi4 --train data/splits/train.jsonl --out runs/phi4mini-2210   # multi-GPU (DDP)

Same recipe as the mlx-lm run (configs/phi4mini_lora_mlx.yaml), so the two backends are comparable:
- validation = a stratified 5% holdout of train.jsonl (mlx_data.holdout, same seed, same ids), so
  test.jsonl is never seen during training; the ids are written to <out>/valid.ids;
- loss on reviewer turns only: each conversation becomes one prompt/completion example per
  assistant turn (mlx_data.expand), and TRL masks the prompt (completion_only_loss);
- LoRA rank 32 / alpha 64 / dropout 0.05 on qkv/o/gate_up/down in every block, lr 2e-4 cosine,
  effective batch 16, 2 epochs.

bf16 on Ampere or newer (A100, A10, L4, RTX 30xx/40xx, H100); fp16 on older GPUs (T4, V100).
--qlora loads the base in 4-bit NF4 (bitsandbytes, Linux + CUDA only). Apple MPS / CPU work for
smoke runs only (--limit); use the mlx-lm path on a Mac.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .mlx_data import expand, holdout

BASE_MODEL = "microsoft/Phi-4-mini-instruct"
# Phi-4-mini (Phi3 architecture) uses fused projections.
LORA_TARGETS = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


def _device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _read(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _prompt_completion(rows: list[dict]) -> Dataset:
    """Conversational prompt/completion pairs, one per assistant turn; the loss covers the completion only."""
    ex = [{"prompt": msgs[:-1], "completion": msgs[-1:]} for r in rows for msgs in expand(r["messages"])]
    return Dataset.from_list(ex)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="data/splits/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--valid", default=None,
                    help="explicit validation split JSONL; default: stratified --val-frac holdout of --train")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048, help="longest example is ~1.9k tokens")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--eval-steps", type=int, default=100, help="optimizer steps between validation runs / saves")
    ap.add_argument("--qlora", action="store_true", help="load the base in 4-bit NF4 (bitsandbytes; CUDA only)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--limit", type=int, default=None, help="train on the first N conversations (smoke test)")
    ap.add_argument("--merge", action="store_true", help="also save a merged full model under <out>/merged")
    args = ap.parse_args()

    device, dtype = _device_dtype()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(f"device={device} dtype={dtype} world_size={world}")
    if args.qlora and device != "cuda":
        raise SystemExit("--qlora needs an NVIDIA GPU (bitsandbytes)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _read(args.train)
    if args.valid:
        train_rows, valid_rows = rows, _read(args.valid)
    else:
        train_rows, valid_rows = holdout(rows, args.val_frac, args.seed)
    if args.limit:
        train_rows = train_rows[: args.limit]
        valid_rows = valid_rows[: max(8, args.limit // 10)]
    (out / "valid.ids").write_text("".join(r["id"] + "\n" for r in valid_rows))
    train_ds, valid_ds = _prompt_completion(train_rows), _prompt_completion(valid_rows)
    print(f"train: {len(train_rows)} conversations -> {len(train_ds)} examples; "
          f"valid: {len(valid_rows)} -> {len(valid_ds)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"

    # transformers >= 4.56 renamed torch_dtype to dtype
    dtype_key = "dtype" if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56) else "torch_dtype"
    load_kw: dict = {dtype_key: dtype, "attn_implementation": args.attn}
    if device == "cuda":
        # one full copy per process under DDP; device_map="auto" would shard across GPUs instead
        load_kw["device_map"] = {"": local_rank}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    if device != "cuda":
        model.to(device)

    peft_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    )

    # 3% linear warmup, as whole steps (transformers 5 dropped warmup_ratio; 4.x needs an int here)
    total_steps = math.ceil(len(train_ds) / (args.batch * world) / args.grad_accum) * args.epochs
    cfg = SFTConfig(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.03 * total_steps)),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        packing=False,
        completion_only_loss=True,
        bf16=(device == "cuda" and dtype == torch.bfloat16),
        fp16=(device == "cuda" and dtype == torch.float16),
        gradient_checkpointing=(device == "cuda"),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False if world > 1 else None,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    if not trainer.is_world_process_zero():
        return
    trainer.save_model(str(out))  # adapter weights of the best (lowest validation loss) checkpoint
    tok.save_pretrained(str(out))
    trainer.state.save_to_json(str(out / "trainer_state.json"))  # loss history for finra2210.loss_report --hf-state
    (out / "train_args.json").write_text(json.dumps(
        {**vars(args), "world_size": world, "effective_batch": args.batch * args.grad_accum * world,
         "train_examples": len(train_ds), "valid_examples": len(valid_ds)}, indent=2))

    if args.merge:
        if args.qlora:
            print("--merge skipped: merge the adapter into a bf16/fp16 base instead of the 4-bit one")
        else:
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(out / "merged", safe_serialization=True)
            tok.save_pretrained(out / "merged")
            print(f"merged model saved to {out / 'merged'}")


if __name__ == "__main__":
    main()
