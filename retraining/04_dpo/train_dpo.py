"""DPO on top of the supervised (SFT) model, with TRL on an NVIDIA GPU.

  PYTHONPATH=src .venv-cuda/bin/python retraining/04_dpo/train_dpo.py \
      --pairs retraining/04_dpo/out/dpo_pairs.jsonl --sft-adapter runs/phi4mini-2210 --out runs/phi4mini-2210-dpo

Starts from the base model with the SFT LoRA adapter merged in (a PEFT adapter from
finra2210.train_phi4; the mlx adapter in runs/phi4mini-2210-mlx is a different format) and trains a
NEW LoRA adapter with the DPO loss. The reference model is the merged SFT model itself (TRL uses the
model with the new adapter switched off), so beta controls how far DPO may move away from it.
Evaluate the result with finra2210.evaluate (--model <merged dir>) and pass it through
06_release_gate before shipping.

Requirements: requirements-train.txt (torch, transformers, peft, trl, datasets). Written against the
TRL DPOTrainer API used by train_phi4.py's TRL version (trl 1.16); not run in this repository yet.
Smoke test first: --limit 16 --epochs 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from finra2210.train_phi4 import BASE_MODEL, LORA_TARGETS, _device_dtype  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="dpo_pairs.jsonl from make_pairs.py")
    ap.add_argument("--sft-adapter", required=True, help="PEFT adapter dir from finra2210.train_phi4 (e.g. runs/phi4mini-2210)")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1, help="strength of the pull back toward the SFT model")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="use the first N pairs (smoke test)")
    ap.add_argument("--merge", action="store_true", help="also save base + SFT + DPO merged under <out>/merged for evaluate.py")
    args = ap.parse_args()

    device, dtype = _device_dtype()
    rows = [json.loads(l) for l in Path(args.pairs).read_text().splitlines() if l.strip()]
    rows = [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows][: args.limit]
    ds = Dataset.from_list(rows)
    print(f"device={device} dtype={dtype} pairs={len(ds)}")

    tok = AutoTokenizer.from_pretrained(args.sft_adapter if (Path(args.sft_adapter) / "tokenizer_config.json").exists() else args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    # transformers >= 4.56 renamed torch_dtype to dtype (same handling as train_phi4.py)
    dtype_key = "dtype" if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(args.base, **{dtype_key: dtype})
    model = PeftModel.from_pretrained(model, args.sft_adapter).merge_and_unload()  # SFT weights become the starting point
    if device != "cuda":
        model.to(device)

    cfg = DPOConfig(
        output_dir=args.out,
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        bf16=(device == "cuda" and dtype == torch.bfloat16),
        fp16=(device == "cuda" and dtype == torch.float16),
        gradient_checkpointing=(device == "cuda"),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )
    peft_cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                          target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    if args.merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(Path(args.out) / "merged")
        tok.save_pretrained(Path(args.out) / "merged")
        print(f"merged model in {Path(args.out) / 'merged'}")


if __name__ == "__main__":
    main()
