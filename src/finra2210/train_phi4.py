"""LoRA fine-tune of microsoft/Phi-4-mini-instruct on the FINRA 2210 review task.

  python -m finra2210.train_phi4 --train data/splits/train.jsonl --eval data/splits/test.jsonl --out runs/phi4mini-2210

Works on CUDA (recommended), Apple MPS (slow but fine for a smoke run), or CPU
(smoke runs only). The dataset is in chat format ({"messages": [...]}) so TRL
applies Phi-4-mini's chat template itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "microsoft/Phi-4-mini-instruct"
# Phi-4-mini (Phi3 architecture) uses fused projections.
LORA_TARGETS = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


def _device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="train on the first N rows (smoke test)")
    ap.add_argument("--assistant-only-loss", action="store_true",
                    help="mask the prompt from the loss; requires a chat template with {% generation %} tags")
    ap.add_argument("--merge", action="store_true", help="also save a merged full model under <out>/merged")
    args = ap.parse_args()

    device, dtype = _device_dtype()
    print(f"device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)

    data_files = {"train": args.train}
    if args.eval:
        data_files["eval"] = args.eval
    ds = load_dataset("json", data_files=data_files)
    keep = ["messages"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    if args.limit:
        ds["train"] = ds["train"].select(range(min(args.limit, len(ds["train"]))))
        if "eval" in ds:
            ds["eval"] = ds["eval"].select(range(min(max(8, args.limit // 10), len(ds["eval"]))))

    peft_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        packing=False,
        assistant_only_loss=args.assistant_only_loss,
        bf16=(dtype == torch.bfloat16 and device == "cuda"),
        gradient_checkpointing=(device == "cuda"),
        logging_steps=10,
        eval_strategy="steps" if args.eval else "no",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="none",
        seed=42,
        use_mps_device=(device == "mps"),
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(args.out)  # adapter weights
    tok.save_pretrained(args.out)
    Path(args.out, "train_args.json").write_text(json.dumps(vars(args), indent=2))

    if args.merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(Path(args.out, "merged"), safe_serialization=True)
        tok.save_pretrained(Path(args.out, "merged"))
        print(f"merged model saved to {Path(args.out, 'merged')}")


if __name__ == "__main__":
    main()
