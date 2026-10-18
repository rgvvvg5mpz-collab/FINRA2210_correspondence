"""Reinforcement learning with the automatic reward in reward.py (GRPO, TRL), on an NVIDIA GPU.

  PYTHONPATH=src .venv-cuda/bin/python retraining/05_grpo/train_grpo.py \
      --split data/rounds/r2/splits/train.jsonl --start-adapter runs/phi4mini-2210-dpo/merged --out runs/phi4mini-2210-grpo

For every prompt the model writes --generations answers; reward.py scores each one against the gold
label, and GRPO moves the model toward the better-than-average answers of the group. No reward
model and no human labels are needed beyond the dataset's own labels. A KL penalty (--beta) keeps
the model close to where it started.

--start-adapter is either a merged model directory (e.g. <dpo out>/merged) or a PEFT adapter from
finra2210.train_phi4 / train_dpo.py, which is merged into --base first. A new LoRA adapter is trained.

Cost: --generations answers per prompt (default 6) of up to --max-new-tokens tokens, so expect several
times the compute of an SFT epoch; start with --limit 200. Written against TRL's GRPOTrainer
(trl 1.16, as in requirements-train.txt); not run in this repository yet.
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
from trl import GRPOConfig, GRPOTrainer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
from finra2210.train_phi4 import BASE_MODEL, LORA_TARGETS, _device_dtype  # noqa: E402
from reward import finra_reward  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="train split (chat format) with gold labels")
    ap.add_argument("--start-adapter", required=True, help="merged model dir, or a PEFT adapter dir to merge into --base")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", type=int, default=6, help="answers sampled per prompt")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--beta", type=float, default=0.04, help="KL penalty toward the starting model")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch", type=int, default=6, help="per-device prompts x generations must divide evenly")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="use the first N prompts (smoke test)")
    args = ap.parse_args()

    device, dtype = _device_dtype()
    rows = [json.loads(l) for l in Path(args.split).read_text().splitlines() if l.strip()][: args.limit]
    ds = Dataset.from_list([{"prompt": r["messages"][:2], "label": json.dumps(r["label"]), "communication": r["communication"]}
                            for r in rows])
    print(f"device={device} dtype={dtype} prompts={len(ds)}")

    dtype_key = "dtype" if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56) else "torch_dtype"
    start = Path(args.start_adapter)
    if (start / "adapter_config.json").exists():
        tok = AutoTokenizer.from_pretrained(args.base)
        model = AutoModelForCausalLM.from_pretrained(args.base, **{dtype_key: dtype})
        model = PeftModel.from_pretrained(model, str(start)).merge_and_unload()
    else:
        tok = AutoTokenizer.from_pretrained(str(start))
        model = AutoModelForCausalLM.from_pretrained(str(start), **{dtype_key: dtype})
    tok.pad_token = tok.pad_token or tok.eos_token
    if device != "cuda":
        model.to(device)

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.generations,
        max_completion_length=args.max_new_tokens,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        bf16=(device == "cuda" and dtype == torch.bfloat16),
        fp16=(device == "cuda" and dtype == torch.float16),
        gradient_checkpointing=(device == "cuda"),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )
    peft_cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                          target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model, reward_funcs=[finra_reward], args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(Path(args.out) / "merged")
    tok.save_pretrained(Path(args.out) / "merged")
    print(f"adapter in {args.out}; merged model for evaluate.py in {Path(args.out) / 'merged'}")


if __name__ == "__main__":
    main()
