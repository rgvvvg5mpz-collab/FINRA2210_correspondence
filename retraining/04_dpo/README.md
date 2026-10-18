# 4. DPO: preference training on the model's own mistakes

DPO shows the model two answers to the same prompt, a preferred one and a rejected one, and moves
it toward the preferred one without a reward model. Here the rejected answers are the SFT model's
own mistakes on training prompts, so no reviewer data is needed (reviewer pairs from `01_feedback`
are added when there are some).

## 1. Pairs

```bash
.venv/bin/python retraining/04_dpo/make_pairs.py --split data/rounds/r2/splits/train.jsonl \
    --preds runs/phi4mini-2210-r2/eval_train.preds.jsonl --skip-ids retraining/02_audit/out/both_ok_ids.txt \
    --extra retraining/01_feedback/out/preferences.jsonl --out retraining/04_dpo/out/dpo_pairs.jsonl
```

- `--preds`: the SFT model's predictions on the **train** split (`finra2210.evaluate --data .../train.jsonl`).
  The script refuses OOD items.
- **chosen** is the reference answer for the gold label; **rejected** is the same answer format
  rendered from the model's wrong verdict, so the pair differs in the verdict, not in style. When
  the model's answer could not be parsed, its raw text is the rejected side.
- Use the audited labels (the round-2 split from `03_sft`): pairs built on wrong labels teach the
  errors. `--skip-ids` leaves out defensible differences found by the audit.
- `--skip-severity-only` drops pairs that differ only in severity.

Output format (TRL conversational preference format), one per line:
`{"id", "kind", "prompt": [system, user], "chosen": [assistant], "rejected": [assistant]}`.
`kind` says what was wrong: `compliant`, `communication_type`, `categories`, `severity`,
`unparsable` or `feedback`.

## 2. Training (NVIDIA GPU)

```bash
PYTHONPATH=src .venv-cuda/bin/python retraining/04_dpo/train_dpo.py \
    --pairs retraining/04_dpo/out/dpo_pairs.jsonl --sft-adapter runs/phi4mini-2210-r2 \
    --out runs/phi4mini-2210-dpo --merge
```

- Starts from base + the SFT adapter merged (a PEFT adapter from `finra2210.train_phi4`; the mlx
  adapter is a different format), trains a new LoRA adapter with the DPO loss, and uses the SFT
  model as the reference. `--beta` (default 0.1) controls how far it may move.
- Defaults: lr 5e-6, 1 epoch, batch 2 × grad-accum 8, LoRA rank 16.
- `--merge` saves `<out>/merged`, which `finra2210.evaluate --backend hf --model <out>/merged` can score.
- Not run in this repository yet (no GPU here). Smoke-test with `--limit 16 --epochs 1`.

Then evaluate on the fixed test and OOD sets and run `06_release_gate`.

## Sample (fake)

`sample/out/dpo_pairs.jsonl`: four pairs from the sample data, two from the model's mistakes on the
audited round-2 split (a missed second category, a missed fund-performance disclosure) and two
from reviewer feedback.
