# 5. GRPO: reinforcement learning with an automatic reward

GRPO samples several answers per prompt, scores each one, and moves the model toward the
better-than-average answers in each group. Because every answer ends in a JSON verdict, the score
comes from code that compares it with the label. No reward model or human ratings are needed.

## The reward (`reward.py`)

| Component | Weight | Scores |
|---|---|---|
| format | gate | no parsable verdict with the six keys → total −1 |
| compliant | 0.30 | right compliant / not-compliant call |
| type | 0.10 | right communication type |
| category | 0.20 | exact-category F1 |
| broad_group | 0.10 | F1 at broad-group level (partial credit for near-misses) |
| severity | 0.10 | right severity |
| grounding | 0.15 | share of cited excerpts that really appear in the message |
| cite | 0.05 | share of findings with the rule their category requires |
| missed_high | −0.50 | a high-severity violation called compliant |

A perfect answer scores +1. The weights encode the business risk: missing a guarantee or a
projection costs far more than an extra finding, and made-up quotes are penalised. Change
`WEIGHTS` to shift the balance; re-run `--demo` to see the effect.

```bash
.venv/bin/python retraining/05_grpo/reward.py --demo
```

```
perfect answer                     +1.00
right group, wrong category        +0.80
made-up excerpt                    +0.85
wrong rule cite                    +0.95
severity off by one                +0.90
called compliant (missed high)     -0.20
no JSON verdict                    -1.00
```

Score a file of answers against a split with `--split ... --completions ...` (format: one
`{"id", "completion"}` per line; see `sample/completions.sample.jsonl`).

## Training (NVIDIA GPU)

```bash
PYTHONPATH=src .venv-cuda/bin/python retraining/05_grpo/train_grpo.py \
    --split data/rounds/r2/splits/train.jsonl --start-adapter runs/phi4mini-2210-dpo/merged \
    --out runs/phi4mini-2210-grpo --limit 200
```

- `--start-adapter`: a merged model dir (the DPO output) or a PEFT adapter to merge into the base.
- `--generations` answers per prompt (default 6), up to `--max-new-tokens` (768) each, with a KL
  penalty `--beta` (0.04) toward the starting model. Expect several times the compute of an SFT
  epoch; start with `--limit 200`.
- Saves the adapter and `<out>/merged` for `finra2210.evaluate --backend hf`.
- Not run in this repository yet (no GPU here).

**Watch for reward hacking.** If the model learns to list many categories (to catch every gold
one) or to stop flagging, the gate's precision and recall checks will show it; adjust `WEIGHTS`.
Do this stage only after the labels are audited (`02_audit`): RL amplifies label errors.
