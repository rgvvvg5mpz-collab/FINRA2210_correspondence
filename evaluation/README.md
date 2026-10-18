# Evaluation

How a model is scored, how to run the scoring, and how the results in [`results/`](../results/)
are built.

| File | What it is |
|---|---|
| `src/finra2210/evaluate.py` | runs a model on a split and scores it (mlx or Hugging Face backend) |
| `src/finra2210/analysis.py` | confusion matrices, per-category and per-broad-group metrics (shared by the two tools below) |
| [`live_confusion.py`](live_confusion.py) | live dashboard for a running eval, refreshed every 10 scored items |
| [`make_results.py`](make_results.py) | builds `results/`: metrics, plots, gzipped predictions and the results README |

## Eval sets

| Set | Items | What it tests |
|---|---|---|
| `data/splits/test.jsonl` | 1,186 | in-domain: same distribution as training, never trained on |
| `data/splits/ood.jsonl` | 5,800 | out-of-distribution: each item shifted on 1–3 axes (channel, product, audience, style, firm type, regulatory twist) and two categories never seen in training |

The model is scored on its first answer to the first user turn (system + user message); follow-up
turns are training-only.

## Running an eval

```bash
PYTHONPATH=src .venv-mlx/bin/python -m finra2210.evaluate --backend mlx --model runs/phi4mini-2210-mlx \
    --data data/splits/test.jsonl --batch 16 --resume --out runs/phi4mini-2210-mlx/eval_test.json
```

- `--model`: an adapter directory (the base is read from `adapter_config.json`), a merged model, or
  a Hugging Face id (zero-shot baseline).
- `--backend hf` for PyTorch (NVIDIA), `--backend mlx` for Apple Silicon.
- Predictions are appended to `<out>.preds.jsonl` as they are made (`{id, pred, raw}`; `raw` holds
  the text when the answer could not be parsed). `--resume` skips ids already there.
- `--score-only` re-scores an existing `<out>.preds.jsonl` without loading a model, for example after
  adding a metric. It is safe while another eval is still writing the file.

Greedy decoding, up to 1,024 new tokens. The verdict is the last fenced json block in the answer.

## Metrics

| Metric | Meaning |
|---|---|
| `json_parse_rate` | share of answers with a parsable JSON verdict |
| `compliant_accuracy` | compliant / not-compliant call correct (an unparsable answer counts as wrong) |
| `violation_detection` | precision / recall / F1 of "at least one violation", per item |
| `communication_type_accuracy` | correspondence vs retail vs institutional correct |
| `severity_accuracy` | none / low / medium / high correct |
| `violation_category_micro` | precision / recall / F1 over (item, category) pairs, exact category |
| `category_rule_level_micro` | the same, counting a category as right if it has the gold category's rule cite (e.g. `promissory_or_guarantee` vs `exaggerated_or_unwarranted_claim`, both (d)(1)(B)) |
| `category_d1_family_micro` | the same, with all 2210(d)(1) content standards as one group |
| `category_broad_group_micro` | the same, at the level of the 5 broad groups (`BROAD_GROUPS` in `taxonomy.py`) |
| `broad_groups` | precision / recall / F1, support, hits and false alarms for each broad group |
| `rule_cite_micro` | precision / recall / F1 of the rule cite text the model wrote |

The model's `compliant` field is read by meaning (`"false"`, `"No"` → not compliant), and
`"violations": null` counts as no findings.

Every report has these metrics overall and broken down by difficulty, gold communication type,
channel, channel group (`chat` = SMS / IM / text blast; `chat_app` = the OOD-only WhatsApp,
LinkedIn, X and Slack/Teams DMs; `non_chat`), chat × communication type and, for OOD, shift axis.

Why four category levels: several categories share a rule paragraph, and most category errors are
near-misses between neighbours (for example `product_specific_omission` read as
`not_fair_and_balanced`). On v0.1.0, category F1 is 0.80 exact and 0.92 at broad-group level on test.

## Live dashboard

```bash
.venv/bin/python evaluation/live_confusion.py
```

Open http://localhost:8765. It reads the predictions file of a running (or finished) eval and shows
the confusion matrices for compliant, type and severity; category precision / recall at the four
levels; broad-group and per-category hits, misses and false alarms; the most common category swaps;
and accuracy by OOD axis and difficulty. Defaults to the fine-tuned model on OOD; for another run:
`--preds runs/base-mlx/eval_ood.preds.jsonl --port 8766`, or `--gold data/splits/test.jsonl` with a
test predictions file. In Claude Code the two dashboards are in `.claude/launch.json`.

## Building `results/`

```bash
PYTHONPATH=src .venv-mlx/bin/python evaluation/make_results.py
```

Reads `runs/phi4mini-2210-mlx/` and `runs/base-mlx/` (`--tuned`, `--base` to change) and rewrites
`results/`: the four eval reports, `per_category.csv`, `summary.json`, nine plots, gzipped
predictions, the training curve and summary, and `results/README.md`. Needs matplotlib (in
`.venv-mlx`). Re-score any predictions file from `results/predictions/` with
`gunzip -k` and `finra2210.evaluate --score-only`.
