# 2. Audit disagreements between the model and the labels

The labels are synthetic. Before training on more data, find out how often the model is right and
the label is wrong. This stage selects the disagreements, has audit agents (Claude Opus in Claude
Code, the same way the dataset was generated) decide each one, and turns their decisions into
label fixes.

## Steps

1. **Predictions on the train split** (never use test or OOD for training fixes). On Apple
   Silicon this takes about 4 hours for the 10,710 train items; `--resume` continues after an
   interruption, and `--limit 2000` is enough for a first audit:

   ```bash
   PYTHONPATH=src .venv-mlx/bin/python -m finra2210.evaluate --backend mlx --model runs/phi4mini-2210-mlx \
       --data data/splits/train.jsonl --batch 16 --resume --out runs/phi4mini-2210-mlx/eval_train.json
   ```

2. **Select disagreements** into chunks of 20, most important first: a different compliant call,
   then a different type or category set (the confusable categories in `--focus` first), then
   severity only.

   ```bash
   .venv/bin/python retraining/02_audit/select_disagreements.py --gold data/splits/train.jsonl \
       --preds runs/phi4mini-2210-mlx/eval_train.preds.jsonl --out retraining/02_audit/tasks --limit 400
   ```

3. **Audit.** In Claude Code, give each chunk to one agent with
   [`AUDITOR_INSTRUCTIONS.md`](AUDITOR_INSTRUCTIONS.md). Each agent writes
   `verdicts/chunk_XXXX.json` with one decision per item:

   | Decision | Meaning | Effect |
   |---|---|---|
   | `gold` | the label is right, the model is wrong | none (a training signal for DPO) |
   | `model` | the model is right, the label is wrong | label := model verdict |
   | `neither` | both wrong or incomplete; the agent writes the label | label := agent's label |
   | `both_ok` | a defensible difference (e.g. one extra (d)(1) category) | none; left out of DPO pairs |

4. **Apply:**

   ```bash
   .venv/bin/python retraining/02_audit/apply_audit.py --tasks retraining/02_audit/tasks \
       --verdicts retraining/02_audit/verdicts --out retraining/02_audit/out
   ```

   Writes `train_label_fixes.jsonl` (for `03_sft`), `eval_label_fixes.jsonl` (test / OOD items:
   never trained on), `both_ok_ids.txt` (for `04_dpo --skip-ids`), `invalid.jsonl` (labels that fail
   the checks) and `audit_report.md` (decision counts and, per category, how often the label was
   wrong).

A human compliance reviewer should spot-check a random sample of the agents' decisions, especially
`model` and `neither`.

## Eval-set fixes

Test and OOD must stay comparable across versions. Never move a test or OOD item into training. If
the audit finds wrong test or OOD labels, publish a corrected eval set as a new version (for
example `test_v2.jsonl`), re-score every model on it with `finra2210.evaluate --score-only`, and say
in `results/` which version a number comes from.

## Sample (fake)

`sample/train.sample.jsonl` (four invented train items), `sample/preds.sample.jsonl` (the model's
verdicts), `sample/tasks/` (what step 2 writes), `sample/verdicts/` (what auditors would write: one
of each decision) and `sample/out/` (what step 4 writes).
