# 3. Supervised retraining round

Builds the dataset for the next supervised (SFT) round and splits it so that every earlier item
keeps its train or test side.

```bash
.venv/bin/python retraining/03_sft/build_round.py --clean data/clean/in_domain.clean.jsonl \
    --label-fixes retraining/02_audit/out/train_label_fixes.jsonl \
    --add retraining/01_feedback/out/corrections.jsonl data/clean/chat_topup.clean.jsonl \
    --keep-test data/splits/test.jsonl --keep-train data/splits/train.jsonl --out data/rounds/r2
```

What it does:
1. **Label fixes** from the audit replace the labels of train items. A fix for a test or OOD item
   stops the script: those change the eval set, not the training data. Each fixed item keeps its
   history in `relabels`.
2. **New samples** (reviewer corrections, targeted new examples such as the chat top-up or
   member-name pairs) are checked like the dataset (verbatim excerpts, cites, type rules) and
   de-duplicated against everything already in the set (6-word shingles, overlap > 0.5). They get
   ids `<--id-prefix>-NNNNNN`; the original id is kept in `source_id`.
3. **Split** with `finra2210.split --keep-test --keep-train`: old items stay where they were; only
   the new ones are stratified. Test stays comparable with v0.1.0.

Outputs in `--out`: `in_domain.clean.jsonl`, `round_manifest.json`, `problems.jsonl` (skipped
items and why) and `splits/train.jsonl` / `splits/test.jsonl`.

## Then train, from the base model

Retrain on the whole new train split, starting from the base model rather than the previous
adapter, so earlier skills are not forgotten. Point the pipeline at the new split:

- **Apple Silicon:** `PYTHONPATH=src .venv-mlx/bin/python -m finra2210.mlx_data --train data/rounds/r2/splits/train.jsonl --out data/mlx`,
  then train with `training/configs/phi4mini_lora_mlx.yaml` and a new `adapter_path` (for example
  `runs/phi4mini-2210-mlx-r2`).
- **NVIDIA:** `PYTHONPATH=src .venv-cuda/bin/python -m finra2210.train_phi4 --train data/rounds/r2/splits/train.jsonl --out runs/phi4mini-2210-r2`.

Evaluate on the **same** `data/splits/test.jsonl` and `data/splits/ood.jsonl` as v0.1.0, then run
the release gate (`06_release_gate`).

## Sample (fake)

`sample/in_domain.clean.sample.jsonl` (five invented items), `sample/train.ids` / `sample/test.ids`
(the "previous" split). With the sample label fixes and feedback corrections, the round applies 2
fixes, drops 2 feedback items as near-duplicates of existing ones, adds 1, and keeps the test item
in test. `sample/out/` is the result.
