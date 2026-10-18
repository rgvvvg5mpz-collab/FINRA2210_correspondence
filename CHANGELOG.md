# Changelog

## Unreleased (on `main`, after the v0.1.0 tag)

No new model or data. The repository was reorganized and documented so that anyone with access can
clone it and use it:

- **Layout:** `training/`, `evaluation/`, `results/`, `retraining/` and `docs/`, each with a README;
  the dataset (`data/splits`, `data/clean`) and its curation sources are now in git.
- **Results:** the full v0.1.0 evaluation (fine-tuned and zero-shot base, test and OOD) with plots,
  per-category tables, OOD-axis and unseen-category results, and gzipped predictions in `results/`.
- **Evaluation:**
  - category metrics at rule, 2210(d)(1)-family and broad-group level, and per broad group;
  - `--score-only` re-scoring;
  - a live confusion-matrix dashboard;
  - fixes for `"violations": null` (crashed an eval) and string `compliant` values.
- **Model use:** `python -m finra2210.review` reviews one message with a trained adapter.
- **Taxonomy:** 5 broad groups and descriptive sub-groups for the 23 categories, all documented in
  the model card.
- **Retraining kit:** reviewer feedback, label audits, SFT rounds, DPO, GRPO and a release gate,
  with sample data.
- **Board deck:** `Marketing/FINRA2210_Board_Deck.html`, a 13-slide interactive briefing (executive summary,
  architecture, evaluation, deployment, governance) with click-through detail panels and a taxonomy explorer.
- **Rule-change watch:** `retraining/00_rule_watch/` diffs the archived FINRA / SEC sources, drafts
  a change memo mapped to categories and brief lines, and marks stale items; `DIGEST_VERSION` is
  stamped on verdicts and eval reports.

## v0.1.0 (2026-10-17)

First trained model: a LoRA adapter for Phi-4-mini-instruct (4-bit, MLX), released with the dataset.

- **Data:** 11,896 in-domain items (10,710 train / 1,186 test) and 5,800 OOD items, written by
  Claude Code subagents and validated by code.
- **Training:** mlx-lm, 2 epochs; validation loss 2.185 → 0.287.
- **Results:**

  | | Compliant / not compliant | Category F1, exact | Category F1, broad group |
  |---|---|---|---|
  | Test | 96.5% | 0.80 | 0.92 |
  | OOD | 95.3% | 0.69 | 0.88 |

- **Release assets:** the adapter, the dataset archive and the four eval reports, each with a
  checksum.
