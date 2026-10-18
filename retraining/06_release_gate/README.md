# 6. Release gate

A candidate model ships only if it is at least as good as the model in production. `gate.py`
compares the two eval reports (from `finra2210.evaluate`) on the fixed test and OOD sets and exits
0 (PASS) or 1 (FAIL), so it can run in CI.

```bash
.venv/bin/python retraining/06_release_gate/gate.py \
    --candidate 'runs/phi4mini-2210-r2/eval_{split}.json' \
    --baseline 'results/metrics/fine_tuned_{split}.json' --report runs/phi4mini-2210-r2/gate.md
```

`{split}` is replaced by `test` and `ood`. Both reports must come from the same split files; the
gate refuses otherwise.

## Checks (`gates.json`)

| Scope | Metrics | Rule |
|---|---|---|
| overall, per split | valid JSON, compliant accuracy, violation recall, type accuracy, severity accuracy, category F1 (exact), category F1 (broad group) | an absolute floor (`min`) and a maximum fall versus the baseline (`max_drop`) |
| chat slice (SMS, IM, text blasts) | compliant accuracy, category F1 | as above |
| each broad group with ≥ 30 gold items | recall, F1 | no more than 0.05 lower |
| each difficulty | compliant accuracy, category F1 | small maximum fall |

The floors sit just under v0.1.0 (for example compliant accuracy 0.94, valid JSON 0.99, violation
recall 0.95), so a new release cannot be worse than the current one by more than noise. Tighten
them as the model improves.

## Sample

`run_examples.sh` runs the gate on the real v0.1.0 reports in `results/metrics/`:
- **fine-tuned vs zero-shot base:** PASS, all 44 checks ([`sample/gate_pass.md`](sample/gate_pass.md));
- **zero-shot base vs fine-tuned:** FAIL ([`sample/gate_fail.md`](sample/gate_fail.md)).
