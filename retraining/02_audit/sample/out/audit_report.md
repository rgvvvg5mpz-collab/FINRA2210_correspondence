# Audit report

Verdicts: 4 of 4 audited items.

| Decision | Count | Share |
|---|---|---|
| gold | 1 | 25% |
| model | 1 | 25% |
| neither | 1 | 25% |
| both_ok | 1 | 25% |

Label fixes: 2 train, 0 test/OOD (eval only), 0 invalid.

## Categories involved in a disagreement

A high label-wrong share means the dataset, not the model, needs work for that category.

| Category | Audited | Label right (gold) | Label wrong (model / neither) | Both OK | Label-wrong share |
|---|---|---|---|---|---|
| `not_fair_and_balanced` | 2 | 0 | 1 | 1 | 50% |
| `member_name_missing` | 1 | 0 | 1 | 0 | 100% |
| `fund_performance_disclosure` | 1 | 1 | 0 | 0 | 0% |
| `product_specific_omission` | 1 | 0 | 1 | 0 | 100% |
