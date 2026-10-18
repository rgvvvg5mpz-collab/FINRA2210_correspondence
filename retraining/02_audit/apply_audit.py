"""Collect auditor verdicts and turn them into label fixes.

  .venv/bin/python retraining/02_audit/apply_audit.py --tasks retraining/02_audit/tasks \
      --verdicts retraining/02_audit/verdicts --out retraining/02_audit/out

  # the sample
  .venv/bin/python retraining/02_audit/apply_audit.py --tasks retraining/02_audit/sample/tasks \
      --verdicts retraining/02_audit/sample/verdicts --out retraining/02_audit/sample/out

Verdict files (verdicts/chunk_XXXX.json, one per task chunk) hold {"verdicts": [{id, decision,
label?, reason}]} with decision one of:
  gold      the dataset label is right; the model is wrong          -> nothing to fix (training signal)
  model     the model's verdict is right; the label is wrong        -> label := model verdict
  neither   both are wrong or incomplete; `label` gives the answer  -> label := auditor's label
  both_ok   a defensible difference (e.g. one extra (d)(1) category) -> nothing to fix

Writes to --out:
  train_label_fixes.jsonl  fixes for train items, consumed by 03_sft/build_round.py
  eval_label_fixes.jsonl   fixes for test / OOD items: NEVER trained on; apply them only to publish a
                           corrected eval set version (and re-score every model on it)
  invalid.jsonl            verdicts whose label fails the label checks
  both_ok_ids.txt          items with a defensible difference; 04_dpo/make_pairs.py --skip-ids leaves them out
  audit_report.md          decision counts, and how often the label was wrong per category
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import categories, check_label, write_jsonl  # noqa: E402

DECISIONS = {"gold", "model", "neither", "both_ok"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = {}
    for f in sorted(Path(args.tasks).glob("chunk_*.json")):
        for it in json.loads(f.read_text())["items"]:
            tasks[it["id"]] = it
    verdicts = []
    for f in sorted(Path(args.verdicts).glob("chunk_*.json")):
        verdicts += json.loads(f.read_text())["verdicts"]

    train_fixes, eval_fixes, invalid, both_ok = [], [], [], []
    decisions: Counter = Counter()
    per_cat: dict[str, Counter] = defaultdict(Counter)
    for v in verdicts:
        it = tasks.get(v.get("id"))
        if it is None or v.get("decision") not in DECISIONS:
            invalid.append({**v, "_problems": ["unknown id or decision"]})
            continue
        d = v["decision"]
        decisions[d] += 1
        for c in categories(it["gold_label"]) ^ categories(it["model_verdict"]):
            per_cat[c][d] += 1
        if d == "both_ok":
            both_ok.append(it["id"])
        if d in ("gold", "both_ok"):
            continue
        label = it["model_verdict"] if d == "model" else v.get("label")
        sample = {k: it[k] for k in ("id", "channel", "metadata", "communication")}
        problems = check_label(sample, label or {})
        if problems:
            invalid.append({**v, "_problems": problems})
            continue
        fix = {"id": it["id"], "split": it["split"], "label": label, "previous_label": it["gold_label"],
               "decision": d, "reason": v.get("reason", ""), "auditor": v.get("auditor", "")}
        (train_fixes if it["split"] == "train" else eval_fixes).append(fix)

    out = Path(args.out)
    write_jsonl(out / "train_label_fixes.jsonl", train_fixes)
    write_jsonl(out / "eval_label_fixes.jsonl", eval_fixes)
    write_jsonl(out / "invalid.jsonl", invalid)
    (out / "both_ok_ids.txt").write_text("".join(f"{i}\n" for i in both_ok))
    n = sum(decisions.values())
    lines = ["# Audit report", "", f"Verdicts: {n} of {len(tasks)} audited items.", "",
             "| Decision | Count | Share |", "|---|---|---|"]
    lines += [f"| {d} | {decisions[d]} | {decisions[d] / max(1, n):.0%} |" for d in ("gold", "model", "neither", "both_ok")]
    lines += ["", f"Label fixes: {len(train_fixes)} train, {len(eval_fixes)} test/OOD (eval only), {len(invalid)} invalid.", "",
              "## Categories involved in a disagreement", "",
              "A high label-wrong share means the dataset, not the model, needs work for that category.", "",
              "| Category | Audited | Label right (gold) | Label wrong (model / neither) | Both OK | Label-wrong share |", "|---|---|---|---|---|---|"]
    for c, k in sorted(per_cat.items(), key=lambda t: -sum(t[1].values())):
        tot = sum(k.values())
        wrong = k["model"] + k["neither"]
        lines.append(f"| `{c}` | {tot} | {k['gold']} | {wrong} | {k['both_ok']} | {wrong / tot:.0%} |")
    (out / "audit_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    for r in invalid:
        print(f"  invalid {r.get('id')}: {'; '.join(r['_problems'])}")


if __name__ == "__main__":
    main()
