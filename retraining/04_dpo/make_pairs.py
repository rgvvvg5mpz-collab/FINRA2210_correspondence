"""Build DPO preference pairs from the current model's own mistakes on the TRAIN split.

  # 1. predictions of the current model on train (not test / OOD)
  PYTHONPATH=src .venv-mlx/bin/python -m finra2210.evaluate --backend mlx --model runs/phi4mini-2210-mlx \
      --data data/splits/train.jsonl --out runs/phi4mini-2210-mlx/eval_train.json
  # 2. pairs
  .venv/bin/python retraining/04_dpo/make_pairs.py --split data/splits/train.jsonl \
      --preds runs/phi4mini-2210-mlx/eval_train.preds.jsonl \
      --extra retraining/01_feedback/out/preferences.jsonl --out retraining/04_dpo/out/dpo_pairs.jsonl

  # the sample
  # the sample: the round-2 split from 03_sft (labels already corrected by the audit)
  .venv/bin/python retraining/04_dpo/make_pairs.py --split retraining/03_sft/sample/out/splits/train.jsonl \
      --preds retraining/02_audit/sample/preds.sample.jsonl --skip-ids retraining/02_audit/sample/out/both_ok_ids.txt \
      --extra retraining/01_feedback/sample/out/preferences.jsonl --out retraining/04_dpo/sample/out/dpo_pairs.jsonl

Each pair is in TRL's conversational preference format:
  {"prompt": [system, user], "chosen": [{"role": "assistant", ...}], "rejected": [{"role": "assistant", ...}]}
chosen = the reference answer for the gold label; rejected = the same answer format rendered from the
model's wrong verdict (so the pair differs only in the verdict, not in style), or the model's raw
text when its answer could not be parsed. Use the corrected labels from 02_audit (rebuild the split
with 03_sft) before making pairs, or the pairs will teach the label errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import answer_for, categories, load_jsonl, prompt_messages, same_verdict, write_jsonl  # noqa: E402


def kind(gold: dict, pred: dict | None) -> str:
    if pred is None:
        return "unparsable"
    if bool(gold["compliant"]) != bool(pred.get("compliant")):
        return "compliant"
    if gold["communication_type"] != pred.get("communication_type"):
        return "communication_type"
    if categories(gold) != categories(pred):
        return "categories"
    return "severity"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="the split the predictions were made on (train.jsonl)")
    ap.add_argument("--preds", required=True, help="<eval out>.preds.jsonl from finra2210.evaluate")
    ap.add_argument("--extra", nargs="*", default=[], help="more pairs in the same format, e.g. 01_feedback preferences.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-severity-only", action="store_true", help="drop pairs whose only difference is severity")
    ap.add_argument("--skip-ids", default=None, help="ids to leave out, one per line (02_audit writes both_ok_ids.txt: defensible differences)")
    args = ap.parse_args()

    rows = {r["id"]: r for r in load_jsonl(args.split)}
    if any(r["id"].startswith(("ood-",)) for r in rows.values()):
        raise SystemExit("refusing to build training pairs from the OOD set")
    skip = set(Path(args.skip_ids).read_text().split()) if args.skip_ids else set()
    pairs, kinds = [], Counter()
    for p in load_jsonl(args.preds):
        r = rows.get(p["id"])
        if r is None or same_verdict(r["label"], p["pred"]):
            continue
        k = kind(r["label"], p["pred"])
        if k == "severity" and args.skip_severity_only:
            continue
        if r["id"] in skip:
            continue
        rejected = p.get("raw")                          # the model's own text when it was unparsable
        if p["pred"] is not None:
            try:                                         # a wrong verdict (or one with a made-up excerpt), in the answer format
                rejected = answer_for(r, p["pred"])
            except (KeyError, TypeError, AttributeError):
                rejected = None                          # verdict too malformed to render; nothing to pair with
        if not rejected:
            continue
        kinds[k] += 1
        pairs.append({"id": r["id"], "kind": k, "prompt": prompt_messages(r),
                      "chosen": [{"role": "assistant", "content": answer_for(r, r["label"])}],
                      "rejected": [{"role": "assistant", "content": rejected}]})
    for path in args.extra:
        for x in load_jsonl(path):
            pairs.append({"id": x.get("source", {}).get("review_id", "extra"), "kind": "feedback",
                          "prompt": x["prompt"], "chosen": x["chosen"], "rejected": x["rejected"]})
            kinds["feedback"] += 1
    write_jsonl(args.out, pairs)
    print(json.dumps({"pairs": len(pairs), "by_kind": dict(kinds), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
