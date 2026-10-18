"""Pick the items where the model and the gold label disagree, and write them as audit chunks.

  # predictions of the current model on the TRAIN split (never audit-to-train from test or OOD)
  PYTHONPATH=src .venv-mlx/bin/python -m finra2210.evaluate --backend mlx --model runs/phi4mini-2210-mlx \
      --data data/splits/train.jsonl --out runs/phi4mini-2210-mlx/eval_train.json
  .venv/bin/python retraining/02_audit/select_disagreements.py --gold data/splits/train.jsonl \
      --preds runs/phi4mini-2210-mlx/eval_train.preds.jsonl --out retraining/02_audit/tasks --limit 400

  # the sample
  .venv/bin/python retraining/02_audit/select_disagreements.py --gold retraining/02_audit/sample/train.sample.jsonl \
      --preds retraining/02_audit/sample/preds.sample.jsonl --out retraining/02_audit/sample/tasks --per-chunk 2

Each chunk (tasks/chunk_0000.json, ...) is one auditor's assignment: AUDITOR_INSTRUCTIONS.md says how
to adjudicate it. Items are ordered by how much the disagreement matters: a different compliant /
not-compliant call first, then a different communication type or category set (categories named
in --focus first), then severity-only differences.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import categories, load_jsonl  # noqa: E402

DEFAULT_FOCUS = ["product_specific_omission", "not_fair_and_balanced", "misleading_in_context",
                 "false_or_misleading_statement", "member_name_missing", "buried_disclosure"]


def disagreement(gold: dict, pred: dict | None) -> list[str]:
    if pred is None:
        return ["unparsable model answer"]
    out = []
    if bool(gold["compliant"]) != bool(pred.get("compliant")):
        out.append("compliant")
    if gold["communication_type"] != pred.get("communication_type"):
        out.append("communication_type")
    g, p = categories(gold), categories(pred)
    if g != p:
        out.append(f"categories: missed {sorted(g - p)}, extra {sorted(p - g)}")
    if gold["severity"] != pred.get("severity"):
        out.append("severity")
    return out


def priority(gold: dict, pred: dict | None, focus: set[str]) -> int:
    if pred is None:
        return 0  # format failures are a training problem, not a labeling question
    score = 0
    if bool(gold["compliant"]) != bool(pred.get("compliant")):
        score += 8
    if gold["communication_type"] != pred.get("communication_type"):
        score += 4
    diff = categories(gold) ^ categories(pred)
    if diff:
        score += 3 + (2 if diff & focus else 0)
    if gold["severity"] != pred.get("severity"):
        score += 1
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="split file the predictions were made on (train.jsonl for training fixes)")
    ap.add_argument("--preds", required=True, help="<eval out>.preds.jsonl from finra2210.evaluate")
    ap.add_argument("--out", required=True, help="directory for chunk_XXXX.json files")
    ap.add_argument("--split-name", default=None, help="train / test / ood (default: from the --gold file name)")
    ap.add_argument("--per-chunk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="audit at most this many items (highest priority first)")
    ap.add_argument("--focus", nargs="*", default=DEFAULT_FOCUS, help="categories whose confusions are audited first")
    args = ap.parse_args()

    split = args.split_name or Path(args.gold).name.split(".")[0]
    gold = {r["id"]: r for r in load_jsonl(args.gold)}
    focus = set(args.focus)
    items = []
    for p in load_jsonl(args.preds):
        r = gold.get(p["id"])
        if r is None:
            continue
        why = disagreement(r["label"], p["pred"])
        score = priority(r["label"], p["pred"], focus)
        if why and score > 0:
            items.append((score, r["id"], {
                "id": r["id"], "split": split, "channel": r["channel"], "metadata": r["metadata"],
                "communication": r["communication"], "gold_label": r["label"], "model_verdict": p["pred"],
                "disagreement": why}))
    items.sort(key=lambda t: (-t[0], t[1]))
    if args.limit:
        items = items[: args.limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunks = [items[i : i + args.per_chunk] for i in range(0, len(items), args.per_chunk)]
    for n, chunk in enumerate(chunks):
        (out / f"chunk_{n:04d}.json").write_text(json.dumps({"split": split, "items": [it for _, _, it in chunk]}, ensure_ascii=False, indent=1))
    kinds = Counter(w.split(":")[0] for _, _, it in items for w in it["disagreement"])
    index = {"split": split, "gold": args.gold, "preds": args.preds, "items": len(items), "chunks": len(chunks),
             "disagreement_kinds": dict(kinds)}
    (out / "INDEX.json").write_text(json.dumps(index, indent=2))
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
