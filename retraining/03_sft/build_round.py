"""Build the data for the next supervised (SFT) round: apply label fixes, add new labeled samples,
de-duplicate, and re-split without moving any earlier item between train and test.

  .venv/bin/python retraining/03_sft/build_round.py --clean data/clean/in_domain.clean.jsonl \
      --label-fixes retraining/02_audit/out/train_label_fixes.jsonl \
      --add retraining/01_feedback/out/corrections.jsonl data/clean/chat_topup.clean.jsonl \
      --keep-test data/splits/test.jsonl --keep-train data/splits/train.jsonl --out data/rounds/r2

  # the sample
  .venv/bin/python retraining/03_sft/build_round.py --clean retraining/03_sft/sample/in_domain.clean.sample.jsonl \
      --label-fixes retraining/02_audit/sample/out/train_label_fixes.jsonl \
      --add retraining/01_feedback/sample/out/corrections.jsonl \
      --keep-test retraining/03_sft/sample/test.ids --keep-train retraining/03_sft/sample/train.ids \
      --out retraining/03_sft/sample/out

Writes --out/in_domain.clean.jsonl, --out/round_manifest.json and --out/splits/{train,test}.jsonl
(via finra2210.split with --keep-test/--keep-train). Then train from the BASE model on
--out/splits/train.jsonl with the usual pipeline (see README.md).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, check_label, load_jsonl, write_jsonl  # noqa: E402
from finra2210.postprocess import Deduper  # noqa: E402

CLEAN_KEYS = ("id", "difficulty", "channel", "metadata", "communication", "label")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="current in-domain clean file")
    ap.add_argument("--label-fixes", nargs="*", default=[], help="train_label_fixes.jsonl from 02_audit")
    ap.add_argument("--add", nargs="*", default=[], help="new labeled samples (clean format), e.g. 01_feedback corrections")
    ap.add_argument("--keep-test", required=True, help="previous test split (JSONL or id list): these ids stay in test")
    ap.add_argument("--keep-train", required=True, help="previous train split (JSONL or id list): these ids stay in train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-prefix", default="in_domain-r2", help="ids for added samples: <prefix>-000000, ...")
    ap.add_argument("--dup-threshold", type=float, default=0.5)
    args = ap.parse_args()

    base = load_jsonl(args.clean)
    by_id = {s["id"]: s for s in base}
    test_ids = {l if not l.startswith("{") else json.loads(l)["id"] for l in Path(args.keep_test).read_text().split("\n") if l.strip()}
    stats: Counter = Counter()
    problems: list[dict] = []

    # 1. label fixes: train items only (test / OOD fixes change the eval set, never the training data)
    for path in args.label_fixes:
        for fix in load_jsonl(path):
            if fix.get("split") != "train" or fix["id"] in test_ids:
                raise SystemExit(f"{path}: {fix['id']} is not a train item; eval-set fixes are not applied here")
            s = by_id.get(fix["id"])
            if s is None:
                problems.append({"id": fix["id"], "problem": "label fix for an id not in --clean"})
                continue
            errs = check_label(s, fix["label"])
            if errs:
                problems.append({"id": fix["id"], "problem": "; ".join(errs)})
                continue
            s.setdefault("relabels", []).append({"previous_label": s["label"], "decision": fix.get("decision"),
                                                 "reason": fix.get("reason"), "source": str(path)})
            s["label"] = fix["label"]
            stats["label_fixes"] += 1

    # 2. new samples: valid labels only, no near-duplicates of anything already in the set
    dd = Deduper(args.dup_threshold)
    for s in base:
        dd.add_existing(s["communication"])
    added = []
    for path in args.add:
        for s in load_jsonl(path):
            missing = [k for k in CLEAN_KEYS if k not in s]
            errs = [f"missing keys {missing}"] if missing else check_label(s, s["label"])
            if errs:
                problems.append({"id": s.get("id"), "source": str(path), "problem": "; ".join(errs)})
                continue
            if not dd.check_and_add(s["communication"]):
                problems.append({"id": s.get("id"), "source": str(path), "problem": "near-duplicate of an existing sample"})
                stats["duplicates_dropped"] += 1
                continue
            new_id = f"{args.id_prefix}-{len(added):06d}"
            added.append({**s, "id": new_id, "source_id": s["id"], "added_from": str(path)})
    stats["added"] = len(added)

    out = Path(args.out)
    write_jsonl(out / "in_domain.clean.jsonl", base + added)
    write_jsonl(out / "problems.jsonl", problems)
    manifest = {"base": args.clean, "base_items": len(base), **stats, "problems": len(problems),
                "total": len(base) + len(added), "label_fix_files": args.label_fixes, "added_files": args.add}
    (out / "round_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    for p in problems:
        print(f"  skipped {p.get('id')}: {p['problem']}")

    # 3. incremental split: earlier items keep their side; only added ids are stratified
    sys.stdout.flush()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-m", "finra2210.split", "--clean", str(out / "in_domain.clean.jsonl"),
                    "--out", str(out / "splits"), "--keep-test", args.keep_test, "--keep-train", args.keep_train],
                   check=True, env=env)


if __name__ == "__main__":
    main()
