"""List the dataset items an approved rule change makes stale, and optionally rewrite a cite string.

  .venv/bin/python retraining/00_rule_watch/mark_stale.py --approved retraining/00_rule_watch/sample/approved_change.sample.json \
      --out retraining/00_rule_watch/sample/out

  # only the cite string changed: write relabeled copies of the clean files (never in place)
  .venv/bin/python retraining/00_rule_watch/mark_stale.py --approved approved_change.json --out retraining/00_rule_watch/out --relabel

The approval file (approved_change.json, drafted by make_memo.py and edited by a human) names:
  categories    items with a violation in any of these categories are stale
  cites         items with a violation whose rule cite starts with any of these are stale
  topics        compliant items whose scenario near_miss_topic mentions any of these words are stale
  relabel_cites {"old cite": "new cite"}: with --relabel, rewrite violations[].rule in copies of the clean files
Writes <out>/stale_ids.jsonl (id, split, reasons), <out>/stale_summary.json and prints a count table.
Stale items are regenerated through the agent loop (data/chunks_*/AGENT_INSTRUCTIONS.md) or, when only
the cite moved, relabeled; then snapshot, postprocess and the incremental split run as usual. Items in
test or OOD that change become a new eval-set version (test_v2); they never enter training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retraining"))
from common import load_jsonl, write_jsonl  # noqa: E402

DEFAULT_CLEAN = ["data/clean/in_domain.clean.jsonl", "data/clean/ood.clean.jsonl"]


def reasons_for(item: dict, approved: dict) -> list[str]:
    cats = set(approved.get("categories", []))
    cites = [c for c in approved.get("cites", []) if c]
    topics = [t.lower() for t in approved.get("topics", []) if t]
    label = item["label"]
    scen = item.get("scenario") or {}
    out = []
    for v in label.get("violations", []):
        if v.get("category") in cats:
            out.append(f"violation category {v['category']}")
        if any(str(v.get("rule", "")).startswith(c) for c in cites):
            out.append(f"violation cite {v.get('rule')}")
    for c in scen.get("violation_categories", []) or []:
        if c in cats and f"violation category {c}" not in out:
            out.append(f"scenario category {c}")
    nm = (scen.get("near_miss_topic") or "").lower()
    if label.get("compliant") and nm and any(t in nm for t in topics):
        out.append("compliant near-miss on affected topic")
    return list(dict.fromkeys(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approved", required=True, help="approved_change.json")
    ap.add_argument("--clean", nargs="*", default=DEFAULT_CLEAN, help="clean files to scan")
    ap.add_argument("--out", default="retraining/00_rule_watch/out")
    ap.add_argument("--relabel", action="store_true", help="apply relabel_cites and write <out>/relabeled/<file>")
    args = ap.parse_args()

    approved = json.loads(Path(args.approved).read_text())
    if not approved.get("approved_by"):
        print("note: approved_by is empty; treating the file as a draft (output is still written)", file=sys.stderr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stale, by_reason, by_split, relabeled = [], Counter(), Counter(), Counter()
    relabel = approved.get("relabel_cites") or {}
    for path in args.clean:
        p = Path(path)
        split = p.name.split(".")[0]
        rows = load_jsonl(p)
        total = len(rows)
        for r in rows:
            why = reasons_for(r, approved)
            if why:
                stale.append({"id": r["id"], "split": split, "reasons": why})
                by_split[split] += 1
                for w in why:
                    by_reason[w.split(" ")[0] + " " + w.split(" ")[1] if w.startswith(("violation", "scenario")) else w] += 1
            if args.relabel and relabel:
                for v in r["label"].get("violations", []):
                    if v.get("rule") in relabel:
                        v["rule"] = relabel[v["rule"]]
                        relabeled[split] += 1
        if args.relabel and relabel:
            dest = out / "relabeled" / p.name
            write_jsonl(dest, rows)
            print(f"relabeled {relabeled[split]} cite(s) in {split} -> {dest}  (re-export the chat splits with finra2210.split; the messages embed the cite)")
        by_split[f"{split} total"] = total

    write_jsonl(out / "stale_ids.jsonl", stale)
    summary = {"approved": args.approved, "categories": approved.get("categories", []), "cites": approved.get("cites", []),
               "topics": approved.get("topics", []), "stale": len(stale), "by_split": dict(by_split), "by_reason": dict(by_reason),
               "relabeled": dict(relabeled) if args.relabel else None}
    (out / "stale_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{len(stale)} stale item(s)")
    for k, v in sorted(by_split.items()):
        print(f"  {k:28} {v:>6}")
    for k, v in by_reason.most_common():
        print(f"  reason: {k:40} {v:>6}")
    print(f"wrote {out / 'stale_ids.jsonl'} and {out / 'stale_summary.json'}")


if __name__ == "__main__":
    main()
