"""Stratified train/test split and export to Phi-4-mini chat format.

  python -m finra2210.split --clean data/in_domain.clean.jsonl --out data/splits --test-frac 0.10
  python -m finra2210.split --clean data/ood.clean.jsonl --out data/splits --name ood   # no split, eval only

Incremental (after appending new samples to the clean file): keep every prior assignment and
stratify only the new ids, so a model trained on the old train.jsonl stays clean on the new test.jsonl:
  python -m finra2210.split --clean data/clean/in_domain.clean.jsonl --out data/splits_new \
      --keep-test old/test.jsonl --keep-train old/train.jsonl
--keep-test/--keep-train take a split JSONL (records with "id") or a plain id list (one per line).

Outputs (chat-format JSONL, one {"messages": [...]} per line, plus the full record):
  data/splits/train.jsonl, test.jsonl   (or ood.jsonl)
  data/splits/<name>.manifest.json      counts per stratum
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .prompts import to_chat_messages


def _stratum(s: dict) -> str:
    cats = sorted(v["category"] for v in s["label"]["violations"])
    primary = cats[0] if cats else "compliant"
    return f"{s['difficulty']}|{s['label']['communication_type']}|{primary}"


def _load_ids(path: str) -> set[str]:
    """Ids from a split JSONL (records with "id") or a plain one-id-per-line list."""
    ids = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ids.add(json.loads(line)["id"] if line.startswith("{") else line)
    return ids


def _export(samples: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for s in samples:
            rec = {
                "id": s["id"],
                "difficulty": s["difficulty"],
                "channel": s["channel"],
                "metadata": s["metadata"],
                "communication": s["communication"],
                "label": s["label"],
                "ood_shifts": s.get("scenario", {}).get("ood_shifts"),
                "teacher": s.get("teacher"),
                "messages": to_chat_messages(s),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None, help="if set, export everything as <name>.jsonl without splitting")
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-test", default=None,
                    help="prior test.jsonl or id list: these ids stay in test; only unseen ids are stratified")
    ap.add_argument("--keep-train", default=None,
                    help="prior train.jsonl or id list: these ids stay in train (required with --keep-test)")
    args = ap.parse_args()
    if bool(args.keep_test) != bool(args.keep_train):
        ap.error("--keep-test and --keep-train must be given together")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(l) for l in Path(args.clean).read_text().splitlines()]

    if args.name:
        _export(samples, out / f"{args.name}.jsonl")
        (out / f"{args.name}.manifest.json").write_text(json.dumps({
            "n": len(samples),
            "difficulty": dict(Counter(s["difficulty"] for s in samples)),
            "compliant": dict(Counter(str(s["label"]["compliant"]) for s in samples)),
        }, indent=2))
        print(f"wrote {len(samples)} -> {out / (args.name + '.jsonl')}")
        return

    rng = random.Random(args.seed)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_stratum[_stratum(s)].append(s)

    keep_test = _load_ids(args.keep_test) if args.keep_test else set()
    keep_train = _load_ids(args.keep_train) if args.keep_train else set()
    if keep_test & keep_train:
        raise SystemExit(f"{len(keep_test & keep_train)} ids are in both --keep-test and --keep-train")
    present = {s["id"] for s in samples}
    missing = (keep_test | keep_train) - present
    if missing:
        print(f"warning: {len(missing)} kept ids are not in {args.clean} (dropped)")

    train, test = [], []
    n_new_train = n_new_test = 0
    for stratum, items in sorted(by_stratum.items()):
        old_test = [s for s in items if s["id"] in keep_test]
        old_train = [s for s in items if s["id"] in keep_train]
        new = [s for s in items if s["id"] not in keep_test and s["id"] not in keep_train]
        rng.shuffle(new)
        k = round(len(items) * args.test_frac)
        # tiny strata: keep at least one in train; put one in test only if >= 4 items
        if len(items) < 4:
            k = 0
        # prior assignments are fixed; new items top the stratum's test share up to k
        k = max(0, min(len(new), k - len(old_test)))
        test.extend(old_test + new[:k])
        train.extend(old_train + new[k:])
        n_new_test += k
        n_new_train += len(new) - k
    rng.shuffle(train)
    rng.shuffle(test)

    _export(train, out / "train.jsonl")
    _export(test, out / "test.jsonl")
    manifest = {
        "train": len(train),
        "test": len(test),
        "test_frac": args.test_frac,
        "seed": args.seed,
        "train_difficulty": dict(Counter(s["difficulty"] for s in train)),
        "test_difficulty": dict(Counter(s["difficulty"] for s in test)),
        "train_compliant": dict(Counter(str(s["label"]["compliant"]) for s in train)),
        "test_compliant": dict(Counter(str(s["label"]["compliant"]) for s in test)),
        "strata": len(by_stratum),
    }
    if args.keep_test:
        manifest.update({
            "keep_test": args.keep_test,
            "keep_train": args.keep_train,
            "kept_test": len(test) - n_new_test,
            "kept_train": len(train) - n_new_train,
            "new_test": n_new_test,
            "new_train": n_new_train,
        })
    (out / "in_domain.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
