"""Build mlx-lm LoRA data (train/valid.jsonl, messages only) from a chat-format split.

  python -m finra2210.mlx_data --train data/splits/train.jsonl --out data/mlx --val-frac 0.05

The validation set is a stratified holdout of train.jsonl (same strata as split.py), so
test.jsonl stays untouched for the final evaluation. mlx-lm's --mask-prompt trains only the
last assistant message, so each multi-turn conversation is expanded into one example per
assistant turn (prefix ending at that turn); otherwise the first review turn of a 5-turn
conversation would never be trained.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .split import _stratum


def expand(messages: list[dict]) -> list[list[dict]]:
    """One prefix per assistant turn, each ending at that turn (the turn the loss is taken on)."""
    return [messages[: i + 1] for i, m in enumerate(messages) if m["role"] == "assistant"]


def holdout(rows: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified (split.py strata) validation holdout of train rows; shared with train_phi4.py."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stratum[_stratum(r)].append(r)
    train, valid = [], []
    for _, items in sorted(by_stratum.items()):
        rng.shuffle(items)
        k = round(len(items) * val_frac) if len(items) >= 4 else 0
        valid.extend(items[:k])
        train.extend(items[k:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def _write(rows: list[dict], path: Path) -> int:
    n = 0
    with path.open("w") as f:
        for r in rows:
            for msgs in expand(r["messages"]):
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(args.train).read_text().splitlines()]

    train, valid = holdout(rows, args.val_frac, args.seed)

    n_train = _write(train, out / "train.jsonl")
    n_valid = _write(valid, out / "valid.jsonl")
    (out / "valid.ids").write_text("".join(r["id"] + "\n" for r in valid))
    manifest = {
        "source": args.train,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "train_conversations": len(train),
        "valid_conversations": len(valid),
        "train_examples": n_train,
        "valid_examples": n_valid,
        "valid_difficulty": dict(Counter(r["difficulty"] for r in valid)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
