"""Report which chunks have validated output, and assemble raw.jsonl for postprocess.

  python -m finra2210.progress data/chunks/in_domain data/agent_out/in_domain
  python -m finra2210.progress data/chunks/in_domain data/agent_out/in_domain --assemble data/in_domain.raw.jsonl
  python -m finra2210.progress data/chunks/in_domain data/agent_out/in_domain --next 10   # print next pending chunk ids
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("outputs")
    ap.add_argument("--assemble", default=None)
    ap.add_argument("--next", type=int, default=0)
    args = ap.parse_args()

    chunk_ids = sorted(p.stem for p in Path(args.chunks).glob("chunk_*.json"))
    out = Path(args.outputs)
    done = [c for c in chunk_ids if (out / f"{c}.raw.jsonl").exists()]
    drafted = [c for c in chunk_ids if (out / f"{c}.json").exists() and not (out / f"{c}.raw.jsonl").exists()]
    pending = [c for c in chunk_ids if c not in done and c not in drafted]
    n_samples = 0
    for c in done:
        for line in (out / f"{c}.raw.jsonl").read_text().splitlines():
            n_samples += len(json.loads(line)["samples"])
    print(f"chunks: {len(chunk_ids)} total, {len(done)} validated ({n_samples} samples), {len(drafted)} drafted-unvalidated, {len(pending)} pending")
    if drafted:
        print("drafted but not validated:", " ".join(drafted))
    if args.next:
        print("next:", " ".join(pending[: args.next]))
    if args.assemble:
        with Path(args.assemble).open("w") as f:
            for c in done:
                f.write((out / f"{c}.raw.jsonl").read_text())
        print(f"assembled {len(done)} chunks -> {args.assemble}")


if __name__ == "__main__":
    main()
