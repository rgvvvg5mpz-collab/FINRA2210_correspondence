"""Write scenario chunks + the examiner brief for Claude Code subagents to realize.

  python -m finra2210.chunks --n 11500 --per-chunk 40 --out data/chunks/in_domain
  python -m finra2210.chunks --n 5800  --per-chunk 40 --ood --seed 99 --out data/chunks/ood

Each chunk_XXXX.json holds a list of scenario specs. BRIEF.md (the same grounding
the API teacher gets) is written next to the chunks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prompts import TEACHER_SYSTEM
from .scenarios import make_scenarios


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--per-chunk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2210)
    ap.add_argument("--ood", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = make_scenarios(args.n, args.seed, args.ood)
    n_chunks = 0
    for i in range(0, len(scenarios), args.per_chunk):
        (out / f"chunk_{i // args.per_chunk:04d}.json").write_text(json.dumps(scenarios[i : i + args.per_chunk], indent=1))
        n_chunks += 1
    (out.parent / "BRIEF.md").write_text(TEACHER_SYSTEM)
    print(f"{len(scenarios)} scenarios -> {n_chunks} chunks in {out}; brief at {out.parent / 'BRIEF.md'}")


if __name__ == "__main__":
    main()
