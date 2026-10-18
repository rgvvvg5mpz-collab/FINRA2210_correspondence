"""Write a timestamped, checksummed snapshot of every validated sample on disk.

  python -m finra2210.snapshot            # -> data/snapshots/<UTC timestamp>/

Copies the validated raw records (pilot data/agent_out, round-1 data/agent_out_r1, targeted
top-up data/agent_out_topup) into one in_domain.raw.jsonl and one ood.raw.jsonl, and writes
MANIFEST.json with chunk/sample counts, teacher mix and sha256 per file. Snapshots are append-only: older ones are never touched.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SOURCES = [Path("data/agent_out"), Path("data/agent_out_r1"), Path("data/agent_out_topup"), Path("data/agent_out_chat")]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("data/snapshots") / stamp
    out.mkdir(parents=True, exist_ok=False)
    manifest: dict = {"created_utc": stamp, "splits": {}}
    for split in ("in_domain", "ood"):
        dest = out / f"{split}.raw.jsonl"
        chunks = samples = 0
        teachers: Counter[str] = Counter()
        sources: Counter[str] = Counter()
        with dest.open("w") as f:
            for src in SOURCES:
                for raw in sorted((src / split).glob("chunk_*.raw.jsonl")):
                    for line in raw.read_text().splitlines():
                        rec = json.loads(line)
                        f.write(line + "\n")
                        chunks += 1
                        sources[str(src)] += 1
                        teacher = (rec.get("usage") or {}).get("teacher", "unknown")
                        for s in rec["samples"]:
                            samples += 1
                            teachers[s.get("teacher", teacher)] += 1
        manifest["splits"][split] = {
            "file": dest.name,
            "records": chunks,
            "samples": samples,
            "records_by_source": dict(sources),
            "samples_by_teacher": dict(teachers),
            "sha256": _sha256(dest),
        }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
