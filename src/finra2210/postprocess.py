"""Validate, deduplicate and assemble teacher output into a clean dataset.

  python -m finra2210.postprocess --raw data/in_domain.raw.jsonl --out data/in_domain.clean.jsonl
  python -m finra2210.postprocess --raw data/ood.raw.jsonl --out data/ood.clean.jsonl --exclude data/in_domain.clean.jsonl

Writes:
  <out>                 accepted samples, one JSON per line, with an `id`
  <out>.rejected.jsonl  samples that failed validation, with reasons
  <out>.stats.json      label / difficulty / category distribution
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .schema import validate_sample

_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9 ]+")


def _norm(text: str) -> str:
    return _WS.sub(" ", _NONWORD.sub(" ", text.lower())).strip()


def _shingles(text: str, k: int = 6) -> set[str]:
    w = _norm(text).split()
    return {" ".join(w[i : i + k]) for i in range(max(1, len(w) - k + 1))}


class Deduper:
    """Exact-normalized hash plus a cheap near-duplicate check on 6-word shingles.

    Near-dup detection indexes shingles -> sample ids and flags a candidate when
    any already-accepted sample shares > `threshold` of its shingles.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.exact: set[str] = set()
        self.index: dict[str, set[int]] = {}
        self.sizes: dict[int, int] = {}
        self.n = 0

    def add_existing(self, text: str) -> None:
        self._insert(text)

    def check_and_add(self, text: str) -> bool:
        """Return True if accepted (and indexed), False if duplicate."""
        h = hashlib.sha1(_norm(text).encode()).hexdigest()
        if h in self.exact:
            return False
        sh = _shingles(text)
        overlap: Counter[int] = Counter()
        for s in sh:
            for sid in self.index.get(s, ()):
                overlap[sid] += 1
        for sid, c in overlap.items():
            if c / max(1, min(len(sh), self.sizes[sid])) > self.threshold:
                return False
        self._insert(text, h, sh)
        return True

    def _insert(self, text: str, h: str | None = None, sh: set[str] | None = None) -> None:
        h = h or hashlib.sha1(_norm(text).encode()).hexdigest()
        sh = sh if sh is not None else _shingles(text)
        self.exact.add(h)
        sid = self.n
        self.n += 1
        self.sizes[sid] = len(sh)
        for s in sh:
            self.index.setdefault(s, set()).add(sid)


_INSTITUTIONAL_AUDIENCES = ("institutional investor who will forward", "plan sponsor", "registered person at another member")


def _fix_stale_audience(scen: dict | None) -> dict | None:
    """Early OOD chunks kept an institutional audience shift even after a retail-facing channel
    forced the message to retail_communication. Say so in the shift so per-axis eval is honest."""
    if not scen:
        return scen
    shifts = scen.get("ood_shifts") or {}
    aud = shifts.get("audience")
    if aud and any(k in aud for k in _INSTITUTIONAL_AUDIENCES) and scen.get("communication_type") != "institutional_communication":
        scen = {**scen, "ood_shifts": {**shifts, "audience": f"{aud} — but sent on a retail-facing channel, so it is a retail communication"}}
    return scen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="clean files whose texts must not be near-duplicated (e.g. train set when cleaning OOD)")
    ap.add_argument("--dup-threshold", type=float, default=0.5)
    ap.add_argument("--prefix", default=None, help="id prefix (default: derived from out filename)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or out.stem.split(".")[0]

    dd = Deduper(args.dup_threshold)
    for ex in args.exclude:
        for line in Path(ex).read_text().splitlines():
            dd.add_existing(json.loads(line)["communication"])

    accepted, rejected = [], []
    reasons: Counter[str] = Counter()
    request_errors: Counter[str] = Counter()
    for line in Path(args.raw).read_text().splitlines():
        rec = json.loads(line)
        if not rec.get("samples"):
            request_errors[rec.get("error", "unknown")] += 1
            continue
        by_id = {s["scenario_id"]: s for s in rec["scenarios"]}
        teacher = (rec.get("usage") or {}).get("teacher") or (rec.get("usage") or {}).get("model") or "claude-fable-5-1"
        for s in rec["samples"]:
            s.setdefault("teacher", teacher)  # per-sample tag wins (resumed chunks are mixed)
            scen = by_id.get(s.get("scenario_id"))
            probs = validate_sample(s, scen)
            if probs:
                for p in probs:
                    reasons[p.split(":")[0]] += 1
                rejected.append({**s, "_problems": probs})
                continue
            if not dd.check_and_add(s["communication"]):
                reasons["near-duplicate"] += 1
                rejected.append({**s, "_problems": ["near-duplicate"]})
                continue
            s["scenario"] = _fix_stale_audience(scen)
            accepted.append(s)

    for i, s in enumerate(accepted):
        s["id"] = f"{prefix}-{i:06d}"

    with out.open("w") as f:
        for s in accepted:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with out.with_suffix(".rejected.jsonl").open("w") as f:
        for s in rejected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats = {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "request_errors": dict(request_errors),
        "reject_reasons": dict(reasons.most_common()),
        "difficulty": dict(Counter(s["difficulty"] for s in accepted)),
        "compliant": dict(Counter(str(s["label"]["compliant"]) for s in accepted)),
        "communication_type": dict(Counter(s["label"]["communication_type"] for s in accepted)),
        "severity": dict(Counter(s["label"]["severity"] for s in accepted)),
        "channel": dict(Counter(s["channel"] for s in accepted).most_common()),
        "violation_categories": dict(Counter(v["category"] for s in accepted for v in s["label"]["violations"]).most_common()),
        "violations_per_sample": dict(Counter(len(s["label"]["violations"]) for s in accepted)),
        "avg_chars": round(sum(len(s["communication"]) for s in accepted) / max(1, len(accepted))),
        "teacher": dict(Counter(s["teacher"] for s in accepted)),
    }
    if accepted and accepted[0].get("scenario", {}).get("distribution") == "ood":
        stats["ood_axes"] = dict(Counter(ax for s in accepted for ax in s["scenario"].get("ood_shifts", {})).most_common())
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
