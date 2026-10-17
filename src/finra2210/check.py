"""Check one agent-written chunk output against its scenarios.

  python -m finra2210.check data/chunks/in_domain/chunk_0003.json data/agent_out/in_domain/chunk_0003.json

Output file format: {"samples": [ ...sample objects matching schema.SAMPLE_SCHEMA... ]}.
Prints one line per problem and a summary; exit code 1 if anything is wrong.
With --emit <raw.jsonl>, appends a raw record (scenarios + samples) for postprocess.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .schema import SAMPLE_SCHEMA, validate_sample

_REQ_TOP = SAMPLE_SCHEMA["required"]
_REQ_META = SAMPLE_SCHEMA["properties"]["metadata"]["required"]
_REQ_LABEL = SAMPLE_SCHEMA["properties"]["label"]["required"]
_REQ_VIOL = SAMPLE_SCHEMA["properties"]["label"]["properties"]["violations"]["items"]["required"]


def _shape_problems(s: dict) -> list[str]:
    probs = [f"missing top-level key {k}" for k in _REQ_TOP if k not in s]
    if isinstance(s.get("metadata"), dict):
        probs += [f"missing metadata.{k}" for k in _REQ_META if k not in s["metadata"]]
        if not isinstance(s["metadata"].get("recipient_count_30d"), int):
            probs.append("metadata.recipient_count_30d must be an integer")
    else:
        probs.append("metadata must be an object")
    if isinstance(s.get("label"), dict):
        probs += [f"missing label.{k}" for k in _REQ_LABEL if k not in s["label"]]
        for i, v in enumerate(s["label"].get("violations", [])):
            if isinstance(v, dict):
                probs += [f"violations[{i}] missing {k}" for k in _REQ_VIOL if k not in v]
            else:
                probs.append(f"violations[{i}] must be an object")
        if not isinstance(s["label"].get("compliant"), bool):
            probs.append("label.compliant must be true/false")
    else:
        probs.append("label must be an object")
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk")
    ap.add_argument("output")
    ap.add_argument("--emit", default=None)
    ap.add_argument("--teacher", default="unknown", help="model that wrote the samples, recorded in the raw record")
    args = ap.parse_args()

    scenarios = json.loads(Path(args.chunk).read_text())
    by_id = {s["scenario_id"]: s for s in scenarios}
    try:
        samples = json.loads(Path(args.output).read_text())["samples"]
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"FAIL: cannot read output: {e}")
        sys.exit(1)

    seen = set()
    n_bad = 0
    for s in samples:
        sid = s.get("scenario_id", "?")
        probs = _shape_problems(s)
        if not probs:
            scen = by_id.get(sid)
            if scen is None:
                probs = [f"unknown scenario_id {sid}"]
            else:
                probs = validate_sample(s, scen)
        if sid in seen:
            probs.append("duplicate scenario_id")
        seen.add(sid)
        if probs:
            n_bad += 1
            for p in probs:
                print(f"{sid}: {p}")
    missing = sorted(set(by_id) - seen)
    for m in missing:
        print(f"{m}: MISSING (no sample written)")
    print(f"\n{len(samples)} samples, {len(samples) - n_bad} ok, {n_bad} with problems, {len(missing)} missing")

    if n_bad or missing:
        sys.exit(1)
    if args.emit:
        with Path(args.emit).open("a") as f:
            f.write(json.dumps({"scenarios": scenarios, "samples": samples, "error": "",
                                "usage": {"source": "claude-code-agent", "teacher": args.teacher}}) + "\n")
        print(f"emitted to {args.emit}")


if __name__ == "__main__":
    main()
