"""Release gate: may a candidate model replace the current one?

  # candidate vs current release, on the fixed test and OOD sets
  .venv/bin/python retraining/06_release_gate/gate.py \
      --candidate runs/phi4mini-2210-r2/eval_{split}.json --baseline results/metrics/fine_tuned_{split}.json

  # the sample: the fine-tuned v0.1.0 model as candidate against the zero-shot base (passes), and the reverse (fails)
  .venv/bin/python retraining/06_release_gate/gate.py \
      --candidate results/metrics/fine_tuned_{split}.json --baseline results/metrics/base_zero_shot_{split}.json
  .venv/bin/python retraining/06_release_gate/gate.py \
      --candidate results/metrics/base_zero_shot_{split}.json --baseline results/metrics/fine_tuned_{split}.json

{split} in the paths is replaced by each split in gates.json ("test", "ood"). Both models must be
scored on the same split files. Rules (gates.json):
  min           absolute floor the candidate must reach
  max_drop      largest allowed fall versus the baseline (absolute, on a 0-1 scale)
Checked overall, on the chat slice (by_channel_group.chat), for every broad group with at least
`min_support` gold items, and for every difficulty. Prints a table and writes --report (Markdown).
Exit code 0 = PASS, 1 = FAIL, so it can run in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def metric(report: dict, path: str):
    """'violation_category_micro.f1' -> report['violation_category_micro']['f1']; None if absent."""
    cur = report
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check(rows: list, scope: str, cand: dict, base: dict, rules: dict) -> None:
    for name, rule in rules.items():
        c, b = metric(cand, name), metric(base, name)
        if c is None:
            continue
        reasons = []
        if "min" in rule and c < rule["min"]:
            reasons.append(f"below floor {rule['min']:.2f}")
        if "max_drop" in rule and b is not None and b - c > rule["max_drop"] + 1e-9:
            reasons.append(f"dropped {b - c:.3f} > {rule['max_drop']:.3f}")
        rows.append({"scope": scope, "metric": name, "candidate": c, "baseline": b, "ok": not reasons, "why": "; ".join(reasons)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="eval report path with {split}, e.g. runs/new/eval_{split}.json")
    ap.add_argument("--baseline", required=True, help="eval report path with {split} for the model in production")
    ap.add_argument("--gates", default=str(HERE / "gates.json"))
    ap.add_argument("--report", default=None, help="write the table as Markdown here")
    args = ap.parse_args()
    gates = json.loads(Path(args.gates).read_text())

    rows: list[dict] = []
    for split in gates["splits"]:
        cand = json.loads(Path(args.candidate.format(split=split)).read_text())
        base = json.loads(Path(args.baseline.format(split=split)).read_text())
        if cand.get("data") != base.get("data"):
            sys.exit(f"{split}: candidate scored on {cand.get('data')}, baseline on {base.get('data')}; use the same split file")
        check(rows, f"{split} overall", cand["overall"], base["overall"], gates["overall"])
        if "chat" in cand.get("by_channel_group", {}):
            check(rows, f"{split} chat", cand["by_channel_group"]["chat"], base.get("by_channel_group", {}).get("chat", {}), gates["chat"])
        for g, bg in cand["overall"].get("broad_groups", {}).items():
            if bg["support"] >= gates["broad_group"]["min_support"]:
                check(rows, f"{split} group {g}", bg, base["overall"].get("broad_groups", {}).get(g, {}), gates["broad_group"]["rules"])
        for d, rep in cand.get("by_difficulty", {}).items():
            check(rows, f"{split} {d}", rep, base.get("by_difficulty", {}).get(d, {}), gates["difficulty"])

    ok = all(r["ok"] for r in rows)
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    lines = [f"# Release gate: {'PASS' if ok else 'FAIL'}", "", f"Candidate `{args.candidate}` vs baseline `{args.baseline}`.", "",
             "| Scope | Metric | Candidate | Baseline | Result |", "|---|---|---|---|---|"]
    lines += [f"| {r['scope']} | {r['metric']} | {fmt(r['candidate'])} | {fmt(r['baseline'])} | {'ok' if r['ok'] else 'FAIL: ' + r['why']} |" for r in rows]
    failed = [r for r in rows if not r["ok"]]
    print(f"{'PASS' if ok else 'FAIL'}: {len(rows) - len(failed)} of {len(rows)} checks passed")
    for r in failed[:15]:
        print(f"  {r['scope']:32} {r['metric']:34} {fmt(r['candidate'])} vs {fmt(r['baseline'])}: {r['why']}")
    if len(failed) > 15:
        print(f"  ... and {len(failed) - 15} more")
    if args.report:
        Path(args.report).write_text("\n".join(lines) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
