"""Shared eval analysis: confusion matrices, per-category and per-broad-group metrics.

Used by evaluation/live_confusion.py (live dashboard) and evaluation/make_results.py (plots and
tables in results/). Overall metrics come from finra2210.evaluate.Scorer, so they match
eval_*.json exactly.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .evaluate import Scorer, as_compliant, channel_group, group_of, pred_violations
from .taxonomy import BROAD_GROUPS

CTYPES = ["correspondence", "retail_communication", "institutional_communication"]
SEVS = ["none", "low", "medium", "high"]


def load_gold(path: Path) -> tuple[dict, list[str]]:
    """id -> {label, axes, difficulty, channel} for a split file, plus the ids in file order."""
    gold, order = {}, []
    with Path(path).open() as f:
        for line in f:
            r = json.loads(line)
            gold[r["id"]] = {"label": r["label"], "axes": list((r.get("ood_shifts") or {}).keys()),
                             "difficulty": r["difficulty"], "channel": r.get("channel")}
            order.append(r["id"])
    return gold, order


def read_preds(path: Path) -> list[dict]:
    """Rows of <eval out>.preds.jsonl; stops at a half-written last line (safe while an eval runs)."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a half-written last line
    return out


def matrix(rows, cols):
    return {"rows": rows, "cols": cols, "cells": [[0] * len(cols) for _ in rows]}


def analyze(preds: list[dict], gold: dict, step: int | None = None) -> dict:
    """Confusion matrices, per-category / per-broad-group metrics, category swaps and accuracy by
    OOD axis, difficulty and channel group. With `step`, only the first multiple of `step` rows is
    scored (the live dashboard uses 10). Unparsable predictions get their own matrix column."""
    preds = [p for p in preds if p["id"] in gold]
    if step:
        preds = preds[: len(preds) // step * step]
    comp = matrix(["violation", "compliant"], ["violation", "compliant", "unparsed"])
    ctype = matrix(CTYPES, CTYPES + ["other", "unparsed"])
    sev = matrix(SEVS, SEVS + ["other", "unparsed"])
    cat = defaultdict(lambda: Counter(gold=0, tp=0, fp=0, fn=0))
    swaps: Counter = Counter()
    axes = defaultdict(lambda: Counter(n=0, ok=0))
    diffs = defaultdict(lambda: Counter(n=0, ok=0))
    groups = defaultdict(lambda: Counter(n=0, ok=0))
    correct = 0
    scorer = Scorer()
    for p in preds:
        g = gold[p["id"]]
        lab, pred = g["label"], p["pred"]
        scorer.add(lab, pred)
        gi = 0 if not lab["compliant"] else 1
        if pred is None:
            comp["cells"][gi][2] += 1
            ctype["cells"][CTYPES.index(lab["communication_type"])][-1] += 1
            sev["cells"][SEVS.index(lab["severity"])][-1] += 1
            ok = False
            p_cats: set = set()
        else:
            p_viol = not as_compliant(pred.get("compliant"))
            comp["cells"][gi][0 if p_viol else 1] += 1
            ok = p_viol == (not lab["compliant"])
            pc = pred.get("communication_type")
            ctype["cells"][CTYPES.index(lab["communication_type"])][CTYPES.index(pc) if pc in CTYPES else 3] += 1
            ps = pred.get("severity")
            sev["cells"][SEVS.index(lab["severity"])][SEVS.index(ps) if ps in SEVS else 4] += 1
            p_cats = {v.get("category") for v in pred_violations(pred)}
        g_cats = {v["category"] for v in lab["violations"]}
        for c in g_cats:
            cat[c]["gold"] += 1
            cat[c]["tp" if c in p_cats else "fn"] += 1
        for c in p_cats - g_cats:
            cat[c]["fp"] += 1
        for m in g_cats - p_cats:
            for x in p_cats - g_cats:
                swaps[(m, x)] += 1
        correct += ok
        for ax in g["axes"]:
            axes[ax]["n"] += 1
            axes[ax]["ok"] += ok
        diffs[g["difficulty"]]["n"] += 1
        diffs[g["difficulty"]]["ok"] += ok
        cg = channel_group(g.get("channel"))
        groups[cg]["n"] += 1
        groups[cg]["ok"] += ok

    def prf(c):
        pr = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        rc = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
        f1 = 2 * pr * rc / (pr + rc) if pr and rc else (0.0 if pr is not None and rc is not None else None)
        return {"precision": pr, "recall": rc, "f1": f1}

    cats = [{"category": k, **v, **prf(v)} for k, v in cat.items()]
    order = list(BROAD_GROUPS)
    for r in cats:
        r["group"] = group_of(r["category"]) if group_of(r["category"]) in BROAD_GROUPS else "unknown"
    cats.sort(key=lambda r: (order.index(r["group"]) if r["group"] in order else len(order), -r["gold"], -r["fp"]))
    rep = scorer.report()
    n = len(preds)
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "parse_rate": sum(p["pred"] is not None for p in preds) / n if n else None,
        "ctype_accuracy": sum(ctype["cells"][i][i] for i in range(3)) / n if n else None,
        "severity_accuracy": sum(sev["cells"][i][i] for i in range(4)) / n if n else None,
        "category_levels": [
            {"level": "Exact category", "note": "what the per-category table scores", **rep["violation_category_micro"]},
            {"level": "Rule level", "note": "categories grouped by taxonomy cite", **rep["category_rule_level_micro"]},
            {"level": "(d)(1) family", "note": "all 2210(d)(1) content standards as one group", **rep["category_d1_family_micro"]},
            {"level": "Broad group", "note": "the five broad groups below", **rep["category_broad_group_micro"]},
        ],
        "rule_cite": rep["rule_cite_micro"],
        "broad_groups": [
            {"group": g, "title": BROAD_GROUPS[g]["title"], "categories": BROAD_GROUPS[g]["categories"], **v}
            for g, v in rep["broad_groups"].items()
        ],
        "group_titles": {g: v["title"] for g, v in BROAD_GROUPS.items()},
        "compliant": comp,
        "ctype": ctype,
        "severity": sev,
        "categories": cats,
        "swaps": [{"missed": m, "predicted": x, "count": c} for (m, x), c in swaps.most_common(8)],
        "axes": {k: {"n": v["n"], "accuracy": v["ok"] / v["n"]} for k, v in sorted(axes.items())},
        "difficulty": {k: {"n": v["n"], "accuracy": v["ok"] / v["n"]} for k, v in sorted(diffs.items())},
        "channel_group": {k: {"n": v["n"], "accuracy": v["ok"] / v["n"]} for k, v in sorted(groups.items())},
    }
