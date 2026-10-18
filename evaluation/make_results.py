"""Build results/ from finished eval runs: metrics, plots, gzipped predictions and a README.

  PYTHONPATH=src .venv-mlx/bin/python evaluation/make_results.py
  PYTHONPATH=src .venv-mlx/bin/python evaluation/make_results.py --tuned runs/phi4mini-2210-mlx --base runs/base-mlx --out results

Needs matplotlib (in .venv-mlx). Reads <run>/eval_{test,ood}.json and .preds.jsonl for the
fine-tuned and the zero-shot base run, the gold splits in data/splits/, and the training
artifacts (training_summary.md, loss.json, loss_curve.png) of the fine-tuned run. Rerun after any
new eval; everything under --out is regenerated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from finra2210.analysis import analyze, load_gold, read_preds  # noqa: E402
from finra2210.taxonomy import BROAD_GROUPS, SUBGROUPS  # noqa: E402

SPLITS = {"test": "data/splits/test.jsonl", "ood": "data/splits/ood.jsonl"}
SPLIT_NAME = {"test": "In-domain test (1,186)", "ood": "Out-of-distribution (5,800)"}
COL = {"base": "#b9b6ad", "tuned": "#1f5fbf", "test": "#1f5fbf", "ood": "#e07b39"}
GROUP_SHORT = {"misleading_content": "Misleading\ncontent", "required_disclosures": "Required\ndisclosures",
               "approval_and_audience": "Approval and\naudience", "regulator_endorsement": "FINRA / SIPC\nendorsement",
               "product_rules": "Product rules\n(options)"}
GROUP_COL = {"misleading_content": "#2f3e57", "required_disclosures": "#2e9e6a", "approval_and_audience": "#8a5cc2",
             "regulator_endorsement": "#d1495b", "product_rules": "#e0a030"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                     "axes.titleweight": "bold", "axes.titlesize": 11})


def headline(o: dict) -> dict:
    return {
        "Valid JSON": o["json_parse_rate"],
        "Compliant / not\ncompliant": o["compliant_accuracy"],
        "Communication\ntype": o["communication_type_accuracy"],
        "Severity": o["severity_accuracy"],
        "Category F1\n(exact)": o["violation_category_micro"]["f1"],
        "Category F1\n(broad group)": o["category_broad_group_micro"]["f1"],
    }


def bar_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.01, fmt.format(h), ha="center", va="bottom", fontsize=7)


def plot_headline(ev: dict, out: Path) -> None:
    keys = list(headline(ev["tuned"]["test"]["overall"]))
    series = [("Base, test", ev["base"]["test"], "#d9d6cf"), ("Base, OOD", ev["base"]["ood"], "#b9b6ad"),
              ("Fine-tuned, test", ev["tuned"]["test"], COL["test"]), ("Fine-tuned, OOD", ev["tuned"]["ood"], COL["ood"])]
    x = np.arange(len(keys)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (name, rep, c) in enumerate(series):
        vals = list(headline(rep["overall"]).values())
        bar_labels(ax, ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=c))
    ax.set_xticks(x, keys); ax.set_ylim(0, 1.12); ax.set_ylabel("score")
    ax.set_title("Fine-tuned vs zero-shot base Phi-4-mini")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_confusions(an: dict, split: str, out: Path) -> None:
    mats = [("Compliant vs violation", an["compliant"]), ("Communication type", an["ctype"]), ("Severity", an["severity"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [3, 5, 6]})
    short = lambda s: s.replace("_communication", "")
    for ax, (title, m) in zip(axes, mats):
        cells = np.array(m["cells"], dtype=float)
        rown = cells / np.maximum(cells.sum(1, keepdims=True), 1)
        ax.imshow(rown, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                if cells[i, j]:
                    ax.text(j, i, f"{int(cells[i, j])}\n{rown[i, j]:.0%}", ha="center", va="center", fontsize=8,
                            color="white" if rown[i, j] > 0.55 else "#222")
        ax.set_xticks(range(len(m["cols"])), [short(c) for c in m["cols"]], rotation=20)
        ax.set_yticks(range(len(m["rows"])), [short(r) for r in m["rows"]])
        ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(title)
        ax.spines[:].set_visible(False)
    fig.suptitle(f"Fine-tuned model: confusion matrices, {SPLIT_NAME[split]}", fontweight="bold")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def category_order() -> list[str]:
    order = []
    for g in BROAD_GROUPS:
        if g in SUBGROUPS:
            for cats in SUBGROUPS[g].values():
                order += cats
        else:
            order += BROAD_GROUPS[g]["categories"]
    return order


def per_category(an: dict) -> dict:
    return {c["category"]: c for c in an["categories"]}


def plot_per_category(an_test: dict, an_ood: dict, out: Path) -> None:
    cats = category_order()
    pt, po = per_category(an_test), per_category(an_ood)
    y = np.arange(len(cats)); h = 0.38
    f = lambda d, c: (d.get(c, {}).get("f1") or 0.0)
    fig, ax = plt.subplots(figsize=(9, 9.5))
    ax.barh(y - h / 2, [f(pt, c) for c in cats], h, color=COL["test"], label="test")
    ax.barh(y + h / 2, [f(po, c) for c in cats], h, color=COL["ood"], label="OOD")
    group_of = {c: g for g, v in BROAD_GROUPS.items() for c in v["categories"]}
    ax.set_yticks(y, cats, fontsize=8.5)
    for t, c in zip(ax.get_yticklabels(), cats):
        t.set_color(GROUP_COL[group_of[c]])
    for i, c in enumerate(cats):
        gt, go = pt.get(c, {}).get("gold", 0), po.get(c, {}).get("gold", 0)
        ax.text(1.01, i, f"n={gt} / {go}", va="center", fontsize=7, color="#666", transform=ax.get_yaxis_transform())
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("F1 (exact category)")
    ax.set_title("Per-category F1, fine-tuned model (label colour = broad group; n = gold test / OOD)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL["test"]), plt.Rectangle((0, 0), 1, 1, color=COL["ood"])]
    handles += [plt.Rectangle((0, 0), 1, 1, color=GROUP_COL[g]) for g in BROAD_GROUPS]
    labels = ["test", "OOD"] + [BROAD_GROUPS[g]["title"].split(" — ")[0] for g in BROAD_GROUPS]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_broad_groups(ev: dict, out: Path) -> None:
    groups = list(BROAD_GROUPS)
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, s in enumerate(["test", "ood"]):
        bg = ev["tuned"][s]["overall"]["broad_groups"]
        vals = [bg[g]["f1"] if bg[g]["support"] else 0 for g in groups]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=COL[s], label="test" if s == "test" else "OOD")
        bar_labels(ax, [b for b, g in zip(bars, groups) if bg[g]["support"]])
        for b, g in zip(bars, groups):
            ax.text(b.get_x() + b.get_width() / 2, -0.09, f"n={bg[g]['support']}", ha="center", fontsize=7, color="#666")
    ax.set_xticks(x, [GROUP_SHORT[g] for g in groups], fontsize=8.5)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1 (any category in the right group)")
    ax.set_title("Broad-group F1, fine-tuned model"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_levels(ev: dict, out: Path) -> None:
    levels = [("Exact\ncategory", "violation_category_micro"), ("Rule\nlevel", "category_rule_level_micro"),
              ("(d)(1)\nfamily", "category_d1_family_micro"), ("Broad\ngroup", "category_broad_group_micro")]
    x = np.arange(len(levels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, s in enumerate(["test", "ood"]):
        o = ev["tuned"][s]["overall"]
        bar_labels(ax, ax.bar(x + (i - 0.5) * w, [o[k]["f1"] for _, k in levels], w, color=COL[s], label="test" if s == "test" else "OOD"))
    ax.set_xticks(x, [l for l, _ in levels]); ax.set_ylim(0, 1.1); ax.set_ylabel("micro F1")
    ax.set_title("Category F1 at four levels of strictness"); ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_breakdown(ev: dict, key: str, title: str, out: Path, splits=("test", "ood"), order=None, names=None) -> None:
    metrics = [("compliant_accuracy", "Compliant / not compliant accuracy"), ("violation_category_micro", "Category F1 (exact)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (m, mname) in zip(axes, metrics):
        cats = order or sorted({k for s in splits for k in ev["tuned"][s].get(key, {})})
        x = np.arange(len(cats)); w = 0.8 / len(splits)
        for i, s in enumerate(splits):
            d = ev["tuned"][s].get(key, {})
            vals = [(d[c][m]["f1"] if isinstance(d[c][m], dict) else d[c][m]) if c in d else 0 for c in cats]
            bars = ax.bar(x + (i - (len(splits) - 1) / 2) * w, vals, w, color=COL[s], label="test" if s == "test" else "OOD")
            bar_labels(ax, [bar for bar, c in zip(bars, cats) if c in d])
            for j, c in enumerate(cats):
                if c in d:
                    ax.text(x[j] + (i - (len(splits) - 1) / 2) * w, 0.02, f"{d[c]['n']}", ha="center", fontsize=6.5, color="white")
        ax.set_xticks(x, [(names or {}).get(c, c) for c in cats], fontsize=8.5, rotation=15)
        ax.set_ylim(0, 1.12); ax.set_title(mname, fontsize=10)
        if len(splits) > 1:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(title, fontweight="bold"); fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fmt(x, pct=False):
    if x is None:
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.2f}"


def write_csv(an: dict, path: Path) -> None:
    rows = []
    group_of = {c: g for g, v in BROAD_GROUPS.items() for c in v["categories"]}
    for split, a in an.items():
        pc = per_category(a)
        for c in category_order():
            r = pc.get(c, {})
            rows.append({"split": split, "broad_group": group_of[c], "category": c, "gold": r.get("gold", 0),
                         "hits": r.get("tp", 0), "missed": r.get("fn", 0), "false_alarms": r.get("fp", 0),
                         "precision": "" if r.get("precision") is None else round(r["precision"], 4),
                         "recall": "" if r.get("recall") is None else round(r["recall"], 4),
                         "f1": "" if r.get("f1") is None else round(r["f1"], 4)})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def base_stats(run: Path, split: str) -> dict:
    """How the zero-shot base answers: parse rate, type accuracy when its own wording is normalised
    ("Retail Communication" -> retail_communication), and how often it says compliant."""
    gold = {r["id"]: r["label"] for r in map(json.loads, Path(SPLITS[split]).read_text().splitlines())}
    preds = read_preds(run / f"eval_{split}.preds.jsonl")
    parsed = [p for p in preds if p["pred"]]
    norm = lambda x: str(x).strip().lower().replace(" ", "_")
    return {
        "n": len(preds), "parsed": len(parsed),
        "lenient_ctype": sum(norm(p["pred"].get("communication_type")) == gold[p["id"]]["communication_type"] for p in parsed) / len(preds),
        "says_correspondence": sum(norm(p["pred"].get("communication_type")) == "correspondence" for p in parsed),
        "says_compliant": sum(str(p["pred"].get("compliant")).lower() in ("true", "yes", "compliant") for p in parsed) / max(1, len(parsed)),
    }


UNSEEN = ["options_communication", "third_party_content_adoption"]  # OOD-only categories (never in training)


def unseen_stats(run: Path) -> dict:
    """For each OOD-only category: gold items, how often the model names it, how often it still says not compliant."""
    from finra2210.evaluate import as_compliant
    gold = {r["id"]: r["label"] for r in map(json.loads, Path(SPLITS["ood"]).read_text().splitlines())}
    out = {}
    for c in UNSEEN:
        ids = {i for i, l in gold.items() if any(v["category"] == c for v in l["violations"])}
        named = flagged = 0
        for p in read_preds(run / "eval_ood.preds.jsonl"):
            if p["id"] in ids and p["pred"]:
                named += any(isinstance(v, dict) and v.get("category") == c for v in (p["pred"].get("violations") or []))
                flagged += not as_compliant(p["pred"].get("compliant"))
        out[c] = {"gold": len(ids), "named": named, "flagged_not_compliant": flagged}
    return out


def readme(ev: dict, an: dict, tuned_dir: Path, bs: dict, unseen: dict) -> str:
    t, o = ev["tuned"]["test"]["overall"], ev["tuned"]["ood"]["overall"]
    bt, bo = ev["base"]["test"]["overall"], ev["base"]["ood"]["overall"]
    rows = [
        ("Valid JSON verdict", "json_parse_rate", True), ("Compliant / not compliant accuracy", "compliant_accuracy", True),
        ("Communication type accuracy", "communication_type_accuracy", True), ("Severity accuracy", "severity_accuracy", True),
    ]
    prf_rows = [("Violation detection (any violation present)", "violation_detection"),
                ("Category, exact", "violation_category_micro"), ("Category, rule level", "category_rule_level_micro"),
                ("Category, (d)(1) family", "category_d1_family_micro"), ("Category, broad group", "category_broad_group_micro"),
                ("Rule cite written by the model", "rule_cite_micro")]
    L = ["# Results — v0.1.0", "",
         "Fine-tuned **Phi-4-mini-instruct + LoRA** (`runs/phi4mini-2210-mlx`, mlx-lm, 2 epochs) against the",
         "**zero-shot base model** (`mlx-community/Phi-4-mini-instruct-4bit`, same system prompt, no examples), on the",
         "in-domain held-out test set (1,186 items) and the out-of-distribution set (5,800 items). Generated by",
         "[`evaluation/make_results.py`](../evaluation/make_results.py); metric definitions are in",
         "[`evaluation/README.md`](../evaluation/README.md).", "",
         "## Headline", "",
         "| Metric | Base, test | Base, OOD | Fine-tuned, test | Fine-tuned, OOD |", "|---|---|---|---|---|"]
    for name, k, pct in rows:
        L.append(f"| {name} | {fmt(bt[k], pct)} | {fmt(bo[k], pct)} | **{fmt(t[k], pct)}** | **{fmt(o[k], pct)}** |")
    for name, k in prf_rows:
        L.append(f"| {name} — F1 | {fmt(bt[k]['f1'])} | {fmt(bo[k]['f1'])} | **{fmt(t[k]['f1'])}** | **{fmt(o[k]['f1'])}** |")
    L += ["", "![Fine-tuned vs base](plots/headline_tuned_vs_base.png)", "",
          "**Reading the base-model column.** The zero-shot base model gets the same system prompt as the fine-tuned",
          "model. The prompt asks for a JSON verdict with the field names, but does not list the 23 category labels or",
          "the exact communication-type and severity values. On test the base model:",
          f"- returns a parsable verdict for {bs['test']['parsed']:,} of {bs['test']['n']:,} items; unparsable answers count as wrong;",
          "- writes labels in its own words (\"Retail Communication\", \"Moderate\", invented category names), so exact-match",
          "  type, severity and category scores are near zero. Matching the type case-insensitively gives",
          f"  {fmt(bs['test']['lenient_ctype'], True)} on test and {fmt(bs['ood']['lenient_ctype'], True)} on OOD; it almost never answers correspondence",
          f"  ({bs['test']['says_correspondence']} of {bs['test']['parsed']:,} parsed test answers), although half the items are correspondence;",
          f"- says compliant in {fmt(bs['test']['says_compliant'], True)} of its parsed verdicts. Its compliant / not-compliant accuracy is below the",
          f"  {fmt(657 / 1186, True)} you would get on test by always answering \"not compliant\".", "",
          "The comparison measures what fine-tuning taught the model, not how a carefully prompted base model would do.", "",
          "## Fine-tuned model in detail", "",
          "### Precision and recall", "",
          "| Metric | Test P | Test R | Test F1 | OOD P | OOD R | OOD F1 |", "|---|---|---|---|---|---|---|"]
    for name, k in prf_rows:
        L.append(f"| {name} | {fmt(t[k]['precision'])} | {fmt(t[k]['recall'])} | {fmt(t[k]['f1'])} | "
                 f"{fmt(o[k]['precision'])} | {fmt(o[k]['recall'])} | {fmt(o[k]['f1'])} |")
    L += ["", "Most category errors are a neighbouring category inside the same rule family: F1 rises from",
          f"{fmt(t['violation_category_micro']['f1'])} (exact) to {fmt(t['category_broad_group_micro']['f1'])} (broad group) on test and from",
          f"{fmt(o['violation_category_micro']['f1'])} to {fmt(o['category_broad_group_micro']['f1'])} on OOD.", "",
          "![Category levels](plots/category_levels.png)", "",
          "### Broad groups", "", "| Broad group | Test n | Test P / R / F1 | OOD n | OOD P / R / F1 |", "|---|---|---|---|---|"]
    for g in BROAD_GROUPS:
        a, b = t["broad_groups"][g], o["broad_groups"][g]
        cell = lambda d: "—" if not d["support"] else f"{fmt(d['precision'])} / {fmt(d['recall'])} / {fmt(d['f1'])}"
        L.append(f"| {BROAD_GROUPS[g]['title']} | {a['support']} | {cell(a)} | {b['support']} | {cell(b)} |")
    L += ["", "![Broad groups](plots/broad_group_f1.png)", "",
          "### Per category", "",
          "Exact-category F1 for all 23 categories (full table with hits, misses and false alarms:",
          "[`metrics/per_category.csv`](metrics/per_category.csv)).", "",
          "![Per-category F1](plots/per_category_f1.png)", "",
          "| Category | Test n | Test F1 | OOD n | OOD F1 |", "|---|---|---|---|---|"]
    pt, po = per_category(an["test"]), per_category(an["ood"])
    for c in category_order():
        a, b = pt.get(c, {}), po.get(c, {})
        L.append(f"| `{c}` | {a.get('gold', 0)} | {fmt(a.get('f1')) if a.get('gold') else '—'} | {b.get('gold', 0)} | {fmt(b.get('f1')) if b.get('gold') else '—'} |")
    swaps = an["ood"]["swaps"][:6]
    L += ["", "Most common swaps on OOD (gold category missed → category predicted instead):", ""]
    L += [f"- `{s['missed']}` → `{s['predicted']}`: {s['count']}" for s in swaps]
    L += ["", "### Confusion matrices", "", "Rows are gold labels, columns are predictions; each cell shows the count and its share of the row.", "",
          "![Confusion, test](plots/confusion_test.png)", "", "![Confusion, OOD](plots/confusion_ood.png)", "",
          "### Breakdowns", "", "![Difficulty](plots/by_difficulty.png)", "", "![Channel group](plots/by_channel_group.png)", "",
          "Channel groups: `chat` = SMS, instant message, text blast; `chat_app` = the OOD-only WhatsApp, LinkedIn,",
          "X and Slack/Teams DMs; `non_chat` = everything else.", "", "![OOD axes](plots/by_ood_axis.png)", "",
          "Each OOD item is shifted on 1–3 axes, so an item counts toward every axis it shifts.", "",
          "| OOD shift axis | Items | Compliant / not compliant | Category F1 (exact) | Category F1 (broad group) |", "|---|---|---|---|---|"]
    ax = ev["tuned"]["ood"]["by_ood_axis"]
    for k in sorted(ax, key=lambda k: -ax[k]["violation_category_micro"]["f1"]):
        a = ax[k]
        L.append(f"| {k} | {a['n']:,} | {fmt(a['compliant_accuracy'], True)} | {fmt(a['violation_category_micro']['f1'])} | {fmt(a['category_broad_group_micro']['f1'])} |")
    L += ["", "The verdict holds on every axis; product shifts cost the most on naming the category.", "",
          "### Categories never seen in training", "",
          "| Category (OOD only) | OOD items | Named by the model | Still flagged not compliant |", "|---|---|---|---|"]
    L += [f"| `{c}` | {u['gold']} | {u['named']} | {u['flagged_not_compliant']} |" for c, u in unseen.items()]
    L += ["", "The model never names a category it was not trained on, but it still flags almost all of these items",
          "as not compliant, usually under a general 2210(d)(1) category such as `not_fair_and_balanced` or",
          "`promissory_or_guarantee`.", "",
          "## Training", "", "![Loss curve](training/loss_curve.png)", "",
          f"Full summary: [`training/training_summary.md`](training/training_summary.md) (copied from `{tuned_dir}`).", "",
          "## Files", "",
          "| Path | Contents |", "|---|---|",
          "| `metrics/fine_tuned_{test,ood}.json`, `metrics/base_zero_shot_{test,ood}.json` | full eval reports from `finra2210.evaluate` (overall and every breakdown) |",
          "| `metrics/per_category.csv` | fine-tuned per-category gold / hits / missed / false alarms / P / R / F1, test and OOD |",
          "| `metrics/summary.json` | headline numbers for all four runs |",
          "| `predictions/*.preds.jsonl.gz` | every prediction (`id`, parsed `pred`, `raw` text when unparsable), for re-scoring with `evaluate.py --score-only` |",
          "| `plots/` | the figures above |",
          "| `training/` | loss curve, loss data and training summary of the fine-tuned run |", "",
          "_Generated by `evaluation/make_results.py` from the eval runs; rerun it after a new eval._", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", default="runs/phi4mini-2210-mlx")
    ap.add_argument("--base", default="runs/base-mlx")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out, tuned, base = Path(args.out), Path(args.tuned), Path(args.base)
    for sub in ("metrics", "plots", "predictions", "training"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    ev = {"tuned": {}, "base": {}}
    for who, run, label in (("tuned", tuned, "fine_tuned"), ("base", base, "base_zero_shot")):
        for s in SPLITS:
            ev[who][s] = json.loads((run / f"eval_{s}.json").read_text())
            shutil.copyfile(run / f"eval_{s}.json", out / "metrics" / f"{label}_{s}.json")
            # mtime=0 and no file name in the gzip header: rebuilding unchanged predictions gives identical bytes
            with (run / f"eval_{s}.preds.jsonl").open("rb") as fi, (out / "predictions" / f"{label}_{s}.preds.jsonl.gz").open("wb") as raw, \
                    gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as fo:
                shutil.copyfileobj(fi, fo)
    for f in ("training_summary.md", "loss.json", "loss_curve.png"):
        if (tuned / f).exists():
            shutil.copyfile(tuned / f, out / "training" / f)

    an = {}
    for s, path in SPLITS.items():
        gold, _ = load_gold(Path(path))
        an[s] = analyze(read_preds(tuned / f"eval_{s}.preds.jsonl"), gold)
        assert an[s]["n"] == ev["tuned"][s]["overall"]["n"], (s, an[s]["n"])

    plot_headline(ev, out / "plots" / "headline_tuned_vs_base.png")
    for s in SPLITS:
        plot_confusions(an[s], s, out / "plots" / f"confusion_{s}.png")
    plot_per_category(an["test"], an["ood"], out / "plots" / "per_category_f1.png")
    plot_broad_groups(ev, out / "plots" / "broad_group_f1.png")
    plot_levels(ev, out / "plots" / "category_levels.png")
    plot_breakdown(ev, "by_difficulty", "Fine-tuned model by difficulty (bar labels at the bottom = n)", out / "plots" / "by_difficulty.png",
                   order=["easy", "medium", "hard"])
    plot_breakdown(ev, "by_channel_group", "Fine-tuned model by channel group (n at the bottom of each bar)", out / "plots" / "by_channel_group.png",
                   order=["chat", "chat_app", "non_chat"])
    plot_breakdown(ev, "by_ood_axis", "Fine-tuned model on OOD, by shift axis (n at the bottom of each bar)", out / "plots" / "by_ood_axis.png",
                   splits=("ood",))
    write_csv(an, out / "metrics" / "per_category.csv")
    summary = {who: {s: ev[who][s]["overall"] for s in SPLITS} for who in ev}
    summary["models"] = {"tuned": ev["tuned"]["test"]["model"], "base": ev["base"]["test"]["model"]}
    (out / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2))
    bs = {s: base_stats(base, s) for s in SPLITS}
    summary["base_zero_shot_notes"] = bs
    (out / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2))
    unseen = unseen_stats(tuned)
    summary["unseen_categories_ood"] = unseen
    (out / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "README.md").write_text(readme(ev, an, tuned, bs, unseen))
    print(f"wrote {out}/ (README.md, metrics/, plots/, predictions/, training/)")


if __name__ == "__main__":
    main()
