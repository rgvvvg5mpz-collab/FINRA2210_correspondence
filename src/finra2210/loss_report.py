"""Loss curve + summary for a LoRA run: mlx-lm (parsed from its train.log) or PyTorch/TRL on CUDA
(from the trainer_state.json that train_phi4.py writes).

  python -m finra2210.loss_report --log runs/phi4mini-2210-mlx/train.log
  python -m finra2210.loss_report --hf-state runs/phi4mini-2210/trainer_state.json

Writes next to the log (or to --out): loss.json (every train/val report plus derived facts),
loss_curve.png (log-scale train and validation loss, epoch boundaries, spikes, best checkpoint)
and training_summary.md. Safe to run mid-training; the summary says whether the run finished.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_VAL = re.compile(r"^\s+(\d+)\s+val\s+([\d.]+)\s+([\d.]+)s")
_TRAIN = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+\S?\s*([\d.]+)\s+([\d.]+)k")
_HEADER = re.compile(r"batch · iters\s+(\d+) · ([\d,]+)")


def parse(log: Path) -> dict:
    train, val, batch, iters = [], [], None, None
    for line in log.read_text(errors="replace").splitlines():
        line = _ANSI.sub("", line)
        if m := _HEADER.search(line):
            batch, iters = int(m[1]), int(m[2].replace(",", ""))
        elif m := _VAL.match(line):
            val.append({"iter": int(m[1]), "loss": float(m[2]), "seconds": float(m[3])})
        elif m := _TRAIN.match(line):
            train.append({"iter": int(m[1]), "loss": float(m[2]), "tok_s": float(m[3]),
                          "trained_tokens": int(float(m[4]) * 1000)})
    return {"batch": batch, "iters_total": iters, "train": train, "val": val}


def parse_hf(state_path: Path) -> tuple[dict, int | None]:
    """Same shape as parse(), from a transformers trainer_state.json; iterations are optimizer steps."""
    state = json.loads(state_path.read_text())
    targs_path = state_path.parent / "train_args.json"
    targs = json.loads(targs_path.read_text()) if targs_path.exists() else {}
    train = [{"iter": h["step"], "loss": h["loss"]} for h in state["log_history"] if "loss" in h]
    val = [{"iter": h["step"], "loss": h["eval_loss"], "seconds": h.get("eval_runtime", 0.0)}
           for h in state["log_history"] if "eval_loss" in h]
    d = {"batch": targs.get("effective_batch"), "iters_total": state.get("max_steps"), "train": train, "val": val,
         "backend": "PyTorch / TRL", "x_unit": "Optimizer step", "train_every": state.get("logging_steps"),
         "checkpoint_fmt": "checkpoint-{}", "epoch_window": 50}
    return d, targs.get("train_examples")


def analyse(d: dict, run_dir: Path, train_examples: int | None) -> dict:
    train, val = d["train"], d["val"]
    last_iter = max([t["iter"] for t in train] + [v["iter"] for v in val] + [0])
    facts: dict = {"last_iter": last_iter, "finished": d["iters_total"] is not None and last_iter >= d["iters_total"]}
    if train_examples and d["batch"]:
        per_epoch = train_examples / d["batch"]
        end = d["iters_total"] or last_iter
        facts["epoch_boundaries"] = [round(per_epoch * k) for k in range(1, 10)
                                     if per_epoch * k < end - 0.05 * per_epoch]
    if val:
        best = min(val, key=lambda v: v["loss"])
        ckpt = run_dir / d.get("checkpoint_fmt", "{:07d}_adapters.safetensors").format(best["iter"])
        facts["best_val"] = {**best, "checkpoint": ckpt.name if ckpt.exists() else None}
        facts["final_val"] = val[-1]
        facts["val_rising"] = val[-1]["loss"] > best["loss"] + 0.005
    # spikes: a report more than 3x the median of up to 10 neighbours on each side
    # (at least 5 on each side, so the initial descent is not flagged)
    spikes = []
    for i, t in enumerate(train):
        before, after = train[max(0, i - 10): i], train[i + 1: i + 11]
        window = [x["loss"] for x in before + after]
        if len(before) >= 5 and len(after) >= 5 and t["loss"] > 3 * statistics.median(window):
            spikes.append({"iter": t["iter"], "loss": t["loss"], "neighbour_median": round(statistics.median(window), 3)})
    facts["spikes"] = spikes
    tail = train[-10:]
    if tail:
        facts["recent_train_mean"] = round(statistics.mean(t["loss"] for t in tail), 4)
        if val:
            facts["train_val_gap"] = round(val[-1]["loss"] - facts["recent_train_mean"], 4)
    steps = []
    w = d.get("epoch_window", 200)  # 200 iterations at batch 4 = 800 examples
    for b in facts.get("epoch_boundaries", []):
        before = [t["loss"] for t in train if b - w <= t["iter"] < b]
        after = [t["loss"] for t in train if b < t["iter"] <= b + w]
        if len(before) >= 5 and len(after) >= 5:
            steps.append({"boundary": b, "train_mean_before": round(statistics.mean(before), 4),
                          "train_mean_after": round(statistics.mean(after), 4)})
    facts["epoch_steps"] = steps
    return facts


def plot(d: dict, facts: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    ax.plot([t["iter"] for t in d["train"]], [t["loss"] for t in d["train"]],
            color="#2a78d6", lw=1.3, label=f"Train loss (every {d.get('train_every') or 20} it)")
    ax.plot([v["iter"] for v in d["val"]], [v["loss"] for v in d["val"]], color="#eb6834", lw=2,
            ls="--", marker="o", ms=5, mec="white", mew=1.5, label="Validation loss")
    for b in facts.get("epoch_boundaries", []):
        ax.axvline(b, color="#898781", ls=":", lw=1)
        ax.text(b + 0.006 * (d["iters_total"] or facts["last_iter"]), 4.2, f"epoch {facts['epoch_boundaries'].index(b) + 2}", color="#898781", fontsize=9)
    for s in facts["spikes"]:
        ax.annotate(f"spike {s['loss']:.2f}", (s["iter"], s["loss"]), xytext=(12, -4),
                    textcoords="offset points", fontsize=8, color="#52514e")
    if best := facts.get("best_val"):
        ax.plot(best["iter"], best["loss"], marker="*", ms=13, color="#eb6834", mec="white", zorder=5)
        ax.annotate(f"best val {best['loss']:.3f} @ {best['iter']:,}", (best["iter"], best["loss"]),
                    xytext=(0, 14), textcoords="offset points", fontsize=9, ha="center", color="#52514e")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 5.5)
    ax.set_yticks([0.2, 0.3, 0.5, 1, 2, 5])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0, d["iters_total"] or facts["last_iter"])
    ax.set_xlabel(f"{d.get('x_unit', 'Iteration')} (batch {d['batch']})")
    ax.set_ylabel("Loss (log scale)")
    status = "finished" if facts["finished"] else f"in progress, {facts['last_iter']:,} / {d['iters_total']:,}"
    ax.set_title(f"Phi-4-mini LoRA ({d.get('backend', 'mlx-lm')}), {status}", fontsize=11, loc="left")
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summary_md(d: dict, facts: dict, log: Path) -> str:
    lines = [f"# Training summary: `{log.parent}`", ""]
    status = "finished" if facts["finished"] else "in progress / stopped early"
    lines += [f"- Status: {status}; last logged iteration {facts['last_iter']:,} of {d['iters_total']:,} "
              f"(batch {d['batch']})."]
    if best := facts.get("best_val"):
        lines.append(f"- Best validation loss: {best['loss']:.3f} at iteration {best['iter']:,}"
                     + (f" (checkpoint `{best['checkpoint']}`)." if best["checkpoint"] else "."))
        fv = facts["final_val"]
        lines.append(f"- Last validation loss: {fv['loss']:.3f} at iteration {fv['iter']:,}; first: {d['val'][0]['loss']:.3f}.")
        lines.append("- Overfitting check: " + (
            f"validation loss has risen {fv['loss'] - best['loss']:.3f} above its minimum; prefer the best checkpoint."
            if facts["val_rising"] else "validation loss is at or within 0.005 of its minimum; no sign of overfitting."))
    if "train_val_gap" in facts:
        lines.append(f"- Train loss (mean of last 10 reports): {facts['recent_train_mean']:.3f}; "
                     f"gap to last validation loss: {facts['train_val_gap']:+.3f}.")
    for s in facts["epoch_steps"]:
        lines.append(f"- Epoch boundary at iteration {s['boundary']:,}: train loss {s['train_mean_before']:.3f} "
                     f"in the 200 iterations before vs {s['train_mean_after']:.3f} after (second pass over seen examples).")
    for s in facts["spikes"]:
        lines.append(f"- Loss spike at iteration {s['iter']:,}: {s['loss']:.3f} vs neighbour median "
                     f"{s['neighbour_median']:.3f}; check that validation loss recovered afterwards.")
    lines += ["", "## Validation loss", "", "| Iteration | Loss | Seconds |", "|---|---|---|"]
    lines += [f"| {v['iter']:,} | {v['loss']:.3f} | {v['seconds']:.0f} |" for v in d["val"]]
    lines += ["", "![loss curve](loss_curve.png)", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--log", help="mlx-lm train.log")
    src.add_argument("--hf-state", help="trainer_state.json from train_phi4.py (PyTorch / CUDA)")
    ap.add_argument("--out", default=None, help="output dir (default: the log's directory)")
    ap.add_argument("--data-manifest", default="data/mlx/manifest.json",
                    help="mlx_data manifest, for train_examples (epoch boundaries)")
    args = ap.parse_args()

    log = Path(args.log or args.hf_state)
    out = Path(args.out) if args.out else log.parent
    out.mkdir(parents=True, exist_ok=True)
    if args.hf_state:
        d, train_examples = parse_hf(log)
    else:
        manifest = Path(args.data_manifest)
        train_examples = json.loads(manifest.read_text())["train_examples"] if manifest.exists() else None
        d = parse(log)
    facts = analyse(d, log.parent, train_examples)
    (out / "loss.json").write_text(json.dumps({**d, "facts": facts}, indent=1))
    plot(d, facts, out / "loss_curve.png")
    (out / "training_summary.md").write_text(summary_md(d, facts, log))
    print(f"wrote {out / 'loss.json'}, {out / 'loss_curve.png'}, {out / 'training_summary.md'}")


if __name__ == "__main__":
    main()
