"""Evaluate a (fine-tuned) Phi-4-mini reviewer on the test and OOD sets.

  python -m finra2210.evaluate --model runs/phi4mini-2210 --data data/splits/test.jsonl --out runs/phi4mini-2210/eval_test.json
  python -m finra2210.evaluate --model runs/phi4mini-2210 --data data/splits/ood.jsonl  --out runs/phi4mini-2210/eval_ood.json
  python -m finra2210.evaluate --model microsoft/Phi-4-mini-instruct --data data/splits/test.jsonl --limit 100   # zero-shot baseline

`--model` may be a LoRA adapter directory (base is read from adapter_config) or a
full model. Reports: JSON parse rate, compliant accuracy / F1, communication_type
accuracy, severity accuracy, violation-category micro P/R/F1, rule-cite F1, and
breakdowns by difficulty and (for OOD) by shift axis.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
_JSON_RE = re.compile(r"\{.*\}", re.S)


def _load(model_path: str):
    p = Path(model_path)
    base = model_path
    adapter = None
    if (p / "adapter_config.json").exists():
        adapter = model_path
        base = json.loads((p / "adapter_config.json").read_text())["base_model_name_or_path"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_path if adapter else base)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    if device != "cuda":
        model.to(device)
    model.eval()
    return tok, model, device


def _parse(text: str) -> dict | None:
    """Prefer the last fenced ```json verdict block; fall back to the widest brace span."""
    fences = _FENCE_RE.findall(text)
    candidates = list(reversed(fences)) if fences else []
    m = _JSON_RE.search(text)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


class Scorer:
    def __init__(self):
        self.n = 0
        self.parsed = 0
        self.compliant_correct = 0
        self.ctype_correct = 0
        self.sev_correct = 0
        self.c_tp = self.c_fp = self.c_fn = 0   # "violation present" as positive class
        self.cat_tp = self.cat_fp = self.cat_fn = 0
        self.rule_tp = self.rule_fp = self.rule_fn = 0

    def add(self, gold: dict, pred: dict | None) -> None:
        self.n += 1
        g_viol = not gold["compliant"]
        if pred is None:
            # unparsable counts as predicting "compliant / no findings"
            self.c_fn += g_viol
            self.cat_fn += len(gold["violations"])
            self.rule_fn += len({v["rule"] for v in gold["violations"]})
            return
        self.parsed += 1
        p_viol = not bool(pred.get("compliant", True))
        self.compliant_correct += g_viol == p_viol
        self.c_tp += g_viol and p_viol
        self.c_fp += (not g_viol) and p_viol
        self.c_fn += g_viol and (not p_viol)
        self.ctype_correct += pred.get("communication_type") == gold["communication_type"]
        self.sev_correct += pred.get("severity") == gold["severity"]
        g_cats = {v["category"] for v in gold["violations"]}
        p_cats = {v.get("category") for v in pred.get("violations", []) if isinstance(v, dict)}
        self.cat_tp += len(g_cats & p_cats); self.cat_fp += len(p_cats - g_cats); self.cat_fn += len(g_cats - p_cats)
        g_rules = {v["rule"] for v in gold["violations"]}
        p_rules = {v.get("rule") for v in pred.get("violations", []) if isinstance(v, dict)}
        self.rule_tp += len(g_rules & p_rules); self.rule_fp += len(p_rules - g_rules); self.rule_fn += len(g_rules - p_rules)

    def report(self) -> dict:
        n = max(1, self.n)
        return {
            "n": self.n,
            "json_parse_rate": round(self.parsed / n, 4),
            "compliant_accuracy": round(self.compliant_correct / n, 4),
            "violation_detection": _prf(self.c_tp, self.c_fp, self.c_fn),
            "communication_type_accuracy": round(self.ctype_correct / n, 4),
            "severity_accuracy": round(self.sev_correct / n, 4),
            "violation_category_micro": _prf(self.cat_tp, self.cat_fp, self.cat_fn),
            "rule_cite_micro": _prf(self.rule_tp, self.rule_fp, self.rule_fn),
        }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]
    tok, model, device = _load(args.model)

    overall = Scorer()
    by_diff: dict[str, Scorer] = defaultdict(Scorer)
    by_axis: dict[str, Scorer] = defaultdict(Scorer)
    preds_out = []

    for i in range(0, len(rows), args.batch):
        batch = rows[i : i + args.batch]
        # score the first review turn only (system + first user message); follow-up turns are training-only
        prompts = [tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(batch, texts):
            pred = _parse(t)
            overall.add(r["label"], pred)
            by_diff[r["difficulty"]].add(r["label"], pred)
            for ax in (r.get("ood_shifts") or {}):
                by_axis[ax].add(r["label"], pred)
            preds_out.append({"id": r["id"], "pred": pred, "raw": t if pred is None else None})
        print(f"{min(i + args.batch, len(rows))}/{len(rows)}  acc={overall.compliant_correct / max(1, overall.n):.3f}")

    report = {
        "model": args.model,
        "data": args.data,
        "overall": overall.report(),
        "by_difficulty": {k: v.report() for k, v in sorted(by_diff.items())},
    }
    if by_axis:
        report["by_ood_axis"] = {k: v.report() for k, v in sorted(by_axis.items())}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        Path(args.out).with_suffix(".preds.jsonl").write_text("\n".join(json.dumps(p) for p in preds_out))


if __name__ == "__main__":
    main()
