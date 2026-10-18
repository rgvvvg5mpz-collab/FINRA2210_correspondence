"""Evaluate a (fine-tuned) Phi-4-mini reviewer on the test and OOD sets.

  python -m finra2210.evaluate --model runs/phi4mini-2210 --data data/splits/test.jsonl --out runs/phi4mini-2210/eval_test.json
  python -m finra2210.evaluate --model runs/phi4mini-2210 --data data/splits/ood.jsonl  --out runs/phi4mini-2210/eval_ood.json
  python -m finra2210.evaluate --model microsoft/Phi-4-mini-instruct --data data/splits/test.jsonl --limit 100   # zero-shot baseline

Apple Silicon (mlx-lm, see configs/phi4mini_lora_mlx.yaml):
  python -m finra2210.evaluate --backend mlx --model runs/phi4mini-2210-mlx --data data/splits/test.jsonl --out runs/phi4mini-2210-mlx/eval_test.json
  python -m finra2210.evaluate --backend mlx --model mlx-community/Phi-4-mini-instruct-4bit --data data/splits/test.jsonl --out runs/base-mlx/eval_test.json

`--model` may be a LoRA adapter directory (base is read from adapter_config; for mlx the
"model" key) or a full model. Predictions are appended to <out>.preds.jsonl as they are
produced; `--resume` skips ids already there, so a long run can be restarted.
`--score-only` re-scores that file without loading a model (e.g. after adding a metric), even
while an eval is still writing it. Reports: JSON parse rate, compliant accuracy / F1, communication_type
accuracy, severity accuracy, violation-category micro P/R/F1, rule-cite F1 (the cite string
the model wrote), the category metric at two coarser levels (category_rule_level_micro: a
category counts as right if it maps to the gold category's taxonomy cite, e.g. promissory vs
exaggerated are both (d)(1)(B); category_d1_family_micro: additionally every 2210(d)(1) content
standard is one group; category_broad_group_micro: categories grouped into the five
taxonomy.BROAD_GROUPS), per-broad-group P/R/F1 (broad_groups), and breakdowns by difficulty, by gold communication type (correspondence / retail /
institutional), by channel, by channel group (chat = sms_text / instant_message /
text_message_blast; chat_app = OOD WhatsApp / LinkedIn / X / Slack-Teams DMs; non_chat; and
chat x communication type) and (for OOD) by shift axis.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .taxonomy import BROAD_GROUP_OF, BROAD_GROUPS, VIOLATION_CATEGORIES

_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
_JSON_RE = re.compile(r"\{.*\}", re.S)

# the deployment target: short-form chat. Reported separately so it is not averaged away by the email/web majority.
CHAT_CHANNELS = {"sms_text", "instant_message", "text_message_blast"}
# OOD-only messaging-app channels: chat-shaped but kept apart so "chat" means the same thing on test and OOD
CHAT_APP_CHANNELS = {"WhatsApp message", "LinkedIn direct message", "X/Twitter DM", "Slack/Teams message to a client"}


def channel_group(channel: str | None) -> str:
    if channel in CHAT_CHANNELS:
        return "chat"
    return "chat_app" if channel in CHAT_APP_CHANNELS else "non_chat"


D1_FAMILY = "2210(d)(1)"


def rule_of(category: str | None) -> str | None:
    """The taxonomy cite for a category; unknown categories stay as themselves (so they count as misses/false alarms)."""
    return VIOLATION_CATEGORIES.get(category, {}).get("rule", category)


def family_of(category: str | None) -> str | None:
    """Rule cite, with every 2210(d)(1) content standard (fair/balanced, promissory, misleading, projection, ...) as one group."""
    r = rule_of(category)
    return D1_FAMILY if r and r.startswith(D1_FAMILY) else r


def group_of(category: str | None) -> str | None:
    """Broad group (taxonomy.BROAD_GROUPS) of a category; unknown categories stay as themselves."""
    return BROAD_GROUP_OF.get(category, category)


def _load_hf(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    @torch.inference_mode()
    def generate(prompts: list[list[dict]], max_new_tokens: int) -> list[str]:
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def _load_mlx(model_path: str):
    from mlx_lm import batch_generate, load

    p = Path(model_path)
    if (p / "adapter_config.json").exists():
        base = json.loads((p / "adapter_config.json").read_text())["model"]
        model, tok = load(base, adapter_path=model_path)
    else:
        model, tok = load(model_path)

    def generate(prompts: list[list[dict]], max_new_tokens: int) -> list[str]:
        ids = [tok.apply_chat_template(m, add_generation_prompt=True) for m in prompts]
        return batch_generate(model, tok, ids, max_tokens=max_new_tokens).texts

    return generate


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
        # predicted categories grouped by their taxonomy cite, and with all of (d)(1) as one group
        self.grp = {"rule_level": [0, 0, 0], "d1_family": [0, 0, 0], "broad_group": [0, 0, 0]}
        self.per_group = {g: [0, 0, 0] for g in BROAD_GROUPS}  # tp, fp, fn per broad group

    def _add_grouped(self, g_cats: set, p_cats: set) -> None:
        for name, key in (("rule_level", rule_of), ("d1_family", family_of), ("broad_group", group_of)):
            g, p = {key(c) for c in g_cats}, {key(c) for c in p_cats}
            t = self.grp[name]
            t[0] += len(g & p); t[1] += len(p - g); t[2] += len(g - p)
        g, p = {group_of(c) for c in g_cats}, {group_of(c) for c in p_cats}
        for name, t in self.per_group.items():
            t[0] += name in g and name in p; t[1] += name in p and name not in g; t[2] += name in g and name not in p

    def add(self, gold: dict, pred: dict | None) -> None:
        self.n += 1
        g_viol = not gold["compliant"]
        if pred is None:
            # unparsable counts as predicting "compliant / no findings"
            self.c_fn += g_viol
            self.cat_fn += len(gold["violations"])
            self.rule_fn += len({v["rule"] for v in gold["violations"]})
            self._add_grouped({v["category"] for v in gold["violations"]}, set())
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
        self._add_grouped(g_cats, p_cats)
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
            "category_rule_level_micro": _prf(*self.grp["rule_level"]),
            "category_d1_family_micro": _prf(*self.grp["d1_family"]),
            "category_broad_group_micro": _prf(*self.grp["broad_group"]),
            "broad_groups": {g: {**_prf(*t), "support": t[0] + t[2], "hits": t[0], "false_alarms": t[1]} for g, t in self.per_group.items()},
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--backend", choices=["hf", "mlx"], default="hf")
    ap.add_argument("--resume", action="store_true", help="skip ids already in <out>.preds.jsonl")
    ap.add_argument("--score-only", action="store_true",
                    help="re-score <out>.preds.jsonl without loading a model; rows without a prediction are left out. "
                         "Safe to run while another eval is still appending to the same file")
    args = ap.parse_args()
    if args.score_only and not args.out:
        ap.error("--score-only needs --out (the predictions are read from <out>.preds.jsonl)")

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]

    preds_path = Path(args.out).with_suffix(".preds.jsonl") if args.out else None
    done: dict[str, dict | None] = {}
    if preds_path:
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        if (args.resume or args.score_only) and preds_path.exists():
            for line in preds_path.read_text().splitlines():
                if line.strip():
                    try:
                        p = json.loads(line)
                    except json.JSONDecodeError:
                        if args.score_only:
                            break  # last line still being written by a running eval
                        raise
                    done[p["id"]] = p["pred"]
        elif preds_path.exists():
            preds_path.unlink()
    todo = [r for r in rows if r["id"] not in done]
    if args.score_only:
        if not done:
            ap.error(f"--score-only: no predictions in {preds_path}")
        if todo:
            print(f"score-only: {len(rows) - len(todo)} of {len(rows)} rows have predictions; scoring those only")
        rows = [r for r in rows if r["id"] in done]
        todo = []
    elif done:
        print(f"resuming: {len(rows) - len(todo)} of {len(rows)} already scored")

    overall = Scorer()
    by_diff: dict[str, Scorer] = defaultdict(Scorer)
    by_ctype: dict[str, Scorer] = defaultdict(Scorer)
    by_axis: dict[str, Scorer] = defaultdict(Scorer)
    by_channel: dict[str, Scorer] = defaultdict(Scorer)
    by_group: dict[str, Scorer] = defaultdict(Scorer)
    by_chat_ctype: dict[str, Scorer] = defaultdict(Scorer)

    def score(r: dict, pred: dict | None) -> None:
        overall.add(r["label"], pred)
        by_diff[r["difficulty"]].add(r["label"], pred)
        by_ctype[r["label"]["communication_type"]].add(r["label"], pred)
        ch = r.get("channel") or "unknown"
        by_channel[ch].add(r["label"], pred)
        by_group[channel_group(ch)].add(r["label"], pred)
        if ch in CHAT_CHANNELS:
            by_chat_ctype[r["label"]["communication_type"]].add(r["label"], pred)
        for ax in (r.get("ood_shifts") or {}):
            by_axis[ax].add(r["label"], pred)

    for r in rows:
        if r["id"] in done:
            score(r, done[r["id"]])

    generate = (_load_mlx if args.backend == "mlx" else _load_hf)(args.model) if todo else None
    preds_f = preds_path.open("a") if preds_path and todo else None
    for i in range(0, len(todo), args.batch):
        batch = todo[i : i + args.batch]
        # score the first review turn only (system + first user message); follow-up turns are training-only
        texts = generate([r["messages"][:2] for r in batch], args.max_new_tokens)
        for r, t in zip(batch, texts):
            pred = _parse(t)
            score(r, pred)
            if preds_f:
                preds_f.write(json.dumps({"id": r["id"], "pred": pred, "raw": t if pred is None else None}) + "\n")
        if preds_f:
            preds_f.flush()
        print(f"{overall.n}/{len(rows)}  acc={overall.compliant_correct / max(1, overall.n):.3f}", flush=True)
    if preds_f:
        preds_f.close()

    report = {
        "model": args.model,
        "data": args.data,
        "overall": overall.report(),
        "by_difficulty": {k: v.report() for k, v in sorted(by_diff.items())},
        "by_communication_type": {k: v.report() for k, v in sorted(by_ctype.items())},
        "by_channel_group": {k: v.report() for k, v in sorted(by_group.items())},
        "chat_by_communication_type": {k: v.report() for k, v in sorted(by_chat_ctype.items())},
        "by_channel": {k: v.report() for k, v in sorted(by_channel.items())},
    }
    if by_axis:
        report["by_ood_axis"] = {k: v.report() for k, v in sorted(by_axis.items())}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
