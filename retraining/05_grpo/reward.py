"""Automatic reward for reinforcement learning (GRPO): score a model answer against the gold label.

  .venv/bin/python retraining/05_grpo/reward.py --demo
  .venv/bin/python retraining/05_grpo/reward.py --split retraining/03_sft/sample/out/splits/train.jsonl \
      --completions retraining/05_grpo/sample/completions.sample.jsonl

No learned reward model: every component is computed by code from the label, the same way the eval
scores the model. Components (weights in WEIGHTS, total clipped to [-1, 1]):
  format       the answer must end in a parsable JSON verdict with the six keys; otherwise -1 overall
  compliant    right compliant / not-compliant call
  type         right communication type
  category     exact-category F1, plus partial credit at broad-group level (near-misses earn some)
  severity     right severity
  grounding    share of cited excerpts that really appear in the message (made-up quotes cost reward)
  cite         share of findings whose rule is the one the category requires
  missed_high  extra penalty when a high-severity violation is called compliant (the costly error)
`finra_reward` is the function to hand to TRL's GRPOTrainer (see train_grpo.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import LABEL_KEYS, answer_for, categories, load_jsonl, parse_verdict  # noqa: E402
from finra2210.evaluate import as_compliant, group_of, pred_violations  # noqa: E402
from finra2210.taxonomy import VIOLATION_CATEGORIES  # noqa: E402

# positive weights sum to 1.0, so a perfect answer scores +1.0
WEIGHTS = {"compliant": 0.30, "type": 0.10, "category": 0.20, "broad_group": 0.10, "severity": 0.10,
           "grounding": 0.15, "cite": 0.05, "missed_high": -0.50}
_WS = re.compile(r"\s+")


def _norm(t: str) -> str:
    return _WS.sub(" ", t.lower()).strip()


def _f1(gold: set, pred: set) -> float:
    if not gold and not pred:
        return 1.0
    tp = len(gold & pred)
    return 2 * tp / (len(gold) + len(pred)) if gold or pred else 1.0


def score(answer: str, gold: dict, communication: str) -> dict:
    """Reward components and the total for one answer."""
    pred = parse_verdict(answer)
    if not pred or any(k not in pred for k in LABEL_KEYS):
        return {"total": -1.0, "format": 0.0}
    viols = pred_violations(pred)
    g_cats, p_cats = categories(gold), {v.get("category") for v in viols}
    text = _norm(communication)
    c = {
        "format": 1.0,
        "compliant": 1.0 if as_compliant(pred.get("compliant")) == bool(gold["compliant"]) else 0.0,
        "type": 1.0 if pred.get("communication_type") == gold["communication_type"] else 0.0,
        "category": _f1(g_cats, p_cats),
        "broad_group": _f1({group_of(x) for x in g_cats}, {group_of(x) for x in p_cats}),
        "severity": 1.0 if pred.get("severity") == gold["severity"] else 0.0,
        "grounding": (sum(bool(v.get("excerpt")) and _norm(str(v["excerpt"])) in text for v in viols) / len(viols)) if viols else 1.0,
        "cite": (sum(VIOLATION_CATEGORIES.get(v.get("category"), {}).get("rule") == v.get("rule") for v in viols) / len(viols)) if viols else 1.0,
        "missed_high": 1.0 if gold["severity"] == "high" and as_compliant(pred.get("compliant")) else 0.0,
    }
    total = sum(WEIGHTS[k] * c[k] for k in WEIGHTS)
    c["total"] = max(-1.0, min(1.0, total))
    return c


def _text(completion) -> str:
    """TRL passes conversational completions as [{"role": "assistant", "content": ...}]; plain text otherwise."""
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def finra_reward(prompts, completions, label, communication, **kwargs) -> list[float]:
    """GRPOTrainer reward function. `label` (JSON string) and `communication` are dataset columns."""
    return [score(_text(c), json.loads(l) if isinstance(l, str) else l, t)["total"]
            for c, l, t in zip(completions, label, communication)]


def demo() -> None:
    item = {"id": "fake-demo-1", "channel": "email",
            "metadata": {"sender_role": "registered representative", "firm": "Lakemont Example Securities LLC",
                         "recipient_description": "a retail client", "recipient_count_30d": 2, "recipient_type": "retail", "subject": ""},
            "communication": "Hi Lee, this income fund is a can't-lose way to get 6% a year. Happy to set it up today. Ana Ruiz, Lakemont Example Securities LLC, Member FINRA/SIPC"}
    gold = {"communication_type": "correspondence", "compliant": False, "severity": "high", "required_actions": ["Return to rep for revision"],
            "rationale": "Promissory under (d)(1)(B).",
            "violations": [{"category": "promissory_or_guarantee", "rule": "2210(d)(1)(B)", "excerpt": "this income fund is a can't-lose way to get 6% a year", "explanation": "Promises a result."}]}
    variants = {
        "perfect answer": gold,
        "right group, wrong category": {**gold, "violations": [{**gold["violations"][0], "category": "exaggerated_or_unwarranted_claim"}]},
        "made-up excerpt": {**gold, "violations": [{**gold["violations"][0], "excerpt": "guaranteed 6% forever"}]},
        "wrong rule cite": {**gold, "violations": [{**gold["violations"][0], "rule": "2210(d)(1)(A)"}]},
        "severity off by one": {**gold, "severity": "medium"},
        "called compliant (missed high)": {**gold, "compliant": True, "violations": [], "severity": "none"},
    }
    print(f"{'answer':34} total  components")
    for name, verdict in variants.items():
        c = score(answer_for(item, verdict), gold, item["communication"])
        print(f"{name:34} {c['total']:+.2f}  " + " ".join(f"{k}={v:.2f}" for k, v in c.items() if k != "total"))
    c = score("I think this is fine.", gold, item["communication"])
    print(f"{'no JSON verdict':34} {c['total']:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="score six variants of one answer and print the components")
    ap.add_argument("--split", help="split file with the gold labels")
    ap.add_argument("--completions", help="JSONL of {id, completion} to score against --split")
    args = ap.parse_args()
    if args.demo or not args.completions:
        demo()
        return
    gold = {r["id"]: r for r in load_jsonl(args.split)}
    for row in load_jsonl(args.completions):
        r = gold[row["id"]]
        c = score(row["completion"], r["label"], r["communication"])
        print(f"{row['id']:22} {row.get('note', ''):32} {c['total']:+.2f}")


if __name__ == "__main__":
    main()
