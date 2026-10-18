"""Turn a reviewer feedback log into training data for the next round.

  .venv/bin/python retraining/01_feedback/feedback_to_data.py \
      --log retraining/01_feedback/sample/feedback_log.sample.jsonl --out retraining/01_feedback/out

Input: one JSON object per reviewed message (format in README.md). Output in --out:
  corrections.jsonl          labeled samples (clean-file format) for an SFT round: every accepted
                             verdict and every valid correction
  preferences.jsonl          DPO pairs {prompt, chosen, rejected}: corrected answer vs the model's answer
  kto.jsonl                  unpaired feedback {prompt, completion, label} for TRL's KTOTrainer
  invalid_feedback.jsonl     corrections that fail the label checks, with the reasons
  summary.json               counts
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import answer_for, check_label, parse_verdict, prompt_messages, same_verdict, write_jsonl  # noqa: E402

DECISIONS = {"accept", "edit", "reject", "thumbs_up", "thumbs_down"}
SAMPLE_KEYS = ("channel", "metadata", "communication")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="reviewer feedback log (JSONL)")
    ap.add_argument("--out", default="retraining/01_feedback/out")
    args = ap.parse_args()

    corrections, prefs, kto, invalid = [], [], [], []
    counts: Counter = Counter()
    for line_no, line in enumerate(Path(args.log).read_text().splitlines(), 1):
        if not line.strip():
            continue
        fb = json.loads(line)
        decision = fb.get("decision")
        if decision not in DECISIONS or any(k not in fb.get("sample", {}) for k in SAMPLE_KEYS):
            invalid.append({**fb, "_problems": [f"line {line_no}: bad decision or sample missing {SAMPLE_KEYS}"]})
            continue
        counts[decision] += 1
        sample = {"id": f"fb-{fb['review_id']}", **fb["sample"]}
        prompt = fb.get("prompt") or prompt_messages(sample)
        model_text = fb.get("model_output", "")
        model_verdict = parse_verdict(model_text)
        source = {k: fb.get(k) for k in ("review_id", "timestamp", "model_version", "reviewer", "decision", "notes")}

        # unpaired signal: was the model's answer acceptable as written?
        if model_text:
            kto.append({"prompt": prompt, "completion": [{"role": "assistant", "content": model_text}],
                        "label": decision in ("accept", "thumbs_up"), "source": source})

        if decision == "accept":
            label = model_verdict
        elif decision in ("edit", "reject"):
            label = fb.get("corrected_verdict")
        else:
            continue  # thumbs only: no label to learn from
        if not label:
            invalid.append({**fb, "_problems": ["no usable verdict (model answer unparsable or corrected_verdict missing)"]})
            continue
        problems = check_label(sample, label)
        if problems:
            invalid.append({**fb, "_problems": problems})
            continue
        corrections.append({**sample, "difficulty": fb.get("difficulty", "unrated"), "label": label,
                            "teacher": f"reviewer:{fb.get('reviewer', 'unknown')}", "source": source})
        if decision in ("edit", "reject") and model_text and not same_verdict(label, model_verdict):
            prefs.append({"prompt": prompt, "chosen": [{"role": "assistant", "content": answer_for(sample, label)}],
                          "rejected": [{"role": "assistant", "content": model_text}], "source": source})

    out = Path(args.out)
    write_jsonl(out / "corrections.jsonl", corrections)
    write_jsonl(out / "preferences.jsonl", prefs)
    write_jsonl(out / "kto.jsonl", kto)
    write_jsonl(out / "invalid_feedback.jsonl", invalid)
    summary = {"decisions": dict(counts), "corrections": len(corrections), "preference_pairs": len(prefs),
               "kto_examples": len(kto), "invalid": len(invalid)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for r in invalid:
        print(f"  invalid {r.get('review_id', '?')}: {'; '.join(r['_problems'])}")


if __name__ == "__main__":
    main()
