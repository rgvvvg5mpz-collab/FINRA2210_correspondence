"""Helpers shared by the retraining stages (feedback, audit, SFT round, DPO, GRPO, release gate).

Every stage script puts the repo's src/ and this folder on sys.path, so run them from the repo root
with the project venv, e.g. `.venv/bin/python retraining/01_feedback/feedback_to_data.py --help`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from finra2210.evaluate import _parse  # noqa: E402
from finra2210.prompts import STUDENT_SYSTEM, render_answer, student_input  # noqa: E402
from finra2210.schema import validate_sample  # noqa: E402

LABEL_KEYS = ("communication_type", "compliant", "violations", "severity", "required_actions", "rationale")


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prompt_messages(sample: dict) -> list[dict]:
    """System + user turn exactly as training and evaluation build them (the first two messages).

    Uses sample["messages"][:2] when the sample already carries a chat conversation (split files),
    otherwise rebuilds the user turn from id / channel / metadata / communication.
    """
    if sample.get("messages"):
        return sample["messages"][:2]
    return [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": student_input(sample)}]


def answer_for(sample: dict, label: dict) -> str:
    """The reviewer answer in the training format (prose + closing ```json verdict) for `label`."""
    return render_answer({**sample, "label": label})


def parse_verdict(text: str) -> dict | None:
    """The verdict JSON from a model answer (last ```json block), as evaluate.py parses it."""
    return _parse(text)


def check_label(sample: dict, label: dict) -> list[str]:
    """Label-consistency problems (verbatim excerpts, cite per category, type vs recipients, ...)."""
    missing = [k for k in LABEL_KEYS if k not in label]
    if missing:
        return [f"label missing keys: {', '.join(missing)}"]
    return validate_sample({**sample, "label": label})


def categories(label: dict | None) -> set[str]:
    if not label:
        return set()
    return {v.get("category") for v in label.get("violations", []) if isinstance(v, dict)}


def same_verdict(a: dict | None, b: dict | None) -> bool:
    """True when two verdicts agree on compliance, communication type, severity and category set."""
    if not a or not b:
        return False
    return (bool(a.get("compliant")) == bool(b.get("compliant")) and a.get("communication_type") == b.get("communication_type")
            and a.get("severity") == b.get("severity") and categories(a) == categories(b))
