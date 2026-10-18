"""Review one communication with the trained model and print its answer and verdict.

  PYTHONPATH=src .venv-mlx/bin/python -m finra2210.review --model runs/phi4mini-2210-mlx \
      --channel sms_text --recipients 3 --firm "Harbor Point Securities LLC" \
      --text "Hi all, the Atlas Income Fund is a can't-lose 7% a year. Want in?"

  # from a file, NVIDIA backend, JSON only
  PYTHONPATH=src python -m finra2210.review --backend hf --model runs/phi4mini-2210 --file email.txt --json

The user turn is built exactly as in training (finra2210.prompts.student_input), from the text and
the distribution metadata. The recipient count decides correspondence (<= 25 retail recipients in
30 days) vs retail communication, so pass the real number. Every verdict needs review by a
registered principal.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .evaluate import _load_hf, _load_mlx, _parse
from .prompts import STUDENT_SYSTEM, student_input


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="adapter dir (e.g. runs/phi4mini-2210-mlx) or a merged model")
    ap.add_argument("--backend", choices=["mlx", "hf"], default="mlx")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="the communication text")
    src.add_argument("--file", help="read the communication from this file ('-' for stdin)")
    ap.add_argument("--channel", default="email", help="email, letter, sms_text, instant_message, social_media_post, website_page, ...")
    ap.add_argument("--recipients", type=int, default=1, help="distinct recipients in the trailing 30 days")
    ap.add_argument("--recipient-type", choices=["retail", "institutional"], default="retail")
    ap.add_argument("--recipient-description", default=None, help='e.g. "an existing retail client"')
    ap.add_argument("--sender-role", default="registered representative")
    ap.add_argument("--firm", default="the member firm")
    ap.add_argument("--subject", default="")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--json", action="store_true", help="print only the parsed verdict as JSON")
    args = ap.parse_args()

    text = args.text if args.text is not None else (sys.stdin.read() if args.file == "-" else Path(args.file).read_text())
    who = args.recipient_description or (
        "institutional investors" if args.recipient_type == "institutional"
        else "a retail client" if args.recipients == 1 else f"{args.recipients} retail clients")
    sample = {"id": "review", "channel": args.channel, "communication": text.strip(),
              "metadata": {"sender_role": args.sender_role, "firm": args.firm, "recipient_description": who,
                           "recipient_count_30d": args.recipients, "recipient_type": args.recipient_type,
                           "subject": args.subject}}
    messages = [{"role": "system", "content": STUDENT_SYSTEM},
                {"role": "user", "content": student_input(sample, random.Random(0))}]
    generate = (_load_mlx if args.backend == "mlx" else _load_hf)(args.model)
    answer = generate([messages], args.max_new_tokens)[0]
    verdict = _parse(answer)
    if args.json:
        print(json.dumps(verdict, indent=2, ensure_ascii=False))
    else:
        print(answer)
    if verdict is None:
        print("\n[no parsable JSON verdict in the answer]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
