# Instructions for an audit agent

You are a senior FINRA Advertising Regulation examiner auditing a labeled dataset. You were given
one task chunk, `retraining/02_audit/tasks/chunk_XXXX.json`. Each item holds a broker-dealer
communication, its distribution metadata, the dataset's label (`gold_label`) and the fine-tuned
model's verdict (`model_verdict`). The two disagree. Decide which one is right.

Working directory: the repository root.

1. Read `data/chunks_r1/BRIEF.md` in full. It has the Rule 2210 digest, the category → rule table,
   the communication-type rules and the labeling rules the dataset was written with. Read the
   category section of `MODEL_CARD.md` for what each category covers and does not cover.
2. For every item, read the whole communication and decide:
   - `gold`: the dataset label is right and the model is wrong.
   - `model`: the model's verdict is right and the label is wrong.
   - `neither`: both are wrong or incomplete. Write the correct `label` yourself.
   - `both_ok`: the difference is defensible either way, for example one extra
     2210(d)(1) category that also fits.
3. A `label` you write follows the labeling rules exactly:
   - keys `communication_type`, `compliant`, `violations`, `severity`, `required_actions`,
     `rationale`;
   - one entry per category, with the rule the table assigns to it;
   - each `excerpt` copied verbatim from the communication (at most 200 characters);
   - only categories that apply to the communication type;
   - `compliant` is true exactly when `violations` is empty and `severity` is `none`.

   List every category that applies, not only the most obvious one.
4. Write `retraining/02_audit/verdicts/chunk_XXXX.json` (same chunk number) as:

   ```json
   {"verdicts": [
     {"id": "...", "decision": "gold | model | neither | both_ok", "label": {...only for neither...},
      "reason": "one or two sentences citing the rule paragraph", "auditor": "<model id you run as>"}
   ]}
   ```

   One verdict per item, in the same order.
5. Do not edit anything else. In particular, never change `data/`, `src/` or other agents' files.
   Reply with one line: the chunk id, the number of verdicts, and how many of each decision.

`apply_audit.py` re-checks every label you write and lists the ones that fail in `out/invalid.jsonl`.
Fix those and run it again.
