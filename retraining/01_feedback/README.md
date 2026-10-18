# 1. Reviewer feedback → training data

When a registered principal reviews a message with the model's help, log the model's answer and
what the principal decided. `feedback_to_data.py` turns that log into three kinds of training data.

```bash
.venv/bin/python retraining/01_feedback/feedback_to_data.py \
    --log retraining/01_feedback/sample/feedback_log.sample.jsonl --out retraining/01_feedback/sample/out
```

## Log format (one JSON object per line)

| Field | Required | Meaning |
|---|---|---|
| `review_id` | yes | unique id of the review |
| `timestamp` | yes | ISO 8601, UTC |
| `model_version` | yes | which model produced `model_output`, e.g. `phi4mini-finra2210-lora-mlx-v0.1.0` |
| `reviewer` | yes | pseudonymous reviewer id (no names) |
| `sample` | yes | the input as the model saw it: `channel`, `metadata` (`sender_role`, `firm`, `recipient_description`, `recipient_count_30d`, `recipient_type`, `subject`) and `communication` |
| `prompt` | no | the exact `[system, user]` messages sent; rebuilt from `sample` if absent |
| `model_output` | yes | the model's full answer text (prose + closing ```json verdict) |
| `decision` | yes | `accept`, `edit`, `reject`, `thumbs_up` or `thumbs_down` |
| `corrected_verdict` | for `edit` / `reject` | the right verdict, in the label format (`communication_type`, `compliant`, `violations[]`, `severity`, `required_actions`, `rationale`) |
| `difficulty` | no | if the reviewer rates it; otherwise `unrated` |
| `notes` | no | free text |

`edit` means the principal changed the verdict; `reject` means the model's verdict was wrong
enough to discard (usually a missed or invented violation). Thumbs alone carry no label.

## Outputs

| File | Built from | Used by |
|---|---|---|
| `corrections.jsonl` | `accept` (the model's verdict, confirmed) and `edit` / `reject` (the corrected verdict), in the clean-file format | `03_sft/build_round.py --add` |
| `preferences.jsonl` | `edit` / `reject` where the corrected verdict differs: `{prompt, chosen, rejected}` | `04_dpo/make_pairs.py --extra` |
| `kto.jsonl` | every decision with a model answer: `{prompt, completion, label}` with label true for accept / thumbs up | TRL `KTOTrainer` (unpaired feedback) |
| `invalid_feedback.jsonl` | corrections that fail the label checks, with the reasons | send back to the reviewer |
| `summary.json` | counts | |

Every corrected verdict goes through the same checks as the dataset (`finra2210.schema`):
excerpts must be verbatim, each category must carry its rule, categories must fit the
communication type, and `compliant` / `severity` / `violations` must agree. In the sample,
`rv-0005` has a paraphrased excerpt and is rejected for that reason.

## Sample (fake)

`sample/feedback_log.sample.jsonl` has six reviews of invented messages: one accept, two edits
(one valid, one with a paraphrased excerpt), one reject (a missed member-firm name), one thumbs
down and one thumbs up. `sample/out/` is what the script writes for it.

## Privacy

Real logs contain client communications. Keep them inside the firm's books-and-records systems,
use pseudonymous reviewer ids, and do not commit them to this repository.
