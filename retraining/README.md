# Retraining and feedback (RLHF)

How the FINRA 2210 reviewer gets better after v0.1.0: collect corrections, turn them into training
data, retrain, and ship the new model only if it beats the current one. Every stage has code, a
README, and small **fake** sample data that runs end to end:

```bash
retraining/run_examples.sh
```

| Stage | Folder | What it does | Runs on |
|---|---|---|---|
| 1 | [`01_feedback/`](01_feedback/) | turns reviewer accept / edit / reject logs into corrected labels, preference pairs and thumbs-up/down examples | any machine |
| 2 | [`02_audit/`](02_audit/) | picks the items where model and label disagree; audit agents (Claude Opus) decide who is right; fixes the labels | any machine + Claude Code |
| 3 | [`03_sft/`](03_sft/) | builds the next supervised round: label fixes + new samples, de-duplicated, re-split without moving old items | any machine, then the usual training pipeline |
| 4 | [`04_dpo/`](04_dpo/) | preference pairs from the model's own mistakes, and DPO training on top of the SFT model | pairs: any machine; training: NVIDIA GPU |
| 5 | [`05_grpo/`](05_grpo/) | an automatic reward computed from the labels, and GRPO reinforcement learning with it | reward: any machine; training: NVIDIA GPU |
| 6 | [`06_release_gate/`](06_release_gate/) | compares a candidate with the current model on test, OOD, chat, broad groups and difficulty; PASS / FAIL | any machine |

`run_examples.sh` rewrites the tracked `sample/out/` files and the gate reports; the scripts are
deterministic, so `git status` stays clean unless a script changed. Shared helpers are in
[`common.py`](common.py). All scripts run from the repository root with the
project venv (`.venv/bin/python ...`); the two training scripts need an NVIDIA GPU with
`requirements-train.txt` (see [`training/TRAINING_NVIDIA.md`](../training/TRAINING_NVIDIA.md)).

## The loop

```
 reviewers / audit agents / the checker
              │  corrections
              ▼
 01_feedback ─┬─► corrected labels ──┐
 02_audit ────┘                      ▼
                         03_sft  new round: fix labels, add samples, re-split (test stays fixed)
                                     │  retrain from the BASE model
                                     ▼
                         04_dpo  preference pairs from the new model's mistakes → DPO
                                     │
                                     ▼
                         05_grpo (optional) reward from the labels → GRPO
                                     │
                                     ▼
                         06_release_gate  vs the current release on test / OOD / chat / groups
                                     │  PASS
                                     ▼
                         release vN+1 (GitHub release, results/ regenerated)
```

1. **Collect corrections** from three sources:
   - **Reviewers (the "human" in RLHF).** A registered principal accepts, edits or rejects the
     model's verdict on a real message. Log what the model said and what the principal decided
     (format in [`01_feedback/README.md`](01_feedback/README.md)).
   - **A stronger model.** Claude Opus re-judges the cases where the model and the dataset label
     disagree ([`02_audit/`](02_audit/)). This is cheaper than reviewer time and is the way to start
     before any production feedback exists. A human should spot-check a sample of its rulings.
   - **The rules.** The checker already catches mechanical errors with no human at all: invalid
     JSON, an excerpt that is not in the text, a category that cannot apply to the communication
     type, a type that contradicts the recipient count. [`05_grpo/reward.py`](05_grpo/reward.py)
     turns these checks into a reward.
2. **Turn corrections into training data.** A corrected verdict becomes a new labeled example; the
   model's wrong answer next to the corrected one becomes a preference pair (preferred vs rejected).
3. **Retrain** with the methods below.
4. **Gate the release.** The new model must not get worse than the current one on the fixed test and
   OOD sets, on the chat-only slice, per broad group and per difficulty
   ([`06_release_gate/`](06_release_gate/)).
5. **Repeat** on a schedule, and whenever FINRA changes the rules: update
   `src/finra2210/taxonomy.py`, generate new scenarios for the affected categories, retrain.

## Training methods, in the order to use them

| Stage | Method | Training data | What it fixes | Cost |
|---|---|---|---|---|
| 1 | **Retrain with corrected data** (supervised fine-tuning) | corrected labels plus targeted new examples, e.g. pairs of messages that differ only in whether the firm name appears | label noise, overlapping categories, weak categories, too little chat | same as v0.1.0: ~13 h on an M5 Pro, ~2 h on an A100 |
| 2 | **DPO** (preference training) | preferred vs rejected answers; the rejected ones are the model's own mistakes on training prompts, so no reviewer data is needed | specific confusions such as `product_specific_omission` → `not_fair_and_balanced` and `misleading_in_context` → `false_or_misleading_statement` | about the same as stage 1 |
| 3 | **Reinforcement learning with automatic scores** (GRPO) | training prompts only; the model writes several answers per prompt and code scores each against the label | grounding errors (e.g. claiming the firm name is missing when it isn't), the balance between misses and false alarms | 4–8× more compute (several answers per prompt); NVIDIA GPU |
| 4 | **Classic RLHF** (a reward model trained on human preferences, then PPO) | large volumes of reviewer preferences | only judgments the labels cannot score | high; not worth it at this stage |

Stages 2 and 3 suit this project because every answer ends in a JSON verdict that code can score.
Most RLHF projects cannot do that and need a learned reward model.

**Tooling.** On NVIDIA, `src/finra2210/train_phi4.py` already uses TRL, which also has DPO, KTO (for
thumbs-up / thumbs-down feedback without pairs), GRPO and reward-model trainers; `04_dpo/train_dpo.py`
and `05_grpo/train_grpo.py` use them. On the Mac, `mlx-lm` only does supervised fine-tuning;
preference training there would need a third-party extension, so stages 2 and 3 are GPU-only here.

## Rules that matter in practice

- **Fix the labels before using RL.** RL amplifies whatever the reward rewards. If a gold label is
  missing a valid second category, the model is penalized for finding it. Audit the disagreements
  ([`02_audit/`](02_audit/)) and relabel so each item lists every 2210(d)(1) category that applies,
  before building DPO pairs or rewards.
- **Retrain from the base model, not on top of the last adapter.** For supervised rounds, train on
  everything (old and new data) so the model does not forget earlier skills. Apply DPO or GRPO
  afterwards as a separate step on that model, with a penalty for drifting too far from it (`--beta`).
- **Shape the score to the business risk.** Missing a high-severity violation (a guarantee, a
  projection) costs more than a false alarm: `reward.py` subtracts 0.5 when a high-severity item is
  called compliant. Otherwise the model learns that flagging less is safer.
- **Keep the test sets frozen.** The split keeps old items on their side when data is added
  (`--keep-test/--keep-train`), so test and OOD scores stay comparable across versions. Corrections
  found on test or OOD items go to `eval_label_fixes.jsonl`, never into training. Production
  feedback needs its own held-out slice.
- **Record which model version produced each review.** The feedback log carries `model_version`. If
  the model's output becomes part of the firm's supervision process, keep the version with the
  review. The registered principal stays the decision-maker.

## Roadmap

| Version | Change | Gate |
|---|---|---|
| v0.2 (SFT) | audit disagreements with Opus; relabel overlapping categories; add paired examples for `member_name_missing` and `product_specific_omission`; generate the planned 2,400-item chat top-up ([`docs/CHAT_TOPUP_PLAN.md`](../docs/CHAT_TOPUP_PLAN.md)); retrain from the base | beat v0.1.0 on test, OOD and chat |
| v0.3 (DPO) | preference pairs from v0.2's own mistakes on the training set. DPO and GRPO start from a PEFT adapter, so v0.2 must be trained on NVIDIA with `train_phi4.py` (the released v0.1.0 adapter is mlx-only) | beat v0.2 |
| v0.4 (GRPO, if needed) | the automatic reward in `05_grpo/reward.py` | beat v0.3 |
| ongoing | reviewer feedback in the `01_feedback` log format feeds v0.2+ rounds | every release |

## Status

All data-preparation code (stages 1, 2, 3, the DPO pairs, the reward and the gate) is tested on the
sample data by `run_examples.sh`. `train_dpo.py` and `train_grpo.py` are written against TRL's
DPOTrainer / GRPOTrainer (the TRL version in `requirements-train.txt`) but have not been run yet:
there is no NVIDIA GPU in this environment. Smoke-test them with `--limit` first.

The sample data is fake: invented firms ("... Example Securities LLC"), people and messages, with
ids starting `fake-` or `fb-rv-`. It only shows the file formats.
