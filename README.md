# FINRA 2210 correspondence reviewer — dataset + Phi-4-mini fine-tune

Teachers: **Claude Fable 5.1** (`claude-fable-5-1`, pilot + first chunks) and **Claude Opus 5.5**
(`claude-opus-5-5`, the rest) mine labeled samples; every sample carries a `teacher` tag.
Student: **microsoft/Phi-4-mini-instruct** (3.8B) is LoRA-fine-tuned to act as a
first-pass reviewer of broker-dealer communications under FINRA Rule 2210.

One-page summary of what the model does and its classes: [MODEL_CARD.md](MODEL_CARD.md).
Training on an NVIDIA GPU (setup, data copy, commands, troubleshooting): [TRAINING_NVIDIA.md](TRAINING_NVIDIA.md).

## Round-1 dataset + targeted top-up (ready to train)

| File | Rows | Use |
|---|---|---|
| `data/splits/train.jsonl` | 10,710 | SFT (chat format, `messages` field) |
| `data/splits/test.jsonl` | 1,186 | in-domain held-out eval (10%, stratified by difficulty × comm type × primary category) |
| `data/splits/ood.jsonl` | 5,800 | out-of-distribution eval only (never trained on) |

- In-domain 11,896 (round-1 11,560 + top-up 336) / OOD 5,800 after validation and near-duplicate
  removal (0 rejects, 0 dupes; OOD was deduped against in-domain, and the top-up was deduped against
  OOD). Max 6-gram overlap between any two round-1 samples ≈ 0.38; top-up vs everything ≈ 0.24.
- Difficulty: in-domain 30% easy / 40% medium / 30% hard; ~44% compliant.
- Comm types (in-domain): correspondence 5,870, retail 4,567, institutional 1,459.
- OOD shift axes: product 2,716, channel 2,711, style 1,516, audience 1,229, regulatory twist 1,035, firm type 910.
- Every gold assistant turn parses back to its label (0 mismatches); longest conversation ≈ 2.2k tokens (< `--max-len 3072`).
- Targeted top-up for the sparse categories (train / test counts):

  | Category | Round-1 train | Now train | Now test |
  |---|---|---|---|
  | brokercheck_link_missing | 3 | 96 | 11 |
  | institutional_forwarding | 35 | 120 | 11 |
  | unapproved_retail_communication | 58 | 144 | 11 |

  336 scenarios from [topup.py](src/finra2210/topup.py) (seed 2211): 104 / 92 / 92 violation
  scenarios with the category forced as primary (easy/medium/hard mix, plus natural co-violations)
  and 16 compliant near-misses per category, each pinned to an `angle` (e.g. BrokerCheck named
  but not hyperlinked, white-label offer to an RIA, rep edited an approved piece). Written by
  Claude Opus 5.5 subagents into `data/agent_out_topup/in_domain/` and checked with `finra2210.check`.
  Round-1 records keep their ids (`in_domain-000000`…`011559`); top-up records are `011560`…`011895`.
  The split is incremental (`finra2210.split --keep-test/--keep-train`, pipeline step 4): all 1,158
  round-1 test ids are still in test and all 10,402 round-1 train ids still in train; only the 336
  top-up ids were stratified (28 test / 308 train). A model trained on the round-1 train.jsonl can
  therefore be evaluated on this test.jsonl without contamination. Round-1 membership is stored as
  id lists in `data/splits_r1_ids/`; the earlier full-reshuffle split (only 305 round-1 test ids kept
  in test) is backed up in `data/splits_reshuffled_20261017/`. Do not use it to evaluate round-1 models.
- Chat (sms_text / instant_message / text_message_blast) is 2,231 of 11,896 in-domain items (18.8%;
  229 of 1,186 test). The model is meant for chat, so a 2,400-scenario chat top-up is planned but not
  generated: [CHAT_TOPUP_PLAN.md](CHAT_TOPUP_PLAN.md) (counts per channel / type / category, cost,
  expected effect).
- Frozen raw copy: `data/snapshots/20261017T045434Z/` (round-1 + top-up; sha256 in `MANIFEST.json`;
  round-1 only: `20261017T044100Z/`); cleaned sets in `data/clean/`.

## The task the SLM learns

Input: a communication (email, letter, text, IM, ...) plus distribution metadata
(sender role, firm, recipient type, distinct recipients in trailing 30 days).

Output: one JSON object

```json
{
  "communication_type": "correspondence | retail_communication | institutional_communication",
  "compliant": false,
  "violations": [
    {"category": "performance_projection", "rule": "2210(d)(1)(F)",
     "excerpt": "should return 8-10% a year", "explanation": "..."}
  ],
  "severity": "none | low | medium | high",
  "required_actions": ["Return to rep for revision"],
  "rationale": "..."
}
```

20 violation categories, each pinned to one rule cite, are defined in
[taxonomy.py](src/finra2210/taxonomy.py). The same file holds the rule digest the
teacher is grounded on, the difficulty definitions, and the scenario axes.

## Difficulty tiers

| Tier | Violation samples | Compliant samples |
|---|---|---|
| easy (30%) | one blatant violation (guarantee, numeric forecast, "FINRA approved") | routine logistics, no claims |
| medium (40%) | needs the whole message: implied projection, omitted risks, bad comparison, fund perf without standardized returns, template that metadata reveals went to >25 | discusses performance/products/recs **correctly** with required disclosures |
| hard (30%) | compound or borderline: true-but-misleading-together, buried footnote, institutional forwarding, backtest-as-illustration, threshold × (d)(7) interaction, hedged projections | looks aggressive but is compliant; tempting-but-wrong cites (e.g. (d)(7) on correspondence) |

About 45% of samples are compliant so the model can't learn "mentions returns ⇒ violation".

## In-domain vs OOD

- **In-domain (10K → train/test 90/10, stratified):** email/letter/SMS/IM from retail BD reps about funds, ETFs, stocks, bonds, retirement accounts, 529s, fixed annuities, munis.
- **OOD (5K, eval only):** every sample shifts on 1–3 axes — channel (LinkedIn DM, WhatsApp, chatbot transcript, voicemail, long threads, OCR noise), product (options, crypto ETFs, Reg D, structured notes, VAs, leveraged ETFs, non-traded REITs, SPACs, margin, CMOs), audience (institutional-that-forwards, plan sponsor vs participant, $40M family office, Spanish-language), style, firm type, and regulatory twists (2210(f), 2212, 2213, 2214, 2241, SIPC, Reg BI, 3240, RN 24-09). The shift axes are recorded per sample so eval can break down by axis.

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH=src
```

### Path A — generate inside Claude Code (Fable subagents, no API key)

This is how the first-round data was produced. Scenario chunks of 40 are written
to disk; each Claude Code subagent reads `data/chunks/BRIEF.md` plus one chunk,
writes `data/agent_out/<split>/chunk_XXXX.json`, runs `finra2210.check` on its own
output, fixes rejects, and emits `chunk_XXXX.raw.jsonl`.

```bash
.venv/bin/python -m finra2210.chunks --n 11520 --per-chunk 40 --out data/chunks/in_domain
```

```bash
.venv/bin/python -m finra2210.chunks --n 5760 --per-chunk 40 --ood --seed 99 --out data/chunks/ood
```

```bash
.venv/bin/python -m finra2210.progress data/chunks/in_domain data/agent_out/in_domain --next 10
```

```bash
.venv/bin/python -m finra2210.progress data/chunks/in_domain data/agent_out/in_domain --assemble data/in_domain.raw.jsonl
```

To freeze everything validated so far into a checksummed, append-only copy
(`data/snapshots/<UTC timestamp>/{in_domain,ood}.raw.jsonl` + `MANIFEST.json`):

```bash
PYTHONPATH=src .venv/bin/python -m finra2210.snapshot
```

Then continue at step 3/4 below (postprocess, split).

Targeted top-up for sparse categories (same agent loop, instructions in
`data/chunks_topup/AGENT_INSTRUCTIONS.md`, output in `data/agent_out_topup/in_domain/`, which
`finra2210.snapshot` picks up):

```bash
PYTHONPATH=src .venv/bin/python -m finra2210.topup --out data/chunks_topup/in_domain
```

```bash
PYTHONPATH=src .venv/bin/python -m finra2210.postprocess --raw data/snapshots/<latest>/in_domain.raw.jsonl --out data/clean/in_domain.clean.jsonl --exclude data/clean/ood.clean.jsonl
```

### Path B — generate through the API (`ANTHROPIC_API_KEY` required)

1. Pilot (live, ~48 samples, a few dollars) — read `data/pilot.clean.jsonl` before spending on the full run:

```bash
.venv/bin/python -m finra2210.generate --n 48 --mode live --out data/pilot
```

```bash
.venv/bin/python -m finra2210.postprocess --raw data/pilot.raw.jsonl --out data/pilot.clean.jsonl
```

2. Full in-domain run via the Batch API (50% price). Over-generate ~15% to absorb validation/dedup rejects:

```bash
.venv/bin/python -m finra2210.generate --n 11500 --mode batch --out data/in_domain
```

```bash
.venv/bin/python -m finra2210.generate --mode batch --collect --out data/in_domain
```

```bash
.venv/bin/python -m finra2210.postprocess --raw data/in_domain.raw.jsonl --out data/in_domain.clean.jsonl
```

3. OOD run (different seed, OOD sampler, near-dups against in-domain are dropped):

```bash
.venv/bin/python -m finra2210.generate --n 5800 --mode batch --ood --seed 99 --out data/ood
```

```bash
.venv/bin/python -m finra2210.generate --mode batch --collect --out data/ood
```

```bash
.venv/bin/python -m finra2210.postprocess --raw data/ood.raw.jsonl --out data/ood.clean.jsonl --exclude data/in_domain.clean.jsonl
```

4. Split and export to Phi-4-mini chat format:

```bash
.venv/bin/python -m finra2210.split --clean data/in_domain.clean.jsonl --out data/splits --test-frac 0.10
```

```bash
.venv/bin/python -m finra2210.split --clean data/ood.clean.jsonl --out data/splits --name ood
```

   After appending new samples to the clean file (e.g. a top-up), a plain rerun reshuffles every
   stratum, so old samples change sides. To keep a model trained on the previous `train.jsonl`
   evaluable, pass the previous split: every id in `--keep-test` stays in test, every id in
   `--keep-train` stays in train, and only unseen ids are stratified (each stratum's test share is
   topped up to `--test-frac` from its new items). Both flags take a split JSONL or a plain id list
   (one id per line) and must be given together; back up `data/splits` (or write to a fresh `--out`)
   before replacing it. The manifest records `kept_test` / `kept_train` / `new_test` / `new_train`.

```bash
.venv/bin/python -m finra2210.split --clean data/clean/in_domain.clean.jsonl --out data/splits --keep-test data/splits_r1_ids/test.ids --keep-train data/splits_r1_ids/train.ids
```

   That is the command that produced the current `data/splits`. For the next top-up, copy the
   current `data/splits/test.jsonl` / `train.jsonl` aside and pass those instead of the round-1 id
   lists. The round-1 id lists are reproducible: run the plain split on the first 11,560 lines of
   `data/clean/in_domain.clean.jsonl`.

5. Fine-tune and 6. evaluate: on an NVIDIA GPU (next section), or on Apple Silicon with mlx-lm
   (the section after).

### NVIDIA GPU (CUDA): steps 5–6

Full step-by-step guide, including how to copy the gitignored `data/splits` to the GPU machine
and check them: [TRAINING_NVIDIA.md](TRAINING_NVIDIA.md). Summary:

PyTorch + TRL LoRA on the full-precision base `microsoft/Phi-4-mini-instruct`
([train_phi4.py](src/finra2210/train_phi4.py)). The recipe matches the mlx run, so results are
comparable:
- **Validation:** the same stratified 5% holdout of `train.jsonl` (same 539 ids). `test.jsonl` is
  never seen during training.
- **Loss:** only on the reviewer's turns. Each conversation becomes one prompt/completion
  example per assistant turn, and the prompt is masked.
- **LoRA:** rank 32 / alpha 64 / dropout 0.05 on `qkv_proj`, `o_proj`, `gate_up_proj` and
  `down_proj` in every block.
- **Optimizer:** lr 2e-4 cosine with 3% warmup, effective batch 16, 2 epochs.
- **Checkpoints:** a validation run and a checkpoint every 100 optimizer steps. The saved adapter
  is the one with the lowest validation loss.

**Hardware.** Weights are ~7.7 GB in bf16.

| GPU memory | Setting |
|---|---|
| 24 GB+ (RTX 3090/4090, A10, L4, A100, H100) | defaults: bf16, batch 4 × grad-accum 4 |
| 16 GB | `--batch 2 --grad-accum 8` |
| 12–16 GB | `--qlora`: 4-bit NF4 base via bitsandbytes, Linux only |

Ampere and newer GPUs train in bf16; T4 and V100 fall back to fp16 automatically.

**Time:** a rough estimate (not measured) is 1.5–2 h on one A100 and 3–5 h on a 24 GB card,
for about 1,660 optimizer steps. Evaluating one model on test + OOD (6,986 rows) is the larger
cost.

Setup, in a Python 3.11 or 3.12 venv (Linux or Windows/WSL with a recent NVIDIA driver):

```bash
python3.12 -m venv .venv-cuda && .venv-cuda/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv-cuda/bin/pip install -r requirements-train.txt
```

`bitsandbytes` is only needed for `--qlora` (`.venv-cuda/bin/pip install bitsandbytes`), and
`flash-attn` only for `--attn flash_attention_2`. Match the wheel index (`cu128` above) to your
driver's CUDA version (`nvidia-smi`). The base model is public (MIT licence) and downloads from
Hugging Face on first use.

Everything at once (train → loss report → eval tuned → eval zero-shot base), resumable like the
mlx script:

```bash
nohup scripts/run_cuda_pipeline.sh > runs/pipeline_cuda.log 2>&1 &
```

The [run_cuda_pipeline.sh](scripts/run_cuda_pipeline.sh) script is configured with environment
variables:
- `STAGES="eval_tuned eval_base"` runs a subset of the stages.
- `TRAIN_ARGS="--qlora"` or `TRAIN_ARGS="--batch 2 --grad-accum 8"` passes extra training flags.
- `NGPU=2` trains with DDP across 2 GPUs. The effective batch scales with the GPU count, so
  halve `--grad-accum` to keep it at 16.
- `EVAL_BATCH=16` sets the eval batch size.
- `PY=...` selects the Python interpreter (default `.venv-cuda/bin/python`).

Outputs in `runs/phi4mini-2210/`:
- the adapter, `train_args.json` and `valid.ids`;
- `trainer_state.json`, from which `loss.json`, `loss_curve.png` and `training_summary.md` are
  built;
- `eval_test.json` and `eval_ood.json`.

The zero-shot baseline goes in `runs/base-hf/`.

Or run the steps one by one:

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.train_phi4 --train data/splits/train.jsonl --out runs/phi4mini-2210
```

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.loss_report --hf-state runs/phi4mini-2210/trainer_state.json
```

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.evaluate --backend hf --model runs/phi4mini-2210 --data data/splits/test.jsonl --batch 16 --resume --out runs/phi4mini-2210/eval_test.json
```

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.evaluate --backend hf --model runs/phi4mini-2210 --data data/splits/ood.jsonl --batch 16 --resume --out runs/phi4mini-2210/eval_ood.json
```

```bash
PYTHONPATH=src .venv-cuda/bin/python -m finra2210.evaluate --backend hf --model microsoft/Phi-4-mini-instruct --data data/splits/test.jsonl --batch 16 --resume --out runs/base-hf/eval_test.json
```

Multi-GPU training directly:

```bash
PYTHONPATH=src .venv-cuda/bin/accelerate launch --num_processes 2 -m finra2210.train_phi4 --train data/splits/train.jsonl --out runs/phi4mini-2210 --grad-accum 2
```

Notes:
- `evaluate.py --model` accepts the adapter directory (it reads the base model from
  `adapter_config.json` and merges in memory) or a merged model.
- `train_phi4.py --merge` also writes a standalone merged model to `runs/phi4mini-2210/merged`
  for serving, e.g. with vLLM. It is skipped with `--qlora`: merge onto a bf16 base instead.
- A quick smoke test (40 conversations, 1 epoch) checks the environment before the full run:
  `--limit 40 --epochs 1 --eval-steps 4 --out runs/smoke`.
- Tested end to end with torch 2.14.1, transformers 5.20.1, TRL 1.16.0 and PEFT 0.21.2, on CPU
  with a tiny random Phi-3-architecture model and the real Phi-4-mini tokenizer. Not yet run on
  an NVIDIA GPU.

Every eval report (either backend) has overall, by-difficulty and by-communication-type metrics.
It also has `by_channel_group`, `chat_by_communication_type` and `by_channel`, so chat
performance is never averaged away by the email/web majority. The channel groups are:
- `chat`: sms_text, instant_message and text_message_blast;
- `chat_app`: the OOD-only WhatsApp, LinkedIn, X and Slack/Teams DMs;
- `non_chat`: everything else.

### Apple Silicon (mlx-lm): steps 5–6

No CUDA needed; QLoRA on the 4-bit base `mlx-community/Phi-4-mini-instruct-4bit`, same LoRA targets,
rank 32 / alpha 64, lr 2e-4 cosine, effective batch 16, 2 epochs, loss on assistant turns only.

```bash
/opt/homebrew/bin/python3.14 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-lm matplotlib
```

```bash
caffeinate -i scripts/run_mlx_pipeline.sh
```

The script runs five stages (`STAGES="data train report eval_tuned eval_base"` selects a subset):
- `data`: [mlx_data.py](src/finra2210/mlx_data.py) holds out a stratified 5% of `train.jsonl` as
  validation (539 conversations; ids in `data/mlx/valid.ids`), so `test.jsonl` is only used for the
  final eval. mlx-lm's `--mask-prompt` trains only the last assistant message, so each 5-turn
  conversation becomes two examples (one ending at each assistant turn): 13,271 train / 705 valid.
- `train`: `mlx_lm lora -c` [configs/phi4mini_lora_mlx.yaml](configs/phi4mini_lora_mlx.yaml) →
  adapter in `runs/phi4mini-2210-mlx/` (checkpoint and full validation loss every 400 iterations).
  Gradient checkpointing is required on a 24 GB M5 Pro (it runs out of GPU memory without it);
  there it trains at ~7 s per iteration of 4 examples, ~13 h for 2 epochs.
- `report`: [loss_report.py](src/finra2210/loss_report.py) parses `train.log` into
  `loss.json`, `loss_curve.png` (log-scale train/validation loss, epoch boundaries, spikes, best
  checkpoint) and `training_summary.md` (best/last validation loss, overfitting check, train/val gap,
  epoch step-down, spikes). It can also be run by hand mid-training.
- `eval_tuned` / `eval_base`: `evaluate.py --backend mlx` on all of test and OOD for the adapter
  and for the zero-shot 4-bit base (`runs/base-mlx/`), ~1.4 s per row. Batch 16, retried at 8 then 4 if the GPU runs out of memory. Predictions are
  appended as they are produced, and `--resume` continues an interrupted eval.

## Quality controls

- Scenario specs (difficulty, compliance target, categories, channel, product, sender, recipient count/type, tone, length) are sampled **before** the teacher writes anything, so diversity is controlled, not hoped for.
- [schema.py](src/finra2210/schema.py) rejects samples where: excerpts are not verbatim in the text; `compliant`/`violations`/`severity` disagree; `communication_type` contradicts the recipient metadata (≤25 retail ⇒ correspondence, >25 ⇒ retail, institutional-only ⇒ institutional); a category is cited for a type it cannot apply to; the sample drifted from its scenario.
- Near-duplicates (6-word shingle overlap > 0.5) are dropped; OOD is additionally de-duplicated against in-domain.
- Teacher output is structured (JSON schema enforced) with the rule digest in a cached system prompt.

## Files

```
src/finra2210/
  taxonomy.py    rule digest, categories→cites, difficulty guide, scenario axes
  scenarios.py   in-domain and OOD scenario sampler
  topup.py       targeted top-up scenarios for sparse categories
  chat_topup.py  chat (SMS / IM / text blast) top-up scenarios; --stats prints the plan tables
  prompts.py     teacher prompt; student prompt/target format
  schema.py      JSON schema + label-consistency validator
  generate.py    Fable generation (live / batch, resumable)
  postprocess.py validate + dedup + stats
  split.py       stratified train/test; chat-format export
  train_phi4.py  LoRA SFT for Phi-4-mini-instruct (PyTorch / NVIDIA CUDA)
  mlx_data.py    train/valid data for mlx-lm LoRA (Apple Silicon)
  loss_report.py loss curve + training summary (mlx-lm train.log or PyTorch trainer_state.json)
  evaluate.py    metrics by difficulty, comm type, channel / chat group, OOD axis (--backend hf | mlx)
configs/phi4mini_lora_mlx.yaml   mlx-lm LoRA config
scripts/run_mlx_pipeline.sh      data → train → report → eval tuned → eval base (mlx)
scripts/run_cuda_pipeline.sh     train → report → eval tuned → eval base (NVIDIA / PyTorch)
data/            generated artifacts (gitignored-worthy)
```

## Caveats

- Labels are synthetic teacher labels, not regulator decisions. Spot-check `*.rejected.jsonl` and a random slice of accepted samples with a compliance SME before trusting the test numbers.
- The rule digest is a working summary; verify against the current rule text at finra.org before production use. FINRA amends 2210 periodically (e.g. the projected-performance exception for institutional / qualified-purchaser communications).
