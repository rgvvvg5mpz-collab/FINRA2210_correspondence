# FINRA Rule 2210 communications reviewer: dataset + Phi-4-mini fine-tune

A small language model that does a **first-pass compliance review of broker-dealer
communications under FINRA Rule 2210**: emails, letters, text messages, instant messages, social
posts, web pages, seminar scripts, newsletters. For each message it says whether it is
correspondence, a retail communication or an institutional communication; whether it complies;
which violations it contains, each with the rule paragraph and the exact offending words; how
serious they are; and what the reviewer should do next.

- **Model:** `microsoft/Phi-4-mini-instruct` (3.8B) with a LoRA adapter, trained on Apple Silicon
  with mlx-lm (an NVIDIA / PyTorch path with the same recipe is included).
- **Data:** 11,896 in-domain + 5,800 out-of-distribution synthetic, labeled communications,
  written by Claude Code subagents (Claude Opus 5.5 for 98%, Claude Fable 5.1 for the pilot) from
  specs sampled in code, checked by code, de-duplicated.
- **Release:** [v0.1.0](https://github.com/rgvvvg5mpz-collab/FINRA2210_correspondence/releases/tag/v0.1.0)
  (adapter + dataset archive). Summary: [MODEL_CARD.md](MODEL_CARD.md) ([HTML](MODEL_CARD.html)).

> **An assistant, not a decision-maker.** Labels are synthetic teacher judgments, not regulator
> decisions, and have not been reviewed by a compliance professional. Every verdict needs review by
> a registered principal.

## Results at a glance

| Metric | Fine-tuned, test (1,186) | Fine-tuned, OOD (5,800) | Zero-shot base, test |
|---|---|---|---|
| Valid JSON verdict | **99.8%** | **99.7%** | 67.8% |
| Compliant / not compliant correct | **96.5%** | **95.3%** | 39.2% |
| Communication type correct | **99.8%** | **98.6%** | 0.0% |
| Severity correct | **89.5%** | **86.4%** | 1.2% |
| Violation detection F1 | **0.97** | **0.96** | 0.44 |
| Violation category F1, exact | **0.80** | **0.69** | 0.00 |
| Violation category F1, broad group | **0.92** | **0.88** | 0.00 |

![Fine-tuned vs base](results/plots/headline_tuned_vs_base.png)

The model is reliable at the verdict and the communication type, including out of distribution.
Its weakness is naming the exact category among overlapping ones (for example
`product_specific_omission` vs `not_fair_and_balanced`, both 2210(d)(1)(A)) and noticing that
something is missing (`member_name_missing`). The zero-shot base writes labels in its own words, so
its type and category scores are near zero. Full results, per-category tables, confusion matrices
and breakdowns by difficulty, chat channel and OOD shift: **[results/README.md](results/README.md)**.

## Repository layout

```
├── README.md                 this file
├── MODEL_CARD.md / .html     what the model does, all 23 categories documented, data, training, results
├── data/                     the dataset (train / test / OOD), its README, and the sources it was curated from
├── src/finra2210/            the Python package: taxonomy, scenario sampler, checker, post-processing,
│                             split, prompts, training, evaluation, single-message review
├── training/                 training pipelines (mlx-lm and NVIDIA), config, NVIDIA guide
├── evaluation/               live confusion-matrix dashboard, results builder, metric definitions
├── results/                  v0.1.0 metrics, plots, predictions, training curve
├── retraining/               feedback, label audit, SFT rounds, DPO, GRPO, release gate (code + sample data)
├── docs/                     chat top-up plan, model-card HTML builder
├── requirements.txt          generation and evaluation (project venv)
└── requirements-train.txt    NVIDIA training
```

| Folder | README |
|---|---|
| `data/` | [data/README.md](data/README.md): files, record format, how the data is used, how it was curated, distributions |
| `training/` | [training/README.md](training/README.md): recipe, Apple Silicon and NVIDIA pipelines; [TRAINING_NVIDIA.md](training/TRAINING_NVIDIA.md) |
| `evaluation/` | [evaluation/README.md](evaluation/README.md): eval sets, metrics, dashboard, building `results/` |
| `results/` | [results/README.md](results/README.md): every number and plot for v0.1.0 |
| `retraining/` | [retraining/README.md](retraining/README.md): how the model improves (RLHF, DPO, GRPO, audits, release gate) |
| `docs/` | [docs/CHAT_TOPUP_PLAN.md](docs/CHAT_TOPUP_PLAN.md): the planned 2,400-item chat top-up |

`runs/` (adapters, checkpoints, logs) is local and ignored by git.

## Quick start

### Install

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
python3 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-lm matplotlib markdown-it-py
```

`.venv` runs the data tools, the evaluation scoring, the dashboard and the retraining scripts.
`.venv-mlx` (Apple Silicon) runs the model. For NVIDIA see
[training/TRAINING_NVIDIA.md](training/TRAINING_NVIDIA.md).

### Use the model

Download `phi4mini-finra2210-lora-mlx-v0.1.0.tar.gz` from the
[v0.1.0 release](https://github.com/rgvvvg5mpz-collab/FINRA2210_correspondence/releases/tag/v0.1.0),
check it against its `.sha256` file, and extract the adapter folder (here `runs/phi4mini-2210-mlx`).
Then review a message:

```bash
PYTHONPATH=src .venv-mlx/bin/python -m finra2210.review --model runs/phi4mini-2210-mlx --channel sms_text --recipients 3 --firm "Harbor Point Securities LLC" --text "Hi all, the Atlas Income Fund is a can't-lose 7% a year. Want in? - Dana" --json
```

```json
{
  "communication_type": "correspondence",
  "compliant": false,
  "violations": [
    {"category": "promissory_or_guarantee", "rule": "2210(d)(1)(B)",
     "excerpt": "the Atlas Income Fund is a can't-lose 7% a year",
     "explanation": "A fixed annual return for a mutual fund is a promissory claim; no fund can guarantee a return."}
  ],
  "severity": "high",
  "required_actions": ["Escalate to supervising principal under Rule 3110(b)",
                       "Send corrective text to all three recipients withdrawing the 'can't-lose' claim"],
  "rationale": "A text to three retail clients is correspondence. Calling a mutual fund 'can't-lose 7%' is a promissory claim prohibited by 2210(d)(1)(B)."
}
```

That is a real output (about 9 s on an M5 Pro). It also shows the known weakness: the text never
names the member firm, so `member_name_missing` (2210(d)(3)) should have been flagged too.

`--recipients` matters: 25 or fewer retail recipients in 30 days is correspondence, more is a retail
communication. Without `--json` the command prints the full review in prose. `--backend hf` runs
a PyTorch adapter or merged model. In your own code, build the messages the same way,
`[{"role": "system", "content": finra2210.prompts.STUDENT_SYSTEM}, {"role": "user", "content": finra2210.prompts.student_input(sample)}]`,
and parse the answer with `finra2210.evaluate._parse`.

### Train

```bash
caffeinate -i training/run_mlx_pipeline.sh
```

Prepares the mlx data, trains (about 13 h on a 24 GB M5 Pro), writes the loss report, and evaluates
the adapter and the zero-shot base on test and OOD. Details and the NVIDIA pipeline:
[training/README.md](training/README.md).

### Evaluate and build the results

```bash
PYTHONPATH=src .venv-mlx/bin/python -m finra2210.evaluate --backend mlx --model runs/phi4mini-2210-mlx --data data/splits/test.jsonl --batch 16 --resume --out runs/phi4mini-2210-mlx/eval_test.json
```

```bash
.venv/bin/python evaluation/live_confusion.py
```

```bash
PYTHONPATH=src .venv-mlx/bin/python evaluation/make_results.py
```

The first scores a model, the second serves a live dashboard at http://localhost:8765 while an eval
runs, the third rebuilds `results/`. See [evaluation/README.md](evaluation/README.md).

## The task

**Input:** the communication text plus its distribution metadata: channel, sender role, firm,
recipient description, recipient type (retail / institutional) and the number of distinct
recipients in the trailing 30 days.

**Output:** a short review in prose that ends with one JSON verdict:

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

### Violation categories

23 categories, each pinned to one rule cite, in 5 broad groups (`VIOLATION_CATEGORIES`,
`BROAD_GROUPS` and `SUBGROUPS` in [taxonomy.py](src/finra2210/taxonomy.py)). Every category is
documented in the [model card](MODEL_CARD.md#violation-categories) with what it flags, what it does
not, an example and its test metrics.

| Broad group (key) | Category labels |
|---|---|
| Misleading or unbalanced content — 2210(d)(1) (`misleading_content`) | `not_fair_and_balanced`, `product_specific_omission`, `buried_disclosure`, `promissory_or_guarantee`, `exaggerated_or_unwarranted_claim`, `false_or_misleading_statement`, `misleading_in_context`, `performance_projection`, `hypothetical_or_backtest_misuse`, `third_party_content_adoption` |
| Missing required disclosures — 2210(d)(2)-(d)(8), 2212 (`required_disclosures`) | `unsupported_comparison`, `member_name_missing`, `tax_claim_incomplete`, `fund_performance_disclosure`, `testimonial_disclosure`, `recommendation_disclosure`, `past_recommendations_reference`, `brokercheck_link_missing`, `fund_ranking_disclosure` |
| Approval and audience — 2210(a), (b) (`approval_and_audience`) | `unapproved_retail_communication`, `institutional_forwarding` |
| Implied FINRA / SIPC endorsement — 2210(e) (`regulator_endorsement`) | `finra_endorsement_implied` |
| Product-specific rules — 2220 (`product_rules`) | `options_communication` |

`third_party_content_adoption` and `options_communication` appear only in the OOD set.

## Data

| Split | Items | Use |
|---|---|---|
| [`data/splits/train.jsonl`](data/splits/) | 10,710 | fine-tuning (chat format, `messages` field); 5% held out for validation |
| [`data/splits/test.jsonl`](data/splits/) | 1,186 | in-domain held-out evaluation (10%, stratified by difficulty × communication type × primary category) |
| [`data/splits/ood.jsonl`](data/splits/) | 5,800 | out-of-distribution evaluation only |

- **Composition (in-domain):** correspondence 5,870, retail 4,567, institutional 1,459; 44%
  compliant; difficulty 30% easy / 40% medium / 30% hard; 16 channels from email (4,246) to
  video scripts (102); chat channels (SMS, IM, text blast) 18.8%.
- **Difficulty tiers:** easy = one blatant violation, or routine logistics; medium = needs the whole
  message (implied projection, omitted risk, fund performance without standardized returns, a
  template that the metadata shows went to more than 25 people), or a compliant message that
  discusses performance correctly; hard = compound or borderline violations, or compliant
  near-misses that look aggressive but pass.
- **OOD:** every item is shifted on 1–3 axes: channel (WhatsApp, LinkedIn, chatbot transcripts,
  voicemail, OCR'd handwriting, ...), product (options, crypto ETFs, Reg D, structured notes,
  leveraged ETFs, non-traded REITs, ...), audience, style, firm type and regulatory twist (2210(f),
  2212–2214, 2241, Reg BI, RN 24-09 on AI-written content).

Record formats, distributions, every category's count, and the full curation process:
**[data/README.md](data/README.md)**.

## Data pipeline

How the dataset was generated, inside Claude Code without an API key (Path A). Each step is
re-runnable; all commands run from the repository root with `PYTHONPATH=src`.

1. **Sources.** Rule 2210, the related rules, the relevant Regulatory Notices, the 2210 FAQ and SEC
   Rules 482 / 34b-1 were archived in `data/raw_finra/` ([SOURCES.md](data/raw_finra/SOURCES.md))
   and distilled into the rule digest and taxonomy in `src/finra2210/taxonomy.py`.
2. **Scenario specs, sampled in code** (difficulty, compliance target, categories, channel,
   product, sender, recipient count and type, tone, length; OOD shift axes):

   ```bash
   .venv/bin/python -m finra2210.chunks --n 11520 --per-chunk 24 --out data/chunks_r1/in_domain
   ```

   ```bash
   .venv/bin/python -m finra2210.chunks --n 5760 --per-chunk 24 --ood --seed 99 --out data/chunks_r1/ood
   ```

   The sampler was corrected during the run (two scenario combinations were fixed), so re-running
   it today gives slightly different specs. The exact spec behind every item is stored in its
   `scenario` field in `data/clean/`.
3. **Writing.** Up to 20 Claude Code subagents in parallel, each given one chunk,
   [`data/chunks_r1/BRIEF.md`](data/chunks_r1/BRIEF.md) and
   [`AGENT_INSTRUCTIONS.md`](data/chunks_r1/AGENT_INSTRUCTIONS.md), write
   `data/agent_out_r1/<split>/chunk_XXXX.json` and loop the checker until it reports no problems:

   ```bash
   .venv/bin/python -m finra2210.check data/chunks_r1/in_domain/chunk_0000.json data/agent_out_r1/in_domain/chunk_0000.json --emit data/agent_out_r1/in_domain/chunk_0000.raw.jsonl --teacher claude-opus-5-5
   ```

   Progress and the next chunks to assign: `.venv/bin/python -m finra2210.progress data/chunks_r1/in_domain data/agent_out_r1/in_domain --next 20`.
4. **Snapshot** every validated record into a timestamped, checksummed, append-only copy:
   `.venv/bin/python -m finra2210.snapshot` writes `data/snapshots/<UTC>/` with `MANIFEST.json`.
5. **Validate and de-duplicate** (exact and 6-word-shingle near-duplicates; OOD against in-domain):

   ```bash
   .venv/bin/python -m finra2210.postprocess --raw data/snapshots/<latest>/ood.raw.jsonl --out data/clean/ood.clean.jsonl
   ```

   ```bash
   .venv/bin/python -m finra2210.postprocess --raw data/snapshots/<latest>/in_domain.raw.jsonl --out data/clean/in_domain.clean.jsonl --exclude data/clean/ood.clean.jsonl
   ```

6. **Targeted top-up** of sparse categories (336 items, same agent loop, `data/chunks_topup/`):
   `.venv/bin/python -m finra2210.topup --out data/chunks_topup/in_domain`.
7. **Split and export to chat format.** The split is incremental: items already in train or test
   stay there, and only new ids are stratified, so a model trained on an earlier `train.jsonl`
   stays uncontaminated on the new `test.jsonl`:

   ```bash
   .venv/bin/python -m finra2210.split --clean data/clean/in_domain.clean.jsonl --out data/splits --keep-test data/splits_r1_ids/test.ids --keep-train data/splits_r1_ids/train.ids
   ```

   ```bash
   .venv/bin/python -m finra2210.split --clean data/clean/ood.clean.jsonl --out data/splits --name ood
   ```

**Quality controls.** Specs are sampled before anything is written, so diversity and label balance
are controlled rather than left to the writer. The checker and `schema.py` reject an item if an
excerpt is not verbatim in the text; `compliant`, `violations` and `severity` disagree; the
communication type contradicts the recipient metadata; a category is cited for a type it cannot
apply to; or the item drifted from its spec. Result: 0 rejects and 0 near-duplicates; every gold
answer parses back to its label.

**Path B (not used for v0.1.0).** `src/finra2210/generate.py` writes the same data through the
Anthropic API (live or Batch) instead of Claude Code agents; it needs `ANTHROPIC_API_KEY` and the
`anthropic` package in `requirements.txt`. Run `.venv/bin/python -m finra2210.generate --help`.

## Retraining and feedback

How the reviewer gets better: reviewer feedback and Claude Opus audits become corrected labels and
preference pairs; supervised rounds retrain from the base model on the corrected data; DPO and GRPO
(with a reward computed from the labels) target specific confusions; a release gate blocks any
model that is worse than the current one on test, OOD, chat or any broad group. Code, sample data
and the full plan: **[retraining/README.md](retraining/README.md)**.

## Code map (`src/finra2210/`)

| Module | Role |
|---|---|
| `taxonomy.py` | rule digest, 23 categories → rule cites, broad groups, sub-groups, difficulty guide, scenario axes |
| `scenarios.py`, `chunks.py` | in-domain and OOD scenario sampler; writes chunks for agents |
| `topup.py`, `chat_topup.py` | targeted top-ups (sparse categories; planned chat top-up) |
| `check.py`, `progress.py`, `snapshot.py` | per-chunk checker for agents; progress; checksummed snapshots |
| `schema.py`, `postprocess.py` | label-consistency validator; validation + de-duplication + stats |
| `split.py`, `prompts.py` | stratified, incremental split; system prompt, user turn and reference answer (chat format) |
| `generate.py` | Path B: generation through the Anthropic API |
| `mlx_data.py`, `train_phi4.py`, `loss_report.py` | mlx data prep; PyTorch/TRL trainer; loss curve and training summary |
| `evaluate.py`, `analysis.py` | scoring (all metrics and breakdowns); confusion matrices and per-category analysis |
| `review.py` | review one message with a trained model |

## Limitations

- **Synthetic labels.** Teacher-model judgments, not regulator decisions; a compliance
  professional should spot-check them before anyone relies on the numbers.
- **Overlapping categories.** Several categories share a rule paragraph; exact-category scores
  understate how often the model finds the right problem (see the broad-group scores).
- **Unseen categories.** `options_communication` and `third_party_content_adoption` were never
  trained on and are never named, although the items are still flagged not compliant (16 of 16
  and 78 of 79 on OOD).
- **Chat is a minority** of the data (18.8%); a chat top-up is planned.
- **Scope.** Text and metadata only. No filing deadlines (2210(c)), recordkeeping or supervisory
  workflow, and nothing outside 2210, 2212 and 2220. US retail broker-dealer focus.
- **Rules change.** The digest reflects the rule text retrieved in October 2026; check current
  FINRA rules before production use.
- **License.** None has been chosen yet.
