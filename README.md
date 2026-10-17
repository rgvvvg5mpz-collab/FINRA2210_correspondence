# FINRA 2210 correspondence reviewer — dataset + Phi-4-mini fine-tune

Teachers: **Claude Fable 5.1** (`claude-fable-5-1`, pilot + first chunks) and **Claude Opus 5.5**
(`claude-opus-5-5`, the rest) mine labeled samples; every sample carries a `teacher` tag.
Student: **microsoft/Phi-4-mini-instruct** (3.8B) is LoRA-fine-tuned to act as a
first-pass reviewer of broker-dealer communications under FINRA Rule 2210.

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

5. Fine-tune (CUDA box recommended; use a Python 3.11/3.12 venv for torch):

```bash
python -m finra2210.train_phi4 --train data/splits/train.jsonl --eval data/splits/test.jsonl --out runs/phi4mini-2210 --merge
```

6. Evaluate on test and OOD (and the zero-shot base as a baseline):

```bash
python -m finra2210.evaluate --model runs/phi4mini-2210 --data data/splits/test.jsonl --out runs/phi4mini-2210/eval_test.json
```

```bash
python -m finra2210.evaluate --model runs/phi4mini-2210 --data data/splits/ood.jsonl --out runs/phi4mini-2210/eval_ood.json
```

```bash
python -m finra2210.evaluate --model microsoft/Phi-4-mini-instruct --data data/splits/test.jsonl --limit 200
```

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
  prompts.py     teacher prompt; student prompt/target format
  schema.py      JSON schema + label-consistency validator
  generate.py    Fable generation (live / batch, resumable)
  postprocess.py validate + dedup + stats
  split.py       stratified train/test; chat-format export
  train_phi4.py  LoRA SFT for Phi-4-mini-instruct
  evaluate.py    metrics by difficulty and OOD axis
data/            generated artifacts (gitignored-worthy)
```

## Caveats

- Labels are synthetic teacher labels, not regulator decisions. Spot-check `*.rejected.jsonl` and a random slice of accepted samples with a compliance SME before trusting the test numbers.
- The rule digest is a working summary; verify against the current rule text at finra.org before production use. FINRA amends 2210 periodically (e.g. the projected-performance exception for institutional / qualified-purchaser communications).
