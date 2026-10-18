# Data: FINRA 2210 communications review dataset (v0.1.0)

Synthetic, labeled broker-dealer communications (emails, letters, texts, IMs, social posts,
website pages, seminar scripts, newsletters, ...) reviewed under FINRA Rule 2210. Each item says
whether the message is correspondence, a retail communication or an institutional communication,
whether it complies, and, if not, which of 23 violation categories it breaks, citing the rule
paragraph and quoting the offending text.

It was built to fine-tune `microsoft/Phi-4-mini-instruct` as a first-pass compliance reviewer for
a retail broker-dealer (see the [model card](../MODEL_CARD.md) and [results](../results/README.md)).

> **Labels are synthetic.** They were written by Claude models acting as a FINRA examiner, checked
> by code for internal consistency, and **not** reviewed by a compliance professional or a regulator.
> Do not treat them as regulatory determinations. All people, firms, amounts and events are fictional.

## What is where

**In this repository** (everything needed to train, evaluate and see how the data was made):

```
data/
  README.md                         this file
  splits/                           ready to train / evaluate (chat format)
    train.jsonl                     10,710  in-domain, for training
    test.jsonl                       1,186  in-domain, held out
    ood.jsonl                        5,800  out-of-distribution, evaluation only
    in_domain.manifest.json         split seed, strata, per-split counts, kept / new ids
    ood.manifest.json
  clean/                            the same items without chat formatting, plus each item's scenario spec
    in_domain.clean.jsonl           11,896  (= train + test)
    ood.clean.jsonl                  5,800  (= ood)
    *.stats.json                    label / channel / category distributions
    *.rejected.jsonl                empty: no item failed validation or de-duplication
  splits_r1_ids/                    train / test ids of round 1, for the incremental split
  chunks/BRIEF.md                   generation brief used for the pilot
  chunks_r1/BRIEF.md                generation brief every round-1 agent followed (rule digest, categories, labeling rules)
  chunks_r1/AGENT_INSTRUCTIONS.md   per-chunk agent procedure (round 1)
  chunks_topup/BRIEF.md             brief for the sparse-category top-up
  chunks_topup/AGENT_INSTRUCTIONS.md
  raw_finra/                        the FINRA / SEC source documents the brief was distilled from
    SOURCES.md                      file, title, URL, retrieval date and how each maps to the dataset
```

**On the [v0.1.0 release](https://github.com/rgvvvg5mpz-collab/FINRA2210_correspondence/releases/tag/v0.1.0)**
(`finra2210-dataset-v0.1.0.tar.gz`, with a `.sha256` file): the splits and clean files above plus
the frozen raw teacher output they were built from (`snapshots/20261017T045434Z/`: 495 in-domain
and 241 OOD chunk records, with SHA-256 checksums in `MANIFEST.json`).

**Local only** (working folders of the generation run, not published): scenario chunks
(`chunks*/in_domain`, `chunks*/ood`), raw agent output (`agent_out*`), agent scratch files
(`agent_work/`), older snapshots, mlx training files (`mlx/`) and a superseded full-reshuffle split
(`splits_reshuffled_20261017/`).

## How the data is used

| Use | File | Where in the repo |
|---|---|---|
| Fine-tuning (chat format, loss on assistant turns) | `splits/train.jsonl` | `training/` (`finra2210.mlx_data` holds out 5% of it as validation) |
| In-domain evaluation | `splits/test.jsonl` | `evaluation/`, `results/` |
| Out-of-distribution evaluation | `splits/ood.jsonl` | `evaluation/`, `results/` |
| A classifier or your own prompt format | `clean/*.clean.jsonl` (hold out the ids in `splits/test.jsonl`) | |
| Retraining rounds (label fixes, new samples, same test set) | `clean/in_domain.clean.jsonl` + `splits/` keep lists | `retraining/03_sft/` |
| Auditing labels | `splits/train.jsonl` + model predictions | `retraining/02_audit/` |

Never train on `test.jsonl` or `ood.jsonl`: they are the fixed yardsticks every version is compared on.

## Quick start

```python
import json

train = [json.loads(l) for l in open("data/splits/train.jsonl")]
ex = train[0]
ex["messages"]          # [{"role": "system"|"user"|"assistant", "content": ...}, ...]
ex["label"]["compliant"], [v["category"] for v in ex["label"]["violations"]]
```

With Hugging Face `datasets`:

```python
from datasets import load_dataset

ds = load_dataset("json", data_files={"train": "data/splits/train.jsonl",
                                      "test": "data/splits/test.jsonl",
                                      "ood": "data/splits/ood.jsonl"})
```

Train on `messages` with the loss on the assistant turns only. The gold verdict is the last
fenced json block in each assistant reply; parse it from your model's output the same way and
compare it with `label` (`finra2210.evaluate` does this). Every gold assistant turn parses back to
its `label`. With the Phi-4 tokenizer the longest training conversation is 1,913 tokens (test
2,027, OOD 2,143); training uses a 2,048-token limit.

MD5 of the splits (to check a copy): `train.jsonl` `f859399efbc7abc2c0e04aa2311d2984`,
`test.jsonl` `36569ce553eed440dd09074cb0bac327`, `ood.jsonl` `b7bcc715295d2f991afb9801912bf207`.

## Record format

### `splits/*.jsonl`

| Field | Description |
|---|---|
| `id` | `in_domain-NNNNNN` or `ood-NNNNNN`; stable across releases |
| `difficulty` | `easy` / `medium` / `hard` |
| `channel` | e.g. `email`, `sms_text`, `website_page`; OOD adds channels like `WhatsApp message` |
| `metadata` | `sender_role`, `firm`, `recipient_description`, `recipient_count_30d`, `recipient_type` (`retail` / `institutional`), `subject` |
| `communication` | the message text exactly as sent, in the channel's native format |
| `label` | the gold review (below) |
| `ood_shifts` | `null` in-domain. For OOD, the axes shifted for that sample (`channel`, `product`, `audience`, `style`, `firm_type`, `regulatory_twist`) |
| `teacher` | model that wrote the sample: `claude-opus-5-5` or `claude-fable-5-1` |
| `messages` | chat conversation for SFT: system, user, assistant, optionally a follow-up user and assistant turn |

`label`:

```json
{
  "communication_type": "correspondence | retail_communication | institutional_communication",
  "compliant": false,
  "violations": [
    {"category": "performance_projection", "rule": "2210(d)(1)(F)",
     "excerpt": "verbatim substring of communication (<= 200 chars)",
     "explanation": "1-2 sentences"}
  ],
  "severity": "none | low | medium | high",
  "required_actions": ["Return to rep for revision", "..."],
  "rationale": "2-3 sentences"
}
```

### `clean/*.clean.jsonl`

These have the same sample fields without `messages`. They also carry:
- `scenario_id`;
- `generator_notes`: the teacher's one-line note on what the sample tests;
- `scenario`: the full spec the sample was generated from, including `near_miss_topic` for compliant samples and `violation_categories` for violations.

### `snapshots/*/*.raw.jsonl`

There is one record per generation chunk: `{"scenarios": [...], "samples": [...], "error": "", "usage": {"source", "teacher"}}`.

### Chat format

- **System prompt:** one fixed reviewer system prompt.
- **User turn:**
  - one of 15 request phrasings (e.g. "Pre-use review request. Verdict, cites, and next steps please.");
  - a sentence of distribution context built from `metadata` (sender, firm, audience, recipient count, sometimes the subject line);
  - the communication.
- **Assistant turn:**
  - classification with the 25-recipients-in-30-days reasoning;
  - a plain-language verdict, the violations with cites and quoted excerpts, and next steps;
  - a closing ```json block with the full `label`.
- **Follow-up:** about 30% of conversations (3,266 train / 347 test / 1,704 OOD) add a second user turn and its answer. The question is one of 15 types, such as who must approve, whether it must be filed, or how to fix it, and the answer is derived from the label.

## Distribution

| | train | test | ood |
|---|---|---|---|
| rows | 10,710 | 1,186 | 5,800 |
| easy / medium / hard | 3,168 / 4,307 / 3,235 | 351 / 479 / 356 | 1,721 / 2,341 / 1,738 |
| compliant / violation | 4,754 / 5,956 | 529 / 657 | 2,632 / 3,168 |
| correspondence / retail / institutional | 5,282 / 4,112 / 1,316 | 588 / 455 / 143 | 2,710 / 2,121 / 969 |
| chat channels (sms_text, instant_message, text_message_blast) | 2,002 | 229 | 605 |

- **Violations per sample** (in-domain): 0 to 4. Of the violation samples, 4,175 have one violation, 1,922 two, 515 three and 1 four. Mean length is about 1,000 characters.
- **Severity** (in-domain): none 5,283, low 345, medium 3,363, high 2,905.
- **Teachers:**
  - in-domain: 11,516 Claude Opus 5.5 and 380 Claude Fable 5.1;
  - OOD: 5,760 Opus 5.5 and 40 Fable 5.1.
- **Per-category, per-channel and per-OOD-axis counts:** in `clean/*.stats.json`.

### Violation categories

Each category is pinned to one rule cite. For reporting they fall into 5 broad groups
(`BROAD_GROUPS` in `src/finra2210/taxonomy.py`); `label.violations[].category` holds the category label.

**Misleading or unbalanced content — 2210(d)(1)** (group key `misleading_content`; in-domain 6,492 / OOD 3,314)

| Category label | Rule | In-domain | OOD |
|---|---|---|---|
| `performance_projection` | 2210(d)(1)(F) | 1,336 | 610 |
| `not_fair_and_balanced` | 2210(d)(1)(A) | 1,324 | 599 |
| `promissory_or_guarantee` | 2210(d)(1)(B) | 1,040 | 429 |
| `exaggerated_or_unwarranted_claim` | 2210(d)(1)(B) | 871 | 423 |
| `false_or_misleading_statement` | 2210(d)(1)(B) | 749 | 378 |
| `misleading_in_context` | 2210(d)(1)(D) | 542 | 239 |
| `buried_disclosure` | 2210(d)(1)(C) | 321 | 137 |
| `product_specific_omission` | 2210(d)(1)(A) | 102 | 307 |
| `hypothetical_or_backtest_misuse` | 2210(d)(1)(F) | 207 | 113 |
| `third_party_content_adoption` | 2210(d)(1) / RN 17-18 | — | 79 |

**Missing required disclosures — 2210(d)(2)-(d)(8), 2212** (group key `required_disclosures`; in-domain 2,445 / OOD 947)

| Category label | Rule | In-domain | OOD |
|---|---|---|---|
| `fund_performance_disclosure` | 2210(d)(5) | 595 | 230 |
| `member_name_missing` | 2210(d)(3) | 379 | 146 |
| `testimonial_disclosure` | 2210(d)(6) | 348 | 137 |
| `tax_claim_incomplete` | 2210(d)(4) | 334 | 144 |
| `unsupported_comparison` | 2210(d)(2) | 243 | 107 |
| `past_recommendations_reference` | 2210(d)(7)(C) | 187 | 78 |
| `recommendation_disclosure` | 2210(d)(7) | 172 | 72 |
| `brokercheck_link_missing` | 2210(d)(8) | 107 | 3 |
| `fund_ranking_disclosure` | 2212 | 80 | 30 |

**Approval and audience — 2210(a), (b)** (group key `approval_and_audience`; in-domain 286 / OOD 241)

| Category label | Rule | In-domain | OOD |
|---|---|---|---|
| `institutional_forwarding` | 2210(a)(4) / 2210(b)(3) | 131 | 181 |
| `unapproved_retail_communication` | 2210(b)(1) | 155 | 60 |

**Implied FINRA / SIPC endorsement — 2210(e)** (group key `regulator_endorsement`; in-domain 345 / OOD 147)

| Category label | Rule | In-domain | OOD |
|---|---|---|---|
| `finra_endorsement_implied` | 2210(e) | 345 | 147 |

**Product-specific rules — 2220** (group key `product_rules`; in-domain — / OOD 16)

| Category label | Rule | In-domain | OOD |
|---|---|---|---|
| `options_communication` | 2220 / 2210(d)(1) | — | 16 |

`third_party_content_adoption` and `options_communication` appear only in OOD, so they test whether a model generalizes to labels it never saw.

### In-domain vs OOD

- **In-domain:**
  - retail broker-dealer reps, branch managers and marketing writing about mutual funds, ETFs, stocks, bonds, IRAs, 401(k) rollovers, 529s, fixed annuities and munis;
  - channels: email, letters, SMS, IM, mass email, social posts, website pages, newsletters, blogs, seminar invitations and scripts, form letters, text blasts, radio/podcast and video scripts, print ads.
- **OOD:** every sample is shifted on 1 to 3 axes, recorded in `ood_shifts`:
  - **channel (2,711):** WhatsApp, LinkedIn and X DMs, Slack/Teams, firm chatbot transcripts, voicemail transcripts, OCR'd handwritten notes, long email threads, payment-app memos, forum replies, video-call summaries;
  - **product (2,716):** options, crypto ETFs, Reg D, structured notes, variable annuities, leveraged ETFs, non-traded REITs, SPACs, margin, CMOs;
  - **audience (1,229):** institutional recipients who forward, plan sponsors vs participants, family offices, Spanish-language;
  - **style (1,516);**
  - **firm type (910);**
  - **regulatory twist (1,035):** 2210(f), 2212 to 2214, 2241, SIPC, Reg BI, 3240, RN 24-09 on AI-generated content.
- **Near-duplicates:** OOD was de-duplicated against in-domain, so no OOD text near-duplicates a training text.

## How it was curated

1. **Sources.** The current text of Rule 2210 and the related rules was archived from finra.org and eCFR on 2026-10-14 (`raw_finra/SOURCES.md` lists files, URLs and dates). The archive covers:
   - Rules 2010, 2020, 2111, 2211 to 2216, 2220, 2241, 2330, 3110, 3240, 4511 and 4512;
   - Regulatory Notices 12-29, 15-50, 16-41, 17-18, 19-31, 19-32, 20-21 and 24-09;
   - the Rule 2210 FAQ;
   - the communications section of the 2025 FINRA Annual Regulatory Oversight Report;
   - SEC Rules 482 and 34b-1.

   These were distilled into a working rule digest, a category-to-cite table and difficulty definitions (`chunks_r1/BRIEF.md`).
2. **Scenarios first.** A seeded sampler fixes each sample's spec before any text is written, so diversity and label balance are controlled rather than left to the generator. The spec covers:
   - difficulty, compliant or not, and which categories;
   - channel, product, sender, firm type;
   - recipient profile, recipient count in 30 days and recipient type, which together determine the communication type;
   - tone and length;
   - for OOD, the shift axes.

   Targets were 30/40/30 easy/medium/hard and about 45% compliant. Compliant samples each get a `near_miss_topic`, so they touch exactly the content a naive model would flag (e.g. "safe" used where it is literally true, a price target with its basis and risks).
3. **Generation.** Scenarios were cut into chunks of 24 (40 in the pilot). Up to 20 Claude Code subagents ran in parallel, each writing one chunk while following the brief, in the role of a senior FINRA Advertising Regulation examiner:
   - Claude Fable 5.1 wrote the pilot and the first chunks;
   - Claude Opus 5.5 wrote the rest.
4. **Per-chunk checker.** Each agent ran `finra2210.check` on its own output and fixed problems until none were reported. The checker requires:
   - metadata, difficulty, compliance and communication type to match the scenario;
   - every required category to be present, with the brief's rule cite;
   - each excerpt to be a verbatim substring of the communication;
   - only categories that can apply to the communication type (e.g. (d)(2) and (d)(7)(A) only for retail; (d)(3) to (d)(6) never for institutional);
   - compliant samples to have no violations and severity `none`.
5. **Snapshot.** Checked output was frozen with a sha256 per file (`snapshots/`).
6. **Validation and dedup** (`finra2210.postprocess`):
   - the schema and label-consistency checks run again, including that the communication type agrees with the recipient metadata (25 or fewer retail recipients means correspondence, more than 25 means retail, institutional-only means institutional);
   - exact-match plus 6-word-shingle near-duplicate removal (overlap > 0.5); in a sampled check, the largest overlap between two kept samples was about 0.38;
   - result: 0 rejects and 0 duplicates.
7. **Fixes found while auditing.** These were applied before release:
   - compliant OOD scenarios that had been given an "AI draft sent without review" twist were reworded to "reviewed and approved by a principal";
   - price-target topics paired with fund products were moved to individual stocks;
   - 29 OOD samples whose institutional audience was sent on a retail channel have that noted in `scenario.ood_shifts.audience`.
8. **Sparse-category top-up.** 336 extra in-domain samples were added for `brokercheck_link_missing`, `institutional_forwarding` and `unapproved_retail_communication` (ids `in_domain-011560` to `011895`). That is about 100 violation samples per category, plus 16 compliant near-misses per category, and each one targets a specific angle, such as BrokerCheck named but not hyperlinked. The brief is in `chunks_topup/`.
9. **Split.** The test set is a 10% split stratified by difficulty, communication type and primary category (147 strata, seed 42). It is incremental: the 1,158 round-1 test ids stay in test, and only the top-up was newly stratified (28 test / 308 train). OOD is never trained on.
10. **Chat export.** Samples were rendered into the conversations described above. Every assistant turn was parsed back and matched against its label.

## Limitations

- **Synthetic labels.** They have not been reviewed by a human; spot-check a random slice with a compliance professional before trusting test numbers.
- **The rule digest is a summary** made in October 2026. FINRA amends 2210 and its guidance; check current rule text before production use.
- **Thin categories.** `fund_ranking_disclosure` is the thinnest category in-domain (80 samples); `brokercheck_link_missing` and `options_communication` are thin in OOD.
- **Chat share.** Chat channels are 18.8% of in-domain data. A chat-focused top-up is planned but not part of this release.
- **One generator family.** The text comes from Claude models, so real reps' writing (typos, abbreviations, forwarded threads, attachments) is under-represented outside the OOD channel and style shifts.
- **US only.** English plus some Spanish (OOD), retail broker-dealer focus. Investment-adviser advertising (SEC Marketing Rule) and insurance-only rules are out of scope.
- **License.** None has been chosen for this dataset yet.

## Reproducing

The code in `src/finra2210/` holds the scenario sampler, the checker, snapshotting, postprocessing
(validation and de-duplication), the split and the chat export; the top-level
[README](../README.md#data-pipeline) gives the commands. The briefs and agent instructions above are
exactly what the generating agents read.
