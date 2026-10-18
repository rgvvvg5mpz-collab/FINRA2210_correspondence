# Chat top-up plan (proposed, not generated)

Status, 2026-10-17: **plan only**. No chunks, agent output, snapshot, clean set or split has been
written. `data/splits` and the running Phi-4-mini mlx run are untouched. Generation needs the user's
go-ahead; retraining needs a separate go-ahead.

## Why

The reviewer is meant for chat (SMS, instant message, text blasts), but chat is a minority of
the training data and almost absent from the retail-only rules:

- In-domain clean set: 2,231 of 11,896 (18.8%) are chat: sms_text 1,074, instant_message 896,
  text_message_blast 261. Test: 229 of 1,186.
- The base sampler only sends chat on correspondence / institutional paths, plus text blasts
  for retail. So there is **no** retail SMS or retail group IM. That is the exact case where a
  rep's "one-off texts" cross the 25-recipient line and become a retail communication.
- Retail-only categories in chat (all data / test): unsupported_comparison 14 / 1,
  recommendation_disclosure 12 / 1, fund_ranking_disclosure 4 / 0, unapproved_retail_communication
  9 / 0. Chat test metrics for these are noise today.

`evaluate.py` now reports `by_channel_group` (chat / chat_app / non_chat), `chat_by_communication_type`
and `by_channel` on every run, so the current run's eval will give the chat baseline.

## What: 2,400 chat scenarios, 100 chunks of 24

Generator: [src/finra2210/chat_topup.py](src/finra2210/chat_topup.py) (seed 2212). To see the tables
without writing anything:

```bash
PYTHONPATH=src python3 -m finra2210.chat_topup --stats
```

60% violations (1,440), each with a forced primary category. 40% compliant chat near-misses (960).

### By channel × communication type

| Cell | Now (all data) | + Top-up | After | Violations / compliant in top-up |
|---|---|---|---|---|
| sms_text / correspondence | 869 | 600 | 1,469 | 360 / 240 |
| instant_message / correspondence | 721 | 500 | 1,221 | 300 / 200 |
| text_message_blast / retail | 261 | 550 | 811 | 330 / 220 |
| sms_text / retail (same text to >25, one at a time or a large group text) | 0 | 150 | 150 | 90 / 60 |
| instant_message / retail (client group chat or broadcast channel, >25 members) | 0 | 150 | 150 | 90 / 60 |
| sms_text / institutional | 205 | 250 | 455 | 150 / 100 |
| instant_message / institutional | 175 | 200 | 375 | 120 / 80 |
| **Total chat** | **2,231 (18.8%)** | **2,400** | **4,631 (32.4%)** | 1,440 / 960 |

Communication-type totals in the top-up: correspondence 1,100, retail 850, institutional 450.

### By category (chat samples carrying the category; test ≈ 10% of new ids)

| Category | Primary in top-up | Any in top-up | Chat now | Chat after | Chat test now → after |
|---|---|---|---|---|---|
| performance_projection | 100 | 217 | 288 | 505 | 28 → ~50 |
| not_fair_and_balanced | 100 | 228 | 253 | 481 | 29 → ~52 |
| promissory_or_guarantee | 100 | 195 | 208 | 403 | 16 → ~36 |
| exaggerated_or_unwarranted_claim | 80 | 164 | 183 | 347 | 13 → ~29 |
| false_or_misleading_statement | 70 | 144 | 156 | 300 | 13 → ~27 |
| misleading_in_context | 70 | 117 | 122 | 239 | 9 → ~21 |
| fund_performance_disclosure | 60 | 110 | 118 | 228 | 11 → ~22 |
| member_name_missing | 60 | 87 | 68 | 155 | 6 → ~15 |
| tax_claim_incomplete | 50 | 73 | 74 | 147 | 15 → ~22 |
| testimonial_disclosure | 50 | 82 | 62 | 144 | 11 → ~19 |
| buried_disclosure | 50 | 82 | 62 | 144 | 6 → ~14 |
| finra_endorsement_implied | 40 | 59 | 64 | 123 | 10 → ~16 |
| unsupported_comparison (retail only) | 80 | 106 | 14 | 120 | 1 → ~12 |
| hypothetical_or_backtest_misuse | 50 | 75 | 43 | 118 | 5 → ~13 |
| unapproved_retail_communication (retail only) | 100 | 100 | 9 | 109 | 0 → ~10 |
| recommendation_disclosure (retail only) | 80 | 95 | 12 | 107 | 1 → ~11 |
| institutional_forwarding (institutional only) | 60 | 60 | 38 | 98 | 4 → ~10 |
| past_recommendations_reference | 50 | 60 | 31 | 91 | 4 → ~10 |
| product_specific_omission | 50 | 63 | 16 | 79 | 4 → ~10 |
| fund_ranking_disclosure (retail only) | 50 | 55 | 4 | 59 | 0 → ~6 |
| third_party_content_adoption | 50 | 50 | 0 | 50 | 0 → ~5 |
| options_communication | 40 | 40 | 0 | 40 | 0 → ~4 |
| brokercheck_link_missing | 0 | 0 | 0 | 0 | n/a: website_page only, cannot occur in chat |

Violations per top-up spec: 1 → 738, 2 → 562, 3 → 140. Difficulty: easy 649, medium 830, hard 921.
Length: one-liner 776, short (2–4 messages) 1,053, thread (5–8 messages) 571.

### Chat-specific angles (each spec carries `chat_angle` and `chat_format`)

- **One-liners with 2+ violations**: 289 specs (20% of violations). A single message under
  160 characters with two or three distinct problems (e.g. a guarantee plus a projection plus
  no firm name).
- **Threads and group chats**: 3–8 message threads with timestamps, where client lines give
  context and the rep's lines are under review. Family or small-group chats count as
  correspondence (≤25). Client group chats and broadcast channels with >25 members count as
  retail.
- **Emoji / abbreviations / voice-to-text typos**: meaning carried by 🚀📈, "tbh", "lol" and
  missing punctuation.
- **Off-channel requests**: client asks to move to a personal cell, WhatsApp or Signal, or to
  delete the thread. 2210 has no category for this. The label covers it in `required_actions`
  and the rationale (escalate under SEA 17a-4 / FINRA 4511 / 3110). Compliant near-misses show
  the rep keeping the conversation on the archived platform.
- **Threshold traps**: the same text sent one at a time to 30+ clients and treated as
  correspondence (the unapproved_retail_communication angle).
- **Pasted links and screenshots with one line of endorsement**: third-party content adoption.
- **Institutional chat**: bank, pension and insurer desk IMs; wholesaler texts to RIAs that
  invite forwarding to clients.
- **Compliant near-misses** (60% chat-specific, 40% from the existing pool): "is this
  guaranteed?" answered correctly; emoji with no claims; a text blast with approval code and
  STOP opt-out; logistics-only blasts; a rep who declines to text a return number. Topics heavy
  on disclosures are never squeezed into a one-liner.

## Cost and time

Measured from the 14 agents of the sparse-category top-up (same instructions, Claude Opus 5.5,
24 samples per chunk). Mean per chunk: 1.38M cache-read, 168k cache-write and ~16–31k output
tokens, 18 turns, 5–8 minutes. Opus 5.5 prices: $4 per million input tokens, $20 per million
output, $0.20 per million cache reads, cache writes $5 (5-minute cache) to $8 (1-hour cache).

| | Per chunk | 100 chunks (2,400 samples) |
|---|---|---|
| Cache read | 1.38M → $0.28 | 138M → ~$28 |
| Cache write | 168k → $0.84–1.34 | 17M → ~$84–134 |
| Output | ~20k → $0.40 | ~2M → ~$40 |
| **Total (API-equivalent)** | **~$1.5–2.0** | **~$150–200** |

Chat samples are shorter than the website / seminar pieces in the last top-up, so output is
likely at the low end. Run through Claude Code subagents (Path A), this draws on the plan's usage
limits instead of an API bill: about 100 agents and ~1,800 turns. Wall time is about 40–60 minutes
at 15–20 agents in parallel, plus about 15 minutes for checks, snapshot, postprocess and split.

A smaller option, 1,200 scenarios (50 chunks, ~$75–100, chat reaches ~26%), would halve every
"+ top-up" column. That leaves the retail-only categories at ~40–50 chat examples and ~5 chat
test cases each, which is too few to read reliably. **Recommendation: the full 2,400.**

### Downstream cost: retraining

- Train grows by ~2,160 conversations, to ~12,870. mlx rows grow ~13,271 → ~16,100. Two epochs
  at batch 4 is ~8,050 iterations (`iters` in configs/phi4mini_lora_mlx.yaml must be updated from
  6,636). That is ~15–16 h at 7 s/iteration; probably less, since chat rows are short.
- Test grows ~1,186 → ~1,426 (chat 229 → ~469). The eval adds ~6 minutes per model at 1.4 s/row.

## Expected effect (to be measured, not promised)

- **Measurement first**: the chat test slice doubles (229 → ~469). At ~80% accuracy, the 95%
  interval on chat accuracy narrows from about ±5 to about ±3.5 points. Every chat-applicable category gets at
  least ~5–10 chat test cases, against 0–1 today for the retail-only rules.
- **Most likely gains**: chat communication_type accuracy (the 25-recipient threshold in
  texts and group chats has no retail SMS/IM examples today), and retail-only categories
  on text blasts (recommendation / comparison / ranking disclosures, unapproved use).
- **Likely gains**: multi-label recall on one-liners. Short texts that pack two or three
  problems are rare in the current data (the existing chat median is ~610 characters).
- **Watch for**: precision on compliant chat. Near-misses are 40% of the top-up for this
  reason. Also, non_chat metrics should not regress; compare `by_channel_group.non_chat`
  before and after.

## Trade-offs to decide

1. **options_communication and third_party_content_adoption are OOD-only today** (in-domain
   weight 0). Their OOD rows (16 + 79) would become partly in-distribution. The off-channel
   angle also overlaps the OOD "recordkeeping failure" regulatory twist (119 rows). Both are
   realistic in chat, so the plan includes them. To keep the OOD set pure instead, set the two
   targets to 0 and move their 90 slots into the core categories in `TARGETS`.
2. **Near-duplicate filter on short texts**: postprocess drops samples with >50% shared 6-word
   shingles. One-liners with shared opt-out boilerplate ("Reply STOP to opt out") could trip it.
   The agent instructions should require varied footers. Expect a few percent rejects.
3. **Cheap complement**: oversample existing chat rows ×2 in `mlx_data.py`. Zero generation
   cost, but it adds no new retail-chat or one-liner coverage. Useful on top of the top-up,
   not instead of it.

## Steps once approved

Recommended order: let the current run (iteration 5,200 / 6,636) finish and evaluate. Its
`by_channel_group` / `chat_by_communication_type` numbers are the chat baseline on today's
test. Then:

1. `PYTHONPATH=src .venv/bin/python -m finra2210.chat_topup --out data/chunks_chat/in_domain` (100 chunks).
2. Write `data/chunks_chat/AGENT_INSTRUCTIONS.md` + `BRIEF.md` (copied from `data/chunks_topup/`,
   with a chat bullet: honor `chat_angle` / `chat_format`, render threads with names and
   timestamps, one-liners under 160 characters with every listed violation evidenced, off-channel
   requests labeled through required_actions, varied footers). Agents write to
   `data/agent_out_chat/in_domain/` and loop `finra2210.check` to 0 problems, then `--emit`.
3. Add `data/agent_out_chat` to `SOURCES` in `snapshot.py` and take a new snapshot (in_domain =
   previous raw + chat raws, byte-for-byte; ood unchanged).
4. `postprocess` with `--exclude data/clean/ood.clean.jsonl`. Check that the first 11,896 records
   are identical, so the new ids are `in_domain-011896`… and OOD clean is unchanged (md5).
5. `cp -R data/splits data/splits_pre_chat_<date>`, then
   `.venv/bin/python -m finra2210.split --clean data/clean/in_domain.clean.jsonl --out data/splits --keep-test data/splits_pre_chat_<date>/test.jsonl --keep-train data/splits_pre_chat_<date>/train.jsonl`.
   The keep lists point at the copy, not the files being rewritten. All current test ids stay
   in test and all train ids stay in train. Only the 2,400 chat ids are stratified. The current model stays clean on
   the new test.jsonl.
6. Retrain (new `iters`), evaluate with the same `evaluate.py`, and compare the chat slices with
   step 0's baseline. Needs a separate go-ahead.
