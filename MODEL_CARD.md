# FINRA 2210 reviewer (Phi-4-mini-instruct + LoRA): one-pager

**What it does.** A first-pass compliance reviewer for broker-dealer communications under FINRA
Rule 2210. Given one communication plus its distribution metadata, it says what kind of
communication it is under 2210, whether it is approvable, which violations it contains (with
the verbatim excerpt and rule paragraph), how serious they are, and what the reviewer should do next.
It is an assistant to a registered principal, not a replacement: every verdict needs human review.

**Input.** The communication text (email, letter, SMS, IM, newsletter, social post, web page, ...)
plus sender role, firm, recipient type, and distinct retail recipients in the trailing 30 days.

**Output.** A short review in prose ending with one JSON verdict. The model makes four predictions:

| Output | Kind | Classes |
|---|---|---|
| `communication_type` | single-label, 3 classes | `correspondence`: retail recipients, ≤ 25 in 30 days · `retail_communication`: retail, > 25 in 30 days or public · `institutional_communication`: institutional investors only |
| `compliant` | binary | true ⇔ no violations and severity `none` |
| `severity` | single-label, 4 classes | `none` · `low`: technical or disclosure gap, little investor harm · `medium`: misleading content a reviewer would send back · `high`: promissory, guarantee, projection, fraud-adjacent, or likely Enforcement referral |
| `violations[]` | multi-label, 0 to 4 per item, 23 categories | each entry has `category`, `rule`, `excerpt`, `explanation` |

It also returns `required_actions` (1 to 3 reviewer steps) and a 2–4 sentence `rationale`.

## The 23 violation categories, in 5 broad groups

Applies to: C = correspondence, R = retail, I = institutional. n = in-domain train+test items with that category (11,896 total). The model outputs the label in the second column. The groups are defined in `BROAD_GROUPS` in [taxonomy.py](src/finra2210/taxonomy.py); `evaluate.py` and the live dashboard report precision / recall / F1 per group (`broad_groups`) and overall (`category_broad_group_micro`), where a hit is any category from the right group.

**A. Misleading or unbalanced content — 2210(d)(1) (10)** · group key `misleading_content`

| # | Category | Model label | What it flags | Rule | Applies | n |
|---|---|---|---|---|---|---|
| 1 | Performance projection | `performance_projection` | Predicts or projects returns | 2210(d)(1)(F) | C R I | 1,336 |
| 2 | Hypothetical / backtest misuse | `hypothetical_or_backtest_misuse` | Backtests or illustrations shown as achievable | 2210(d)(1)(F) | C R I | 207 |
| 3 | Promissory language or guarantee | `promissory_or_guarantee` | "Can't lose", "risk-free", "safe" | 2210(d)(1)(B) | C R I | 1,040 |
| 4 | Exaggerated or unwarranted claim | `exaggerated_or_unwarranted_claim` | Unsupported superlatives | 2210(d)(1)(B) | C R I | 871 |
| 5 | False or misleading statement | `false_or_misleading_statement` | Materially untrue fact | 2210(d)(1)(B) | C R I | 749 |
| 6 | Not fair and balanced | `not_fair_and_balanced` | Benefits without the matching risks | 2210(d)(1)(A) | C R I | 1,324 |
| 7 | Product-specific omission | `product_specific_omission` | e.g. surrender charges, REIT illiquidity, leveraged-ETF reset | 2210(d)(1)(A) | C R I | 102 |
| 8 | Misleading in context | `misleading_in_context` | True statements that mislead together | 2210(d)(1)(D) | C R I | 542 |
| 9 | Buried disclosure | `buried_disclosure` | Material risk hidden in a footnote | 2210(d)(1)(C) | C R I | 321 |
| 10 | Third-party content adoption | `third_party_content_adoption` | Adopted third-party content that itself violates | 2210(d)(1) / RN 17-18 | C R | OOD only |

**B. Missing required disclosures — 2210(d)(2)-(d)(8), 2212 (9)** · group key `required_disclosures`

| # | Category | Model label | What it flags | Rule | Applies | n |
|---|---|---|---|---|---|---|
| 11 | Unsupported comparison | `unsupported_comparison` | Comparison without the material differences | 2210(d)(2) | R | 243 |
| 12 | Member firm name missing | `member_name_missing` | Firm name not prominent | 2210(d)(3) | C R | 379 |
| 13 | Incomplete tax claim | `tax_claim_incomplete` | "Tax-free" without saying which taxes | 2210(d)(4) | C R | 334 |
| 14 | Fund performance disclosure | `fund_performance_disclosure` | Missing standardized returns, sales charge, expense ratio | 2210(d)(5) | C R | 595 |
| 15 | Testimonial disclosure | `testimonial_disclosure` | Missing "not representative" or paid-testimonial disclosure | 2210(d)(6) | C R | 348 |
| 16 | Recommendation disclosure | `recommendation_disclosure` | Security recommendation missing conflicts or price | 2210(d)(7) | R | 172 |
| 17 | Past recommendations reference | `past_recommendations_reference` | Cites past winners without the full one-year list | 2210(d)(7)(C) | C R | 187 |
| 18 | BrokerCheck link missing | `brokercheck_link_missing` | Retail web page lacks a BrokerCheck link | 2210(d)(8) | R | 107 |
| 19 | Fund ranking disclosure | `fund_ranking_disclosure` | Fund ranking without Rule 2212 disclosures | 2212 | R | 80 |

**C. Approval and audience — 2210(a), (b) (2)** · group key `approval_and_audience`

| # | Category | Model label | What it flags | Rule | Applies | n |
|---|---|---|---|---|---|---|
| 20 | Unapproved retail communication | `unapproved_retail_communication` | Retail communication used without principal pre-approval | 2210(b)(1) | R | 155 |
| 21 | Institutional forwarding | `institutional_forwarding` | "Institutional" piece that will reach retail investors | 2210(a)(4) / (b)(3) | I | 131 |

**D. Implied FINRA / SIPC endorsement — 2210(e) (1)** · group key `regulator_endorsement`

| # | Category | Model label | What it flags | Rule | Applies | n |
|---|---|---|---|---|---|---|
| 22 | Implied FINRA endorsement | `finra_endorsement_implied` | Implies FINRA or SIPC endorses or approved something | 2210(e) | C R I | 345 |

**E. Product-specific rules — 2220 (1)** · group key `product_rules`

| # | Category | Model label | What it flags | Rule | Applies | n |
|---|---|---|---|---|---|---|
| 23 | Options communication | `options_communication` | Options risk omitted or promoted before ODD delivery | 2220 / 2210(d)(1) | C R | OOD only |

## Training data

All items are synthetic broker-dealer communications with reference labels. People and firms
are fictional, and dates fall between 2023 and 2026.

| Set | Items | Use |
|---|---|---|
| In-domain train | 10,710 | fine-tuning. 5% is held out as validation (539), so 10,171 are trained on |
| In-domain test | 1,186 | held-out evaluation (10%, never trained on) |
| Out-of-distribution (OOD) | 5,800 | evaluation only: each item is shifted on 1–3 axes |

**In-domain composition (11,896 items).**
- **Communication type:** correspondence 5,870 (49%), retail communication 4,567 (38%),
  institutional 1,459 (12%). Type follows the distribution metadata: ≤ 25 retail recipients in 30
  days is correspondence, > 25 or the public is retail, and institutional-only recipients are
  institutional.
- **Verdict:** 5,283 compliant (44%; 42–46% within each type), 6,613 not compliant. Severity:
  none 5,283, low 345, medium 3,363, high 2,905.
- **Violations per item:** 0 → 5,283, 1 → 4,175, 2 → 1,922, 3 → 515, 4 → 1.
- **Difficulty:** easy 3,519 (30%), medium 4,786 (40%), hard 3,591 (30%).
  - Easy items have one blatant violation, or are routine compliant messages.
  - Medium items need the whole message read, for example an implied projection or a missing
    disclosure.
  - Hard items are compound or borderline violations, or compliant near-misses that look
    aggressive but pass.
- **Channels (16):**
  - Email 4,246, letter 1,112, SMS 1,074, instant message 896.
  - Mass email 843, social post 687, web page 626, newsletter 445, blog 340.
  - Seminar invitation 322, seminar deck/script 306, form letter 284, text blast 261.
  - Radio/podcast script 208, print ad 141, video script 102.
- **Products:** mutual funds 2,081, ETFs 1,840, stocks 1,733, bonds/CDs 1,075, 401(k)/IRA
  rollover 1,052, market commentary 860, target-date funds 817, munis 792, logistics with no
  product 706, 529 plans 530, fixed annuities 410.
- **Senders:**
  - Registered reps 55% and dual-registered advisors 18%.
  - Branch managers, unregistered client-service associates, wholesalers, compliance/ops staff
    and the firm's marketing department make up the rest.
- **Tones (8, roughly even):** formal, friendly-professional, casual/texting, salesy,
  apologetic complaint response, terse, educational, urgent market event.
- **Length:** median 915 characters; 10th–90th percentile 401–1,747; maximum 5,104.

**OOD set (5,800 items, 2,632 compliant).** Each item shifts on 1–3 axes:
- **Product (2,716):** options, crypto ETFs, Reg D private placements, structured notes, variable
  annuities, leveraged ETFs, non-traded REITs, SPACs, margin, CMOs.
- **Channel (2,711):** LinkedIn / WhatsApp / X DMs, chatbot transcripts, voicemail, long email
  threads, scanned handwriting with OCR noise, payment-app memos.
- **Style (1,516):** slang and emoji, non-native English, all caps, pasted third-party links.
- **Audience (1,229):** an institution that forwards to retail, a $40M family office (not
  institutional), a Spanish-speaking client, a journalist.
- **Regulatory twist (1,035):** 2210(f) public appearances, Rules 2212–2214 and 2241, SIPC,
  Reg BI, Rule 3240, AI-drafted text under RN 24-09.
- **Firm type (910):** bank-affiliated, insurance, crypto-adjacent, robo-hybrid, new member.

The axes are recorded per item, so OOD results break down by axis.

**Training format.** Each item becomes a chat conversation:
- A fixed reviewer system prompt.
- A user turn: one of 15 request phrasings, a sentence of context (sender, recipients, 30-day
  count), and the communication text.
- The reviewer's answer: the classification with its threshold reasoning, the verdict, each
  finding as rule + verbatim excerpt + explanation, the rationale, the next steps and the JSON
  verdict.
- In 30% of conversations, a follow-up exchange: how to fix it, which finding is most serious,
  or why it is (or is not) correspondence.

That gives 7,444 single-turn and 3,266 two-turn conversations in train. Loss is computed only
on the reviewer's turns, so each two-turn conversation becomes two training examples: 13,271
train / 705 validation.

## How the data was curated

1. **Taxonomy first.** A working digest of Rule 2210 was written before any data, along with
   23 violation categories. Each category is pinned to one rule citation and to the
   communication types it can apply to; for example, (d)(2)–(d)(7) disclosures apply to
   correspondence and retail but never to institutional. The digest also defines the
   difficulty tiers and the severity scale.
2. **Scenarios are sampled before anything is written.** Every item starts as a spec, so the
   data's diversity is set by the specs rather than left to the writer. A spec fixes:
   - difficulty (30/40/30) and the compliance target (~45% compliant);
   - the exact violation categories, weighted toward what review desks see most and limited to
     those valid for the type and channel;
   - channel, product, sender role, tone and length;
   - the recipient profile and 30-day recipient count, which determine the communication type.

   Compliant medium and hard specs carry a "near-miss" topic. For example: fund returns quoted
   with full standardized performance, a hypothetical clearly labeled, or "guaranteed" used only
   where it is literally true. This keeps the model from learning "mentions returns ⇒
   violation".
3. **Teacher writing.** Claude Code subagents wrote the items, acting as a senior FINRA
   Advertising Regulation examiner and former chief compliance officer.
   - Claude Opus 5.5 wrote 11,516 in-domain and 5,760 OOD items; Claude Fable 5.1 wrote 380
     and 40.
   - Each subagent read the rule digest, the category-to-rule table and the writing and
     labeling rules, then wrote one item per spec in chunks of 24–40.
   - Each item is the communication in the channel's native format, plus its label.
4. **Automatic validation, repeated until clean.** Every chunk was checked against its specs,
   and the writer fixed and re-checked until it reported zero problems. The checker requires:
   - every rule citation to be the one assigned to its category;
   - every excerpt to be a verbatim substring of the text (≤ 200 characters);
   - no category cited for a communication type it cannot apply to;
   - the communication type to agree with the recipient metadata;
   - `compliant`, `violations` and `severity` to agree with each other;
   - the item to match its spec (difficulty, channel, recipient count, type, categories) and
     to be 40–6,000 characters long.
5. **Frozen snapshot.** Validated output is copied into a timestamped, append-only snapshot
   with SHA-256 checksums before any cleaning.
6. **Re-validation and de-duplication.** Every item was validated again, then de-duplicated:
   - exact duplicates (after normalization) are removed;
   - near-duplicates are removed when more than 50% of their 6-word sequences match an earlier
     item;
   - OOD items were de-duplicated against in-domain, and the top-up against OOD.

   Result: 0 rejects and 0 duplicates. The largest 6-gram overlap between any two round-1 items
   is ≈ 0.38, and ≈ 0.24 between the top-up and everything else.
7. **Targeted top-up of sparse categories.** Three categories were rare after round 1, so 336
   extra items made each one the primary violation (plus 16 compliant near-misses per
   category). Each item varies how the category appears, such as a BrokerCheck name with no
   link, or an "institutional" piece offered for white-labeling to an RIA.

   | Category | Train before | Train after |
   |---|---|---|
   | brokercheck_link_missing | 3 | 96 |
   | institutional_forwarding | 35 | 120 |
   | unapproved_retail_communication | 58 | 144 |
8. **Splitting.**
   - Test is 10% of in-domain, stratified by difficulty × communication type × primary
     category.
   - The split is incremental: when items are added, earlier items keep their train or test
     side, and only the new items are stratified. A model trained on an earlier train set
     therefore stays uncontaminated on the current test set.
   - OOD is never trained on.
   - A stratified 5% of train is held out for validation during fine-tuning.
9. **Format check.** Every reference answer was parsed back into JSON and compared with its
   label, with 0 mismatches. The longest training example is 1,913 tokens, under the
   2,048-token limit used for training.

**Not done yet:** no human compliance expert has reviewed the labels. Chat is 19% of the data,
and a chat-focused top-up is planned in [CHAT_TOPUP_PLAN.md](CHAT_TOPUP_PLAN.md).

## Training and current status

- **Training.** LoRA (rank 32, alpha 64, dropout 0.05) on 4-bit Phi-4-mini-instruct with mlx-lm
  on an M5 Pro.
  - Adapters on the attention and MLP projections in all 32 blocks.
  - AdamW; learning rate 2e-4 with 50 warmup steps, then cosine decay to 2e-6.
  - Batch 4 with 4-step gradient accumulation (effective batch 16).
  - 2 epochs (6,636 iterations); loss on reviewer turns only.
  - The same recipe runs on NVIDIA GPUs with PyTorch/TRL on the bf16 base
    (`scripts/run_cuda_pipeline.sh`; guide: [TRAINING_NVIDIA.md](TRAINING_NVIDIA.md)).
  - Run status: [runs/phi4mini-2210-mlx/training_summary.md](runs/phi4mini-2210-mlx/training_summary.md).
  - Full test, OOD and zero-shot baseline results will be in `runs/*/eval_*.json`, broken
    down by chat vs non-chat channel.
- **Interim result.** Checkpoint 5,200 of 6,636, on the first 128 test rows (indicative only):

  | Metric | Fine-tuned | Zero-shot base (same first 64 rows) |
  |---|---|---|
  | Valid JSON output | 100% | 66% |
  | Compliant / not compliant correct | 97.7% | 30% |
  | Communication type correct | 100% | 0% |
  | Severity correct | 91% | 2% |
  | Violation detection F1 | 0.98 | 0.30 |
  | Category F1 | 0.77 | 0.00 |
  | Rule-citation F1 | 0.78 | 0.00 |

  - On the 63 correspondence rows, the compliant / not-compliant call was 100% correct and category F1 was 0.77.
  - The main weakness is naming the exact category among overlapping ones: `not_fair_and_balanced` vs `misleading_in_context` vs `false_or_misleading_statement`.

## Examples: chat messages (checkpoint 5,200, test set)

Five real SMS / instant-message items from the test set; the model never saw them in training. Its predicted labels are shown in the format it outputs them: type · compliant · severity · violations.

**1. Group chat with a hypothetical: correctly passed** (`in_domain-002282`, hard). A rep in a firm-approved group chat with 6 retail clients posts a compounding illustration: "HYPOTHETICAL ILLUSTRATION ONLY: $10,000 growing at an assumed constant 6%…". The post adds that the rate "is not a prediction", ignores fees and taxes, and that returns "can be negative, including losses of principal."
- Model: `correspondence` · compliant · `none` · no violations. ✓ Matches the reference label.
- What makes it hard: the post shows numbers, but the hypothetical is properly framed, so neither `performance_projection` nor `hypothetical_or_backtest_misuse` applies.

**2. Advisor chat with a prospect: two violations found** (`in_domain-005314`, medium). An IM thread about a 401(k) rollover:
"you get growth, you get diversification, you get low cost. kind of the whole package" … one client "told me last month 'Jake turned my sleepy rollover into the best-performing money I've ever had, up 34% in under two years.'"
- Model: `correspondence` · not compliant · `medium` · ✓ matches both violations:
  - `not_fair_and_balanced` (2210(d)(1)(A))
  - `testimonial_disclosure` (2210(d)(6))
- Actions: remove the testimonial or add the (d)(6) disclosures; send a follow-up that balances the pitch with risks and costs.

**3. Institutional chat with a forecast: caught** (`in_domain-010331`, medium). A rep messages a retirement-plan CIO: "the 2040 should compound right around 7-7.5% a year over the next decade, vs. maybe 5-6 for the big passive series."
- Model: `institutional_communication` · not compliant · `high` · `performance_projection` (2210(d)(1)(F)). ✓ Matches.
- The projection rule applies to institutional communications too; the chat also discusses risk, but that does not cure the forecast.

**4. Text message quoting fund returns: disclosure gap caught** (`in_domain-000814`, easy). A rep texts a retail client whose CD is maturing: "our Bayshore Core Bond Fund (Class A) returned 11.4% over the past 12 months … want me to set aside a piece of it for the fund?"
- Model: `correspondence` · not compliant · `medium` · `fund_performance_disclosure` (2210(d)(5)). ✓ Matches.
- Action: follow up with standardized quarter-end returns, the maximum sales charge and the expense ratio.

**5. A partial miss: one-line chat with two problems** (`in_domain-005140`, hard). A rep writes on the firm's secure chat: "remember I had you grab the Clark County water bonds at 91 in Oct, they're 98 now. more tax-free paper on Thursday's calendar, want me to hold you 50k?"
- Reference: `past_recommendations_reference` (2210(d)(7)(C)) and `tax_claim_incomplete` (2210(d)(4)), severity `medium`.
- Model: finds the past-recommendation violation and gets the verdict and severity right, but misses the incomplete "tax-free" claim. ✗ (partial)
- Two violations packed into one short message are where the model most often drops one.

On the 19 chat items in the 128-row sample, all compliant / not-compliant calls and all severities were correct, and category F1 was 0.85. That is a small sample.

## Limits

- **Synthetic labels.** These are teacher-model judgments, not FINRA or regulator decisions; have a compliance SME spot-check before relying on them.
- **Chat is a minority of the data.** Only 19% of in-domain items are chat: 1,074 SMS, 896 IM and 261 retail text blasts. The rest are email, letters and public retail channels, so chat performance should be measured on its own slice.
- **Two categories never seen in training.** `options_communication` and `third_party_content_adoption` appear only in the OOD set, so expect low recall on them.
- **Scope.** It reviews only the text and metadata it is given. It does not check filing deadlines (2210(c)), recordkeeping or supervisory workflow, or rules outside 2210, 2212 and 2220.
