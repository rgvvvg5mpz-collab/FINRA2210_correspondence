# FINRA 2210 reviewer (Phi-4-mini-instruct + LoRA): model card

**What it does.** A first-pass compliance reviewer for broker-dealer communications under FINRA
Rule 2210. Given one communication plus its distribution metadata, it says what kind of
communication it is under 2210, whether it is approvable, which violations it contains (with
the verbatim excerpt and rule paragraph), how serious they are, and what the reviewer should do next.
It is an assistant to a registered principal, not a replacement: every verdict needs human review.

**Get it.** The LoRA adapter (mlx format) is attached to the
[v0.1.0 release](https://github.com/rgvvvg5mpz-collab/FINRA2210_correspondence/releases/tag/v0.1.0);
the training and eval data are in [data/](data/); how to run it is in the [README](README.md#use-the-model).

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

## Violation categories

### How the categories are organized

Every finding the model reports is one of **23 categories**. Each category is pinned to exactly
one rule citation, so the `rule` the model writes follows from the `category` it picks. For
reporting, the 23 categories sit in a three-level hierarchy:

1. **Broad group (5).** Which part of the rules is broken: misleading content, a missing
   required disclosure, approval and audience, implied regulator endorsement, or a
   product-specific rule. Defined as `BROAD_GROUPS` in [taxonomy.py](src/finra2210/taxonomy.py)
   and scored by `evaluate.py` and the live dashboard.
2. **Sub-group.** Descriptive only, and only inside the two large groups. For example,
   "One-sided" vs "Overclaiming" inside misleading content. Defined as `SUBGROUPS` in the same
   file; not scored.
3. **Category (23).** The label the model outputs in `violations[].category`. Defined with its
   rule and the communication types it can apply to in `VIOLATION_CATEGORIES`.

**Labeling conventions** (enforced by the checker on every training item):
- One entry per category per message, with a verbatim excerpt of at most 200 characters.
  A message can have up to 4 categories.
- A category is only used for a communication type it can apply to. The applies-to column below
  uses C = correspondence, R = retail communication, I = institutional communication.
- A compliant message has no violations and severity `none`.

**How to read the numbers.**
- **Labels:** how many items carry the category: in-domain (11,896 items, train + test) /
  out-of-distribution (OOD, 5,800 items). An item with two categories counts once for each.
- **Test P / R / F1:** precision, recall and F1 of the fine-tuned model for that exact category
  on the 1,186-item held-out test set; `test n` is how many test items carry it. A dash means
  the category never appears in the test set or the model never predicts it.

### The 5 broad groups

| # | Broad group | Key | Rules | The question it asks | Categories | Labels (in-domain / OOD) | Test F1 (group) |
|---|---|---|---|---|---|---|---|
| A | Misleading or unbalanced content | `misleading_content` | 2210(d)(1)(A)–(F) | Is the message fair, balanced and not misleading? | 10 | 6,492 / 3,314 | 0.96 |
| B | Missing required disclosures | `required_disclosures` | 2210(d)(2)–(d)(8), 2212 | Does it include the specific items the rules require? | 9 | 2,445 / 947 | 0.83 |
| C | Approval and audience | `approval_and_audience` | 2210(a)(4), (b)(1), (b)(3) | Did the right person approve it, and does it reach the audience it claims? | 2 | 286 / 241 | 0.88 |
| D | Implied FINRA / SIPC endorsement | `regulator_endorsement` | 2210(e) | Does it suggest a regulator approved or guarantees something? | 1 | 345 / 147 | 0.96 |
| E | Product-specific rules | `product_rules` | 2220 | Does it follow the extra rules for options? | 1 | 0 / 16 | — (OOD only) |

Group test F1 counts a finding as correct when it names any category from the right group.
On the test set the model is far better at the group than at the exact category: inside group A
most mistakes are a neighbouring category with the same or a related rule.

### All 23 categories at a glance

| # | Group | Sub-group | Category label | Rule | Applies | Labels (in-domain / OOD) | Test n | Test P / R / F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | A | One-sided | `not_fair_and_balanced` | 2210(d)(1)(A) | C R I | 1,324 / 599 | 135 | 0.73 / 0.73 / 0.73 |
| 2 | A | One-sided | `product_specific_omission` | 2210(d)(1)(A) | C R I | 102 / 307 | 10 | — / 0.00 / 0.00 |
| 3 | A | One-sided | `buried_disclosure` | 2210(d)(1)(C) | C R I | 321 / 137 | 32 | 0.86 / 0.56 / 0.68 |
| 4 | A | Overclaiming | `promissory_or_guarantee` | 2210(d)(1)(B) | C R I | 1,040 / 429 | 100 | 0.88 / 0.91 / 0.90 |
| 5 | A | Overclaiming | `exaggerated_or_unwarranted_claim` | 2210(d)(1)(B) | C R I | 871 / 423 | 87 | 0.93 / 0.87 / 0.90 |
| 6 | A | Overclaiming | `false_or_misleading_statement` | 2210(d)(1)(B) | C R I | 749 / 378 | 73 | 0.62 / 0.75 / 0.68 |
| 7 | A | Overclaiming | `misleading_in_context` | 2210(d)(1)(D) | C R I | 542 / 239 | 54 | 0.47 / 0.30 / 0.36 |
| 8 | A | Predicting returns | `performance_projection` | 2210(d)(1)(F) | C R I | 1,336 / 610 | 130 | 0.97 / 0.95 / 0.96 |
| 9 | A | Predicting returns | `hypothetical_or_backtest_misuse` | 2210(d)(1)(F) | C R I | 207 / 113 | 21 | 0.87 / 0.95 / 0.91 |
| 10 | A | Other people's content | `third_party_content_adoption` | 2210(d)(1) / RN 17-18 | C R | 0 / 79 | 0 | — (OOD only) |
| 11 | B | Firm identity | `member_name_missing` | 2210(d)(3) | C R | 379 / 146 | 36 | 0.30 / 0.08 / 0.13 |
| 12 | B | Firm identity | `brokercheck_link_missing` | 2210(d)(8) | R | 107 / 3 | 11 | 0.55 / 0.55 / 0.55 |
| 13 | B | Fund performance and rankings | `fund_performance_disclosure` | 2210(d)(5) | C R | 595 / 230 | 58 | 0.92 / 0.81 / 0.86 |
| 14 | B | Fund performance and rankings | `fund_ranking_disclosure` | 2212 | R | 80 / 30 | 6 | 1.00 / 0.83 / 0.91 |
| 15 | B | Recommendations | `recommendation_disclosure` | 2210(d)(7) | R | 172 / 72 | 20 | 0.82 / 0.70 / 0.76 |
| 16 | B | Recommendations | `past_recommendations_reference` | 2210(d)(7)(C) | C R | 187 / 78 | 19 | 1.00 / 0.84 / 0.91 |
| 17 | B | Claims that need qualifiers | `unsupported_comparison` | 2210(d)(2) | R | 243 / 107 | 29 | 0.90 / 0.62 / 0.73 |
| 18 | B | Claims that need qualifiers | `tax_claim_incomplete` | 2210(d)(4) | C R | 334 / 144 | 43 | 1.00 / 0.88 / 0.94 |
| 19 | B | Claims that need qualifiers | `testimonial_disclosure` | 2210(d)(6) | C R | 348 / 137 | 28 | 1.00 / 0.89 / 0.94 |
| 20 | C | — | `unapproved_retail_communication` | 2210(b)(1) | R | 155 / 60 | 11 | 1.00 / 1.00 / 1.00 |
| 21 | C | — | `institutional_forwarding` | 2210(a)(4) / (b)(3) | I | 131 / 181 | 11 | 0.88 / 0.64 / 0.74 |
| 22 | D | — | `finra_endorsement_implied` | 2210(e) | C R I | 345 / 147 | 37 | 1.00 / 0.92 / 0.96 |
| 23 | E | — | `options_communication` | 2220 / 2210(d)(1) | C R | 0 / 16 | 0 | — (OOD only) |

The weakest categories on test are `member_name_missing` (the model has to notice that
something is absent), `product_specific_omission` (the model labels these
`not_fair_and_balanced`, which has the same rule) and `misleading_in_context` (mostly read as
`false_or_misleading_statement` or `not_fair_and_balanced`).

### A. Misleading or unbalanced content — 2210(d)(1) · `misleading_content`

The general content standards. They apply to **every** communication, including correspondence
and institutional communications, and they make up two thirds of all violation labels. The
categories overlap by nature (a "can't lose" promise is usually also unbalanced), so the data
follows one convention: name the category that best describes each problem, one entry per
category, with at most one natural extra.

#### One-sided

**1. `not_fair_and_balanced` — Not fair and balanced** · 2210(d)(1)(A) · C R I
- **Flags:** benefits presented without the matching risks, an omitted material fact or
  qualification, or cherry-picked facts. Also used for a misleading comparison in
  correspondence or institutional communications, where (d)(2) does not apply.
- **Not this category:** the missing item is a product-specific feature (use
  `product_specific_omission`); the risk is disclosed but hidden (use `buried_disclosure`).
- **Example** (`in_domain-000569`, video script, retail): "Our high-yield corporate bond
  portfolio is paying clients 7.25% a year, deposited monthly." The spot never mentions
  default, rate, call or liquidity risk.

**2. `product_specific_omission` — Product-specific omission** · 2210(d)(1)(A) · C R I
- **Flags:** a material feature of the specific product left out: surrender charges on an
  annuity, the illiquidity of a non-traded REIT or private placement, the daily reset of a
  leveraged ETF, call risk on a structured note, prepayment risk on a CMO (Rule 2216).
- **Not this category:** a general missing risk that is not tied to the product's structure
  (use `not_fair_and_balanced`).
- **Example** (`in_domain-008657`, website page, retail): "The correct rate is 5.15% per year,
  guaranteed for all seven years, with a $25,000 minimum". No surrender schedule and no
  mention that the guarantee depends on the insurer.
- **Note:** rare in-domain (102) but common in OOD (307), because OOD shifts the product.

**3. `buried_disclosure` — Buried disclosure** · 2210(d)(1)(C) · C R I
- **Flags:** a material risk or qualification that is present but placed in a footnote, legend
  or fine print in a way that keeps the reader from understanding it.
- **Not this category:** the disclosure is missing altogether (use `not_fair_and_balanced`).
- **Example** (`in_domain-006627`, SMS, correspondence): "*Bonds called by issuer; final
  redemption at par 6/1/25, no interest paid after that date." The call notice sits after the
  opt-out boilerplate while the body says payments will continue.

#### Overclaiming

**4. `promissory_or_guarantee` — Promissory language or guarantee** · 2210(d)(1)(B) · C R I
- **Flags:** "guaranteed", "can't lose", "risk-free" or "safe" for an investment with market
  risk, or a promise of an outcome.
- **Not this category:** a guarantee that is literally true and properly attributed (an
  FDIC-insured CD within limits, a fixed-annuity rate backed by the insurer with the
  claims-paying caveat). These appear as compliant near-misses.
- **Example** (`in_domain-000471`, IM, institutional): "which is guaranteed to hold its value,
  so there's no risk to the portfolio in the meantime" (about an ultra-short bond fund with a
  floating NAV).

**5. `exaggerated_or_unwarranted_claim` — Exaggerated or unwarranted claim** · 2210(d)(1)(B) · C R I
- **Flags:** unsupported superlatives or claims: "best fund", "#1 advisor", "never had a
  losing year".
- **Not this category:** a promise about outcomes or safety (use `promissory_or_guarantee`);
  a fund ranking without its required disclosures (use `fund_ranking_disclosure`).
- **Example** (`in_domain-004499`, email, correspondence): "the Wynfield Equity Income Fund is
  the single most reliable income fund available anywhere".

**6. `false_or_misleading_statement` — False or misleading statement** · 2210(d)(1)(B) · C R I
- **Flags:** a materially untrue statement of fact about fees, insurance coverage, product
  features or credentials.
- **Not this category:** each statement is true but the combination misleads (use
  `misleading_in_context`).
- **Example** (`in_domain-003282`, mass email, retail): "you can redeem a brokered CD through
  us at any time at its full face value with no penalty". Brokered CDs are sold on the
  secondary market, often below face value.

**7. `misleading_in_context` — Misleading in context** · 2210(d)(1)(D) · C R I
- **Flags:** statements that are individually true but mislead in combination or in the
  context in which they are made, for example a conclusion that the facts given contradict.
- **Not this category:** one statement is false on its own (use
  `false_or_misleading_statement`).
- **Example** (`in_domain-010113`, SMS, correspondence): "With 42 consecutive years of payments
  behind it, this is an income stream you can count on", sent after mentioning a 28% price
  drop and a 112% payout ratio.

#### Predicting returns

**8. `performance_projection` — Performance projection** · 2210(d)(1)(F) · C R I
- **Flags:** predicting or projecting performance, implying past performance will recur, or an
  unwarranted forecast. Applies to institutional communications too: FINRA's proposed exception
  for institutional and qualified-purchaser projections is not treated as current rule.
- **Not this category:** a clearly labeled hypothetical illustration of mathematical principles
  (such as compounding at an assumed rate) that does not predict an investment; output of an
  investment analysis tool under Rule 2214; a price target in a research report with a
  reasonable basis, valuation method and risks (Rule 2241).
- **Example** (`in_domain-002129`, print ad, retail): "is positioned to deliver rising income
  and double-digit total returns over the next decade".

**9. `hypothetical_or_backtest_misuse` — Hypothetical or backtest misuse** · 2210(d)(1)(F) · C R I
- **Flags:** backtested or hypothetical results presented as achievable, or an "illustration"
  that in substance projects an investment's performance.
- **Not this category:** the sender forecasts directly rather than through a backtest or model
  (use `performance_projection`).
- **Example** (`in_domain-009646`, newsletter, retail): "That backtest is a good guide to what
  a ladder built today can deliver over the next decade".

#### Other people's content

**10. `third_party_content_adoption` — Third-party content adoption** · 2210(d)(1), Regulatory Notice 17-18 · C R
- **Flags:** forwarding, sharing or linking third-party content (an article, ranking or chart)
  that the firm has adopted (for example by endorsing it) or become entangled with (for
  example by helping prepare it), when that content itself breaks the content standards.
- **Not this category:** the rep's own words break the standards (use the matching category
  above).
- **Example** (`ood-000180`, email, correspondence): "found this and honestly it sums it up
  perfectly. this is exactly why we like these for you!", forwarding a newsletter that calls
  CMOs completely safe.
- **Note:** OOD only (79). The model never saw it in training.

### B. Missing required disclosures — 2210(d)(2)–(d)(8), Rule 2212 · `required_disclosures`

Specific items the rules require. None of them applies to institutional communications; some
apply to retail communications only.

#### Firm identity

**11. `member_name_missing` — Member firm name missing** · 2210(d)(3) · C R
- **Flags:** a retail communication or correspondence that does not prominently disclose the
  name of the FINRA member firm (or the name under which it does business). A nonmember
  affiliate's name may appear if the relationship is clear.
- **Not this category:** the member firm is named where a reader would normally look for it,
  such as the signature block or the footer.
- **Example** (`in_domain-001781`, website page, retail): "Securities offered through a
  registered broker-dealer, member FINRA (www.finra.org)/SIPC." The page names only the rep's
  DBA, never the member firm.
- **Note:** the model's weakest category on test: it rarely notices the absence and sometimes
  flags it when the firm is named.

**12. `brokercheck_link_missing` — BrokerCheck link missing** · 2210(d)(8) · R
- **Flags:** a member's website page meant for retail investors, or a registered person's
  profile page, with no readily apparent BrokerCheck reference and hyperlink.
- **Not this category:** emails, texts and other messages that are not website pages
  (in-domain it is labeled only on website pages; OOD has one WhatsApp exception).
- **Example** (`in_domain-011686`, website page, retail): a profile page headed "Eleanor
  Vasquez-Hart, CRPC(R), Registered Representative | Granite Arch Securities, Inc." with no
  BrokerCheck link anywhere.

#### Fund performance and rankings

**13. `fund_performance_disclosure` — Fund performance disclosure** · 2210(d)(5) · C R
- **Flags:** performance of a non-money-market open-end fund shown without standardized 1-, 5-
  and 10-year (or since-inception) average annual total returns as of the latest calendar
  quarter-end, the maximum sales charge and the gross expense ratio, or with non-standardized
  returns more prominent than standardized ones. Applies to correspondence too, so a text
  quoting a fund's return counts.
- **Not this category:** a fund ranking (use `fund_ranking_disclosure`); a money market fund
  yield shown with its required disclosures.
- **Example** (`in_domain-000897`, SMS, correspondence): "the Pacific Crest Municipal Income
  Fund you bought in March is up 5.2% since your purchase".

**14. `fund_ranking_disclosure` — Fund ranking disclosure** · Rule 2212 · R
- **Flags:** a retail communication that uses an investment company ranking or rating without
  the ranking entity, the category, the number of funds in it, the period, the criteria, and
  that past performance is no guarantee.
- **Not this category:** a superlative with no ranking behind it (use
  `exaggerated_or_unwarranted_claim`).
- **Example** (`in_domain-004082`, mass email, retail): "has been ranked in the top 3% of its
  category - one of the highest-ranked funds anywhere!"

#### Recommendations

**15. `recommendation_disclosure` — Recommendation disclosure** · 2210(d)(7) · R
- **Flags:** a retail communication recommending a specific security (not a fund or variable
  product) without disclosing market-making, a financial interest, or a recent underwriting
  role; without offering supporting information; or, for a corporate equity, without the
  price at the time.
- **Not this category:** correspondence, institutional communications, research reports under
  Rules 2241 and 2242, or recommendations of registered funds and variable products only.
- **Example** (`in_domain-007240`, text blast, retail): "Our pick this week is Cedar Falls (IA)
  Community School District GO 4.10% due 06/01/2041", sent to 1,270 clients from a firm
  inventory block with no disclosures.

**16. `past_recommendations_reference` — Past recommendations reference** · 2210(d)(7)(C) · C R
- **Flags:** a reference to specific past recommendations that were or would have been
  profitable, without providing or offering a list of all such recommendations for at least
  the prior year and the required legend.
- **Not this category:** the full list and legend are provided or offered.
- **Example** (`in_domain-003710`, website page, retail): "We told clients to buy Corvane
  Systems at $38 in March 2023. It closed at $112 last Friday."

#### Claims that need qualifiers

**17. `unsupported_comparison` — Unsupported comparison** · 2210(d)(2) · R
- **Flags:** a retail communication that compares investments or services without the material
  differences: objectives, costs, liquidity, safety, guarantees or insurance, fluctuation of
  principal or return, tax features.
- **Not this category:** the same comparison in correspondence or an institutional
  communication (use `not_fair_and_balanced`).
- **Example** (`in_domain-006545`, mass email, retail): "Dividend stocks beat money markets on
  every measure that matters for long-term investors."

**18. `tax_claim_incomplete` — Incomplete tax claim** · 2210(d)(4) · C R
- **Flags:** "tax-free" or "tax-exempt" without saying which taxes apply (state tax, AMT,
  capital gains); calling tax-deferred income tax-free; or a tax-deferred vs taxable
  illustration that breaks (d)(4)(C).
- **Not this category:** the income really is free of all applicable taxes, or the claim names
  which taxes it covers.
- **Example** (`in_domain-000770`, mass email, retail): "Buy them inside your traditional IRA
  and every dividend and every dollar of gain is tax-free".

**19. `testimonial_disclosure` — Testimonial disclosure** · 2210(d)(6) · C R
- **Flags:** a testimonial about investment advice or performance without stating that it may
  not be representative of other clients' experience, that it is no guarantee of future
  performance, and, if more than $100 was paid, that it is a paid testimonial.
- **Not this category:** a client's comment about service (wait times, friendliness) rather
  than advice or performance.
- **Example** (`in_domain-009590`, mass email, retail): "Jen helped me rebuild my fund lineup
  after I retired and my portfolio grew 22% last year."

### C. Approval and audience — 2210(a)(4), (b)(1), (b)(3) · `approval_and_audience`

About how a message was handled, not what it says.

**20. `unapproved_retail_communication` — Unapproved retail communication** · 2210(b)(1) · R
- **Flags:** content that is really a retail communication (a template sent to more than 25
  retail investors in 30 days, a seminar deck, a mass email, a social post) that the text or
  metadata shows was already used without a registered principal's approval.
- **Not this category:** material submitted for review before use (that is the normal pre-use
  posture), or the exceptions in (b)(1) such as interactive forum posts.
- **Example** (`in_domain-011725`, newsletter, retail): "resending this morning's newsletter,
  which went out to my entire client list at 7:15 a.m." The 207-client send was never
  approved.

**21. `institutional_forwarding` — Institutional forwarding** · 2210(a)(4), (b)(3) · I
- **Flags:** a communication treated as institutional although the firm has reason to believe
  it will reach retail investors: plan participants, an adviser's retail clients, "feel free
  to share with your clients".
- **Labeling convention:** the item keeps the type `institutional_communication`, and the
  category records why that treatment fails.
- **Example** (`in_domain-011585`, IM, institutional): "Pass it around to your clients if you
  want, they'll get a lot out of the duration section."

### D. Implied FINRA / SIPC endorsement — 2210(e) · `regulator_endorsement`

**22. `finra_endorsement_implied` — Implied FINRA endorsement** · 2210(e) · C R I
- **Flags:** stating or implying that FINRA, another regulator or SIPC endorses, indemnifies,
  guarantees or approved the firm, its practices or a security: "FINRA-approved",
  "FINRA-regulated so your money is safe".
- **Not this category:** stating FINRA or SIPC membership; "Reviewed by FINRA" or "FINRA
  Reviewed" on a piece that Advertising Regulation actually reviewed.
- **Example** (`in_domain-008498`, radio script, retail): "our rollover process is
  FINRA-approved, so you know every step is being done the right way".

### E. Product-specific rules — Rule 2220 · `product_rules`

**23. `options_communication` — Options communication** · Rule 2220, 2210(d)(1) · C R
- **Flags:** options discussed with retail investors in a way that omits the risks, implies
  limited risk, or promotes options before the Options Disclosure Document is delivered.
- **Not this category:** misleading statements about a non-options product (use the group A
  category that fits).
- **Example** (`ood-001245`, webinar follow-up, retail): "A conservative, limited-risk way to
  turn a stagnant portfolio into a dependable income stream" (about covered-call writing).
- **Note:** OOD only (16). The model never saw it in training.

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
and a chat-focused top-up is planned in [docs/CHAT_TOPUP_PLAN.md](docs/CHAT_TOPUP_PLAN.md).

## Training and results

- **Training.** LoRA (rank 32, alpha 64, dropout 0.05) on 4-bit Phi-4-mini-instruct with mlx-lm
  on an M5 Pro.
  - Adapters on the attention and MLP projections in all 32 blocks.
  - AdamW; learning rate 2e-4 with 50 warmup steps, then cosine decay to 2e-6.
  - Batch 4 with 4-step gradient accumulation (effective batch 16).
  - 2 epochs (6,636 iterations, about 13 hours); loss on reviewer turns only.
  - Validation loss fell from 2.185 to 0.287 and was still at its minimum at the last
    iteration, so there is no sign of overfitting. Loss curve and summary:
    [results/training/](results/training/).
  - The same recipe runs on NVIDIA GPUs with PyTorch/TRL on the bf16 base
    (`training/run_cuda_pipeline.sh`; guide: [training/TRAINING_NVIDIA.md](training/TRAINING_NVIDIA.md)).
- **Results** (full tables, per-category numbers and plots: [results/README.md](results/README.md)).
  The zero-shot base is the same 4-bit Phi-4-mini with the same system prompt and no examples.

  | Metric | Fine-tuned, test (1,186) | Fine-tuned, OOD (5,800) | Zero-shot base, test | Zero-shot base, OOD |
  |---|---|---|---|---|
  | Valid JSON verdict | **99.8%** | **99.7%** | 67.8% | 65.5% |
  | Compliant / not compliant correct | **96.5%** | **95.3%** | 39.2% | 39.6% |
  | Communication type correct | **99.8%** | **98.6%** | 0.0% | 0.0% |
  | Severity correct | **89.5%** | **86.4%** | 1.2% | 2.0% |
  | Violation detection F1 | **0.97** | **0.96** | 0.44 | 0.48 |
  | Category F1 (exact) | **0.80** | **0.69** | 0.00 | 0.00 |
  | Category F1 (broad group) | **0.92** | **0.88** | 0.00 | 0.00 |
  | Rule-citation F1 | **0.82** | **0.74** | 0.00 | 0.00 |

  - **Chat** (SMS, instant message, text blast): compliant / not-compliant 97.4% on the 229 test
    chat items and 96.4% on the 605 OOD chat items; exact-category F1 0.79 and 0.69.
  - **Where it is weak:** naming the exact category among overlapping ones. Most misses are a
    neighbouring category with the same or a related rule, so F1 rises from 0.80 (exact) to 0.92 (broad group) on test.
    The weakest categories are `member_name_missing`, `product_specific_omission` and
    `misleading_in_context` (see the category table above). The two OOD-only categories are not
    detected at all.
  - **The base model** writes labels in its own words and calls about half of what it can
    parse compliant, so it is not usable without fine-tuning. See
    [results/README.md](results/README.md) for how to read its numbers.
  - **Improving it:** the plan for audits, retraining, DPO and GRPO is in
    [retraining/README.md](retraining/README.md).

## Examples: chat messages (test set)

Five SMS / instant-message items from the test set, with the final v0.1.0 model's answers; the model never saw them in training. Its predicted labels are shown in the format it outputs them: type · compliant · severity · violations.

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

On all 229 chat items in the test set, the compliant / not-compliant call is right 97.4% of the time and exact-category F1 is 0.79.

## Limits

- **Synthetic labels.** These are teacher-model judgments, not FINRA or regulator decisions; have a compliance SME spot-check before relying on them.
- **Chat is a minority of the data.** Only 19% of in-domain items are chat: 1,074 SMS, 896 IM and 261 retail text blasts. The rest are email, letters and public retail channels, so chat performance should be measured on its own slice.
- **Two categories never seen in training.** `options_communication` and `third_party_content_adoption` appear only in the OOD set, so expect low recall on them.
- **Scope.** It reviews only the text and metadata it is given. It does not check filing deadlines (2210(c)), recordkeeping or supervisory workflow, or rules outside 2210, 2212 and 2220.
