# Rule-change memo

Generated from `retraining/00_rule_watch/sample/out/changes.json` (fixtures run). Dataset digest version: `2026-10-14`; sources archived 2026-10-14.

## Summary

| Source | Lines added / removed | Cites touched | Categories affected |
|---|---|---|---|
| `rule_2210.md` | +2 / -2 | (d)(5)(A) | `fund_performance_disclosure` |

## rule_2210.md

2210. Communications with the Public | FINRA.org  
<https://www.finra.org/rules-guidance/rulebooks/finra-rules/2210>  
Archived 2026-10-14; new version saved as `retraining/00_rule_watch/sample/out/archive/rule_2210.fixture.md`.

### Changed text (unified diff, excerpt)

```diff
--- archived 2026-10-14
+++ current fixture
@@ -134,3 +134,3 @@
 (A) Retail communications and correspondence that present non-money market fund open-end management investment company performance data as permitted by Securities Act Rule 482 and Rule 34b-1 under the Investment Company Act must disclose:
-(i) the standardized performance information mandated by Securities Act Rule 482 and Rule 34b-1 under the Investment Company Act; and
+(i) the standardized performance information mandated by Securities Act Rule 482 and Rule 34b-1 under the Investment Company Act, current to the most recent month-end rather than calendar quarter-end; and
 (ii) to the extent applicable:
@@ -138,3 +138,3 @@
 b. the total annual fund operating expense ratio, gross of any fee waivers or expense reimbursements, as stated in the fee table of the investment company's prospectus described in paragraph (d)(5)(A)(ii)(a).
-(B) All of the information required by paragraph (d)(5)(A) must be set forth prominently, and in any print advertisement, in a prominent text box that contains only the required information and, at the member's option, comparative performance and fee data and disclosures required by Securities Act Rule 482 and Rule 34b-1 under the Investment Company Act.
+(B) All of the information required by paragraph (d)(5)(A) must be set forth prominently and updated within 10 business days of each month-end, and in any print advertisement, in a prominent text box that contains only the required information and, at the member's option, comparative performance and fee data and disclosures required by Securities Act Rule 482 and Rule 34b-1 under the Investment Company Act.
 (6) Testimonials
```

### Categories whose rule cite is touched

| Category | Rule cite | Applies to | Matched on |
|---|---|---|---|
| `fund_performance_disclosure` | 2210(d)(5) | correspondence, retail_communication | (d)(5) |

### Lines in `data/chunks_r1/BRIEF.md` that mention these cites (4)

- L101: (d)(5) Fees, expenses and standardized performance: retail communications and
- L189: fund_performance_disclosure          cite 2210(d)(5)               applies to: correspondence, retail_communication
- L234: it RIGHT — historical returns with full 2210(d)(5) disclosure, a hypothetical compounding
- L306: (d)(4), (d)(5), (d)(6) apply to correspondence and retail communications but NOT institutional.

### Lines in the rule digest (`taxonomy.RULE_DIGEST`) that mention these cites (1)

- L97: (d)(5) Fees, expenses and standardized performance: retail communications and

## Sources not compared

| Source | Status |
|---|---|
| `rule_2010.md` | no fixture |
| `rule_2020.md` | no fixture |
| `rule_2111.md` | no fixture |
| `rule_2211.md` | no fixture |
| `rule_2212.md` | no fixture |
| `rule_2213.md` | no fixture |
| `rule_2214.md` | no fixture |
| `rule_2215.md` | no fixture |
| `rule_2216.md` | no fixture |
| `rule_2220.md` | no fixture |
| `rule_2241.md` | no fixture |
| `rule_2330.md` | no fixture |
| `rule_3110.md` | no fixture |
| `rule_3240.md` | no fixture |
| `rule_4511.md` | no fixture |
| `rule_4512.md` | no fixture |
| `notice_12-29.md` | no fixture |
| `notice_12-29.pdf` | no fixture |
| `notice_15-50.md` | no fixture |
| `notice_15-50.pdf` | no fixture |
| `notice_16-41.md` | no fixture |
| `notice_16-41.pdf` | no fixture |
| `notice_17-18.md` | no fixture |
| `notice_17-18.pdf` | no fixture |
| `notice_19-31.md` | no fixture |
| `notice_19-31.pdf` | no fixture |
| `notice_19-32.md` | no fixture |
| `notice_19-32.pdf` | no fixture |
| `notice_20-21.md` | no fixture |
| `notice_20-21.pdf` | no fixture |
| `notice_24-09.md` | no fixture |
| `notice_24-09.pdf` | no fixture |
| `sec_rule_34b-1.xml` | no fixture |
| `sec_rule_482.xml` | no fixture |
| `faq_advertising_regulation.md` | no fixture |
| `keytopic_advertising_regulation.md` | no fixture |
| `oversight_report_communications.md` | no fixture |

## Decision (human)

Fill in, then save the approval file (`approved_change.draft.json` → `approved_change.json`) for `mark_stale.py`.

- [ ] What changed, in one paragraph, and from what effective date:

  _…_

- [ ] Does the rule digest in `src/finra2210/taxonomy.py` need editing? Which lines?

- [ ] Does `data/chunks_r1/BRIEF.md` need editing? Which lines?

- [ ] Categories to add, split, retire, or re-cite (and the new cite string):

- [ ] Which existing items are stale: all items in the categories above, only compliant near-misses on a topic, or only a cite string? (`mark_stale.py` takes categories, cites, topics and relabel_cites.)

- [ ] Approved by (compliance owner, engineering owner) and date:
