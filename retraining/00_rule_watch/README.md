# 0. Rule-change watch

FINRA amends Rule 2210 and its guidance; the SEC approves the changes and sets effective dates.
Every label in this dataset was written against the rule text archived on 2026-10-14
(`DIGEST_VERSION` in `src/finra2210/taxonomy.py`). This stage notices when that text moves,
drafts a memo of what it touches, and lists the dataset items that need refreshing. The rest of
the kit (stages 1–6) then regenerates, retrains and gates as usual.

```bash
# offline, on the fixture (what run_examples.sh does)
.venv/bin/python retraining/00_rule_watch/watch_sources.py --fixtures retraining/00_rule_watch/sample/fixtures --out retraining/00_rule_watch/sample/out
.venv/bin/python retraining/00_rule_watch/make_memo.py --changes retraining/00_rule_watch/sample/out/changes.json
.venv/bin/python retraining/00_rule_watch/mark_stale.py --approved retraining/00_rule_watch/sample/approved_change.sample.json --out retraining/00_rule_watch/sample/out
```

## The three steps

| Step | Script | Automatic? | Output |
|---|---|---|---|
| Detect | `watch_sources.py` | yes (weekly cron) | `changes.json`: per source unchanged / changed / blocked, with a unified diff; changed versions archived as `<stem>.<date>.<ext>` |
| Triage | `make_memo.py` | draft yes, decision no | `change_memo.md`: changed paragraphs, the categories whose rule cite they touch, the brief and digest lines to edit; `approved_change.draft.json` |
| Mark | `mark_stale.py` | yes, from the approved file | `stale_ids.jsonl` (id, split, reasons) and a count table; `--relabel` rewrites a moved cite string in copies of the clean files |

**What stays human.** Approving the memo. A rule change can mean a new cite string (mechanical), a
changed meaning for a category (regenerate those items), or a new obligation (new category and new
scenarios). The compliance owner and the engineering owner decide which, edit the rule digest and
the agent brief, bump `DIGEST_VERSION`, and save `approved_change.json`. No LLM is called anywhere
in this stage; the cite mapping is rule-based so it is predictable, and Claude can be asked to
write the narrative section of the memo by hand.

## Detecting changes

`watch_sources.py` reads `data/raw_finra/SOURCES.md` (file, URL, retrieval date), normalises the
archived copy and the current version to one paragraph per line, hashes both and diffs them.

- **finra.org blocks plain HTTP clients** (403); the archive was captured with a browser. The
  script uses Playwright when it is installed (`pip install playwright && playwright install
  chromium`), otherwise it tries urllib with a browser user agent and records `blocked (403)`
  when that fails. A live diff of an HTML page can include navigation text; read it as "something
  changed, look", not as the edit itself.
- **eCFR XML** (SEC Rules 482, 34b-1) and **PDF notices** fetch fine; PDFs are compared by bytes.
- **Feeds.** `--check-feeds` also lists Regulatory Notices from FINRA's RSS feed and SR-FINRA
  filings from EDGAR full-text search since the archive date, which catch new guidance and SEC
  approval orders that a diff of existing pages cannot. Both are best effort; an unreachable feed
  is reported, not treated as a change.
- `--fail-on-change` exits 2 when any source changed, so a cron job or CI step can alert.
- `--fixtures DIR` reads "new version" files from a folder instead of the network, for tests.

Suggested schedule: weekly, with `--check-feeds --fail-on-change`, plus a run after any FINRA
notice the compliance team hears about.

## Triage and the memo

`make_memo.py` pulls paragraph cites out of the changed lines (`(d)(5)(A)(ii)` and the like) and
matches them, by prefix with at least two components, against the rule cites of the 23 categories:
a change under `(d)(5)` maps to `fund_performance_disclosure`; one under `(d)(1)` maps to every
content-standard category; a changed `rule_2212.md` maps by rule number; a notice maps to
categories that cite it (`RN 17-18`) and is otherwise flagged for reading. The memo also lists the
lines of `data/chunks_r1/BRIEF.md` and of the rule digest that mention those cites, so the human
knows exactly what to edit. The sample memo is `sample/out/change_memo.md`.

## Marking stale items

`mark_stale.py` scans `data/clean/*.clean.jsonl` with the approved file and marks an item stale when
it has a violation in an affected category, a violation whose cite starts with an affected cite,
a scenario that targets an affected category, or (compliant items) a near-miss topic that mentions
an affected term. For the fixture change this is 1,269 items, 897 in-domain and 372 OOD (summary in
`sample/out/stale_summary.json`).

Then:
- **Cite string only changed:** `--relabel` with `relabel_cites` writes relabeled copies under
  `<out>/relabeled/`; re-export the chat splits with `finra2210.split` because the messages embed
  the cite.
- **Meaning changed:** regenerate the stale scenarios through the agent loop (the specs are in each
  item's `scenario` field), snapshot, postprocess and split incrementally (`03_sft/build_round.py`).
- **New obligation:** sample new scenarios the way the sparse-category top-up did, after adding the
  category to `taxonomy.py`.
- **Test and OOD items that change** become a new eval-set version (`test_v2.jsonl`); they never
  enter training, and every model is re-scored on the new version so results stay comparable.

## Versioning

`DIGEST_VERSION` is stamped on every verdict by `finra2210.review` (`digest_version` in the JSON)
and on every eval report by `finra2210.evaluate`. Bump it when the digest or the brief changes after
an approved memo, so any past decision can be traced to the rule text it was judged by, and the
review desk can flag verdicts made under an older digest once a change takes effect.

## Sample (fake)

`sample/fixtures/rule_2210.md` is the archived rule text with an invented change to (d)(5)
(month-end rather than quarter-end performance). `sample/out/` holds what the three scripts write for
it; `sample/approved_change.sample.json` is the human-edited approval that drives the stale scan.
