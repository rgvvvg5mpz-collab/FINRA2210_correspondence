"""Turn changes.json into a change memo: which rule paragraphs moved, which categories, brief lines
and digest lines they touch, and a draft approval file for mark_stale.py.

  .venv/bin/python retraining/00_rule_watch/make_memo.py --changes retraining/00_rule_watch/sample/out/changes.json \
      --out retraining/00_rule_watch/sample/out

Writes <out>/change_memo.md and <out>/approved_change.draft.json. The mapping is rule-based: cite
tokens such as "(d)(5)(A)(ii)" are read out of the changed lines and matched, by prefix, against
the rule cites in finra2210.taxonomy.VIOLATION_CATEGORIES ("2210(d)(5)" matches "(d)(5)" and any
deeper paragraph under it), against lines of data/chunks_r1/BRIEF.md and against the rule digest.
A change to a related rule file (rule_2212.md ...) maps by rule number; a Regulatory Notice maps
to categories that cite it ("RN 17-18") and is otherwise flagged for reading. No LLM is called:
the memo leaves a marked section for the human (or Claude-assisted) narrative and decision.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from finra2210.taxonomy import DIGEST_VERSION, RULE_DIGEST, VIOLATION_CATEGORIES  # noqa: E402

BRIEF = ROOT / "data" / "chunks_r1" / "BRIEF.md"
CITE_RE = re.compile(r"(?:\((?:[a-z]|\d{1,2}|[A-Z]|i{1,3}|iv|v|vi{0,3})\)){1,5}")


def cite_tokens(lines: list[str]) -> list[str]:
    """Paragraph cites found in the text, e.g. '(d)(5)(A)(ii)', deduplicated in order of appearance."""
    seen: list[str] = []
    for line in lines:
        for m in CITE_RE.finditer(line):
            tok = m.group(0)
            if tok.count("(") == 1:  # a bare "(a)", "(1)" or "(i)" is a list marker, not a paragraph path
                continue
            if tok not in seen:
                seen.append(tok)
    return seen


def prefixes(tok: str) -> list[str]:
    parts = re.findall(r"\([^)]+\)", tok)
    return ["".join(parts[: i + 1]) for i in range(len(parts))]


def rule_number(file: str) -> str | None:
    m = re.match(r"rule_(\d{4})\.md$", file)
    return m.group(1) if m else None


def notice_number(file: str) -> str | None:
    m = re.match(r"notice_(\d{2}-\d{2})\.(md|pdf)$", file)
    return m.group(1) if m else None


def affected_categories(src: dict) -> tuple[list[dict], list[str]]:
    """Categories whose rule cite is touched by this source's change, with the matching cites."""
    rule = rule_number(src["file"])
    notice = notice_number(src["file"])
    changed = src.get("added", []) + src.get("removed", [])
    toks = cite_tokens(changed)
    hits: dict[str, set[str]] = {}
    for cat, spec in VIOLATION_CATEGORIES.items():
        cite = spec["rule"]
        if rule == "2210":
            for tok in toks:
                for pre in prefixes(tok)[1:]:  # at least two components: "(d)(5)", never a bare "(d)"
                    if f"2210{pre}" in cite:
                        hits.setdefault(cat, set()).add(pre)
        elif rule and rule in cite:
            hits.setdefault(cat, set()).add(rule)
        elif notice and f"RN {notice}" in cite:
            hits.setdefault(cat, set()).add(f"RN {notice}")
    out = [{"category": c, "rule": VIOLATION_CATEGORIES[c]["rule"], "applies_to": VIOLATION_CATEGORIES[c]["applies_to"],
            "matched": sorted(hits[c], key=len)} for c in VIOLATION_CATEGORIES if c in hits]
    return out, toks


def grep_lines(text: str, needles: list[str]) -> list[tuple[int, str]]:
    found = []
    for i, line in enumerate(text.split("\n"), 1):
        if any(n in line for n in needles):
            found.append((i, line.strip()))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--changes", default="retraining/00_rule_watch/out/changes.json")
    ap.add_argument("--out", default=None, help="default: the directory of --changes")
    ap.add_argument("--max-excerpt", type=int, default=30, help="diff lines shown per source in the memo")
    args = ap.parse_args()
    changes = json.loads(Path(args.changes).read_text())
    out = Path(args.out) if args.out else Path(args.changes).parent
    out.mkdir(parents=True, exist_ok=True)
    brief = BRIEF.read_text() if BRIEF.exists() else ""
    changed = [s for s in changes["sources"] if s.get("status") == "changed"]
    other = [s for s in changes["sources"] if s.get("status") not in ("changed", "unchanged")]

    L = ["# Rule-change memo", "",
         f"Generated from `{args.changes}` ({changes.get('mode')} run" + (f", {changes['generated']}" if changes.get("mode") == "live" else "") + f"). "
         f"Dataset digest version: `{DIGEST_VERSION}`; sources archived {changes.get('archive_date')}.", ""]
    all_cats: dict[str, dict] = {}
    all_cites: set[str] = set()
    if not changed:
        L += ["**No source changed.** Nothing to decide.", ""]
    else:
        L += ["## Summary", "", "| Source | Lines added / removed | Cites touched | Categories affected |", "|---|---|---|---|"]
        per = []
        for s in changed:
            cats, toks = affected_categories(s)
            per.append((s, cats, toks))
            for c in cats:
                all_cats.setdefault(c["category"], c)
                all_cites.update(f"2210{m}" if m.startswith("(") else m for m in c["matched"])
            L.append(f"| `{s['file']}` | +{len(s.get('added', []))} / -{len(s.get('removed', []))} | {', '.join(toks[:8]) or '—'}"
                     f"{' …' if len(toks) > 8 else ''} | {', '.join('`' + c['category'] + '`' for c in cats) or 'none by cite (read the excerpt)'} |")
        L.append("")
        for s, cats, toks in per:
            L += [f"## {s['file']}", "", f"{s['title']}  ", f"<{s['url']}>  ", f"Archived {s['retrieved']}; new version saved as `{s.get('archived_as', '')}`.", "",
                  "### Changed text (unified diff, excerpt)", "", "```diff"]
            L += s.get("diff", [])[: args.max_excerpt]
            if s.get("diff_truncated") or len(s.get("diff", [])) > args.max_excerpt:
                L.append(f"... ({len(s.get('diff', []))} diff lines in changes.json)")
            L += ["```", "", "### Categories whose rule cite is touched", ""]
            if cats:
                L += ["| Category | Rule cite | Applies to | Matched on |", "|---|---|---|---|"]
                L += [f"| `{c['category']}` | {c['rule']} | {', '.join(c['applies_to'])} | {', '.join(c['matched'])} |" for c in cats]
            else:
                L.append("None matched by cite. The change may be guidance (how a category is applied) rather than a rule paragraph; read the excerpt and name the categories by hand below.")
            needles = [p for t in toks for p in prefixes(t)[1:3]] or ([f"Rule {rule_number(s['file'])}", rule_number(s["file"])] if rule_number(s["file"]) else [])
            if notice_number(s["file"]):
                needles += [notice_number(s["file"]), f"RN {notice_number(s['file'])}", f"Notice {notice_number(s['file'])}"]
            needles = [n for n in dict.fromkeys(needles) if n]
            bl = grep_lines(brief, needles)
            dl = grep_lines(RULE_DIGEST, needles)
            L += ["", f"### Lines in `data/chunks_r1/BRIEF.md` that mention these cites ({len(bl)})", ""]
            L += [f"- L{i}: {t[:160]}" for i, t in bl[:20]] or ["- none"]
            L += ["", f"### Lines in the rule digest (`taxonomy.RULE_DIGEST`) that mention these cites ({len(dl)})", ""]
            L += [f"- L{i}: {t[:160]}" for i, t in dl[:20]] or ["- none"]
            L.append("")
    if other:
        L += ["## Sources not compared", "", "| Source | Status |", "|---|---|"]
        L += [f"| `{s['file']}` | {s.get('status')} |" for s in other]
        L.append("")
    if "feeds" in changes:
        f = changes["feeds"]
        L += [f"## Feeds since {f['since']}", "", f"- Regulatory Notices: {len(f['notices'])}", f"- SR-FINRA filings on EDGAR: {len(f['filings'])}"]
        L += [f"  - {n.get('date') or ''} {n['title']} <{n['link']}>" for n in f["notices"][:20]]
        L += [f"  - {x.get('filed') or ''} {x.get('form') or ''} {x.get('name') or ''}" for x in f["filings"][:20]]
        L += [f"- feed errors: {e}" for e in f["errors"]] + [""]
    L += ["## Decision (human)", "",
          "Fill in, then save the approval file (`approved_change.draft.json` → `approved_change.json`) for `mark_stale.py`.", "",
          "- [ ] What changed, in one paragraph, and from what effective date:", "", "  _…_", "",
          "- [ ] Does the rule digest in `src/finra2210/taxonomy.py` need editing? Which lines?", "",
          "- [ ] Does `data/chunks_r1/BRIEF.md` need editing? Which lines?", "",
          "- [ ] Categories to add, split, retire, or re-cite (and the new cite string):", "",
          "- [ ] Which existing items are stale: all items in the categories above, only compliant near-misses on a topic, or only a cite string? "
          "(`mark_stale.py` takes categories, cites, topics and relabel_cites.)", "",
          "- [ ] Approved by (compliance owner, engineering owner) and date:", ""]
    (out / "change_memo.md").write_text("\n".join(L))

    draft = {"memo": str(out / "change_memo.md"), "digest_version_before": DIGEST_VERSION, "effective_date": None,
             "sources": [s["file"] for s in changed],
             "categories": sorted(all_cats), "cites": sorted(all_cites), "topics": [],
             "relabel_cites": {}, "notes": "DRAFT written by make_memo.py from cite matching; review every field before approving.",
             "approved_by": None, "approved_on": None}
    (out / "approved_change.draft.json").write_text(json.dumps(draft, indent=2))
    print(f"{len(changed)} changed source(s); categories affected by cite: {', '.join(sorted(all_cats)) or 'none'}")
    print(f"wrote {out / 'change_memo.md'} and {out / 'approved_change.draft.json'}")


if __name__ == "__main__":
    main()
