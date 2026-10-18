"""Detect changes to the FINRA / SEC sources the dataset was built from.

  # offline check against "new version" files (how the sample and run_examples.sh work)
  .venv/bin/python retraining/00_rule_watch/watch_sources.py --fixtures retraining/00_rule_watch/sample/fixtures \
      --out retraining/00_rule_watch/sample/out

  # live check: re-fetch every source in data/raw_finra/SOURCES.md and compare
  .venv/bin/python retraining/00_rule_watch/watch_sources.py --out retraining/00_rule_watch/out --fail-on-change

  # live feeds only: new Regulatory Notices and SR-FINRA filings since the archive date
  .venv/bin/python retraining/00_rule_watch/watch_sources.py --feeds-only --out retraining/00_rule_watch/out

For every source row in SOURCES.md the script loads the archived copy, gets the current version
(from --fixtures, or by fetching the URL), normalises both to plain text, hashes them and writes a
unified diff for anything that changed. Changed versions are archived next to the original as
<stem>.<YYYY-MM-DD><ext> (in --archive-dir; fixture runs archive under --out instead). Output:
<out>/changes.json, read by make_memo.py.

How fetching works, honestly:
- finra.org answers plain HTTP clients with 403 (the archive was captured with a browser). The
  script tries Playwright if it is installed (`pip install playwright && playwright install
  chromium`); otherwise it tries urllib with a browser user agent and records "blocked" when
  that fails. Fetched pages are stripped of HTML and compared on normalised text, so a live diff
  can include navigation boilerplate; read it as "something changed, look", not as the edit itself.
- eCFR XML and PDF sources fetch fine with urllib.
- --check-feeds (or --feeds-only) also asks FINRA's Regulatory Notice RSS feed and the SEC EDGAR
  full-text search for SR-FINRA filings newer than the archive. Both are read-only GET requests
  and both are best effort: a feed that cannot be reached is reported, not treated as a change.
Standard library only; no LLM is called.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES_MD = ROOT / "data" / "raw_finra" / "SOURCES.md"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
NOTICES_RSS = "https://www.finra.org/rss/notices"
EDGAR_FTS = "https://efts.sec.gov/LATEST/search-index"
HEADER_KEYS = ("Title:", "Source URL:", "Retrieved:", "Retrieval method:")


# ----------------------------------------------------------------------------- sources
def read_sources(path: Path = SOURCES_MD) -> list[dict]:
    """Rows of the SOURCES.md table: file, title, url, retrieved."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| filename") or line.startswith("|---"):
            continue
        cells = [c.strip().replace("\x00", "|") for c in line.strip().replace("\\|", "\x00").strip("|").split("|")]
        if len(cells) < 4 or not re.search(r"\.(md|pdf|xml)$", cells[0]):
            continue
        m = re.search(r"https?://\S+", cells[2])
        rows.append({"file": cells[0], "title": cells[1], "url": m.group(0).rstrip(")") if m else "", "retrieved": cells[3]})
    return rows


def strip_header(text: str) -> str:
    """Drop the 4-line provenance header the archived .md files carry."""
    lines = text.split("\n")
    i = 0
    while i < len(lines) and i < 6 and lines[i].startswith(HEADER_KEYS):
        i += 1
    return "\n".join(lines[i:])


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in ("p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr", "td", "th", "section", "article"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(doc: str) -> str:
    p = _Text()
    p.feed(doc)
    return html.unescape("".join(p.parts))


def normalise(text: str) -> list[str]:
    """One paragraph per line, whitespace collapsed, empties dropped: what gets hashed and diffed."""
    out = []
    for line in text.replace("\r", "").split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return out


def sha(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ----------------------------------------------------------------------------- fetching
def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,application/xml,application/pdf,*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_with_playwright(url: str) -> str | None:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        return None
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(user_agent=UA)
        page.goto(url, wait_until="networkidle", timeout=60000)
        for sel in ("main", "article", "body"):
            el = page.query_selector(sel)
            if el:
                text = el.inner_text()
                break
        browser.close()
    return text


def current_version(src: dict, fixtures: Path | None) -> tuple[str, bytes | None, str | None]:
    """(status, raw bytes or None, fetch method)."""
    if fixtures is not None:
        f = fixtures / src["file"]
        return ("fixture", f.read_bytes(), "fixture") if f.exists() else ("no fixture", None, None)
    if not src["url"]:
        return ("no url", None, None)
    if src["file"].endswith(".md") and "finra.org" in src["url"]:
        text = None
        try:
            text = fetch_with_playwright(src["url"])
        except Exception as e:  # playwright installed but failed
            return (f"playwright error: {e.__class__.__name__}", None, "playwright")
        if text is not None:
            return ("fetched", text.encode(), "playwright")
    try:
        raw = _get(src["url"])
    except urllib.error.HTTPError as e:
        return (f"blocked ({e.code})" if e.code in (401, 403) else f"http {e.code}", None, "urllib")
    except Exception as e:
        return (f"unreachable: {e.__class__.__name__}", None, "urllib")
    if src["file"].endswith(".md"):
        return ("fetched", html_to_text(raw.decode("utf-8", "replace")).encode(), "urllib+html")
    return ("fetched", raw, "urllib")


# ----------------------------------------------------------------------------- feeds
def check_feeds(since: date, rss_url: str) -> dict:
    out: dict = {"since": since.isoformat(), "notices": [], "filings": [], "errors": []}
    try:
        xml = _get(rss_url).decode("utf-8", "replace")
        for item in re.findall(r"<item>(.*?)</item>", xml, flags=re.S):
            title = html.unescape(re.sub(r"<[^>]+>", "", re.search(r"<title>(.*?)</title>", item, re.S).group(1))).strip()
            link = (re.search(r"<link>(.*?)</link>", item, re.S) or re.search(r"<guid[^>]*>(.*?)</guid>", item, re.S))
            pub = re.search(r"<pubDate>(.*?)</pubDate>", item, re.S)
            when = None
            if pub:
                try:
                    when = datetime.strptime(pub.group(1).strip()[:25], "%a, %d %b %Y %H:%M:%S").date()
                except ValueError:
                    when = None
            if when is None or when >= since:
                out["notices"].append({"title": title, "link": link.group(1).strip() if link else "", "date": when.isoformat() if when else None})
    except Exception as e:
        out["errors"].append(f"notices rss ({rss_url}): {e.__class__.__name__}: {e}")
    try:
        q = urllib.parse.urlencode({"q": '"SR-FINRA"', "dateRange": "custom", "startdt": since.isoformat(), "enddt": date.today().isoformat()})
        data = json.loads(_get(f"{EDGAR_FTS}?{q}").decode())
        for hit in data.get("hits", {}).get("hits", [])[:50]:
            s = hit.get("_source", {})
            out["filings"].append({"name": ", ".join(s.get("display_names", [])), "form": s.get("form"), "filed": s.get("file_date"),
                                   "id": hit.get("_id")})
    except Exception as e:
        out["errors"].append(f"EDGAR full-text search: {e.__class__.__name__}: {e}")
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(SOURCES_MD))
    ap.add_argument("--fixtures", default=None, help="directory of 'new version' files named like the sources (offline mode)")
    ap.add_argument("--out", default="retraining/00_rule_watch/out")
    ap.add_argument("--archive-dir", default=None, help="where changed versions are saved (default data/raw_finra; with --fixtures, <out>/archive)")
    ap.add_argument("--only", nargs="*", default=None, help="check only these source files")
    ap.add_argument("--check-feeds", action="store_true", help="also query the FINRA notices RSS and EDGAR for SR-FINRA filings")
    ap.add_argument("--feeds-only", action="store_true")
    ap.add_argument("--rss-url", default=NOTICES_RSS)
    ap.add_argument("--max-diff-lines", type=int, default=400)
    ap.add_argument("--fail-on-change", action="store_true", help="exit 2 if any source changed (for cron / CI alerts)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixtures = Path(args.fixtures) if args.fixtures else None
    archive_dir = Path(args.archive_dir) if args.archive_dir else (out / "archive" if fixtures else ROOT / "data" / "raw_finra")
    sources = read_sources(Path(args.sources))
    if args.only:
        sources = [s for s in sources if s["file"] in set(args.only)]
    archive_date = max((s["retrieved"] for s in sources), default="2026-10-14")
    today = date.today().isoformat()
    stamp = today if fixtures is None else "fixture"

    results = []
    if not args.feeds_only:
        for src in sources:
            archived_path = ROOT / "data" / "raw_finra" / src["file"]
            rec = {"file": src["file"], "title": src["title"], "url": src["url"], "retrieved": src["retrieved"]}
            if not archived_path.exists():
                rec.update(status="archived copy missing")
                results.append(rec)
                continue
            status, raw, method = current_version(src, fixtures)
            rec["fetch"] = method
            if raw is None:
                rec.update(status=status)
                results.append(rec)
                continue
            if src["file"].endswith(".pdf"):
                a, b = hashlib.sha256(archived_path.read_bytes()).hexdigest(), hashlib.sha256(raw).hexdigest()
                rec.update(status="unchanged" if a == b else "changed", archived_sha=a, new_sha=b, compared="bytes")
            else:
                old_lines = normalise(strip_header(archived_path.read_text(errors="replace")))
                new_lines = normalise(strip_header(raw.decode("utf-8", "replace")))
                a, b = sha(old_lines), sha(new_lines)
                rec.update(archived_sha=a, new_sha=b, archived_lines=len(old_lines), new_lines=len(new_lines), compared="text")
                if a == b:
                    rec["status"] = "unchanged"
                else:
                    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"archived {src['retrieved']}", tofile=f"current {stamp}", lineterm="", n=1))
                    rec["status"] = "changed"
                    rec["diff"] = diff[: args.max_diff_lines]
                    rec["diff_truncated"] = len(diff) > args.max_diff_lines
                    rec["added"] = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
                    rec["removed"] = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]
            if rec["status"] == "changed":
                archive_dir.mkdir(parents=True, exist_ok=True)
                stem, ext = src["file"].rsplit(".", 1)
                dest = archive_dir / f"{stem}.{'fixture' if fixtures else today}.{ext}"
                dest.write_bytes(raw)
                rec["archived_as"] = str(dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest)
            results.append(rec)

    report = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if fixtures is None else "fixture run",
              "mode": "fixtures" if fixtures else "live", "sources_md": str(Path(args.sources)), "archive_date": archive_date,
              "sources": results}
    if args.check_feeds or args.feeds_only:
        report["feeds"] = check_feeds(date.fromisoformat(archive_date), args.rss_url)
    (out / "changes.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"{len(results)} sources checked ({report['mode']}): " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for r in results:
        if r["status"] == "changed":
            print(f"  CHANGED {r['file']}: +{len(r.get('added', []))} / -{len(r.get('removed', []))} lines -> {r.get('archived_as', '')}")
    if "feeds" in report:
        f = report["feeds"]
        print(f"feeds since {f['since']}: {len(f['notices'])} notices, {len(f['filings'])} SR-FINRA filings" + (f"; errors: {len(f['errors'])}" if f["errors"] else ""))
        for e in f["errors"]:
            print("  " + e)
    print(f"wrote {out / 'changes.json'}")
    if args.fail_on_change and counts.get("changed"):
        sys.exit(2)


if __name__ == "__main__":
    main()
