"""Render MODEL_CARD.md as a standalone, styled HTML page (MODEL_CARD.html next to it).

  .venv-mlx/bin/python docs/build_model_card_html.py
  .venv-mlx/bin/python docs/build_model_card_html.py --src MODEL_CARD.md --out MODEL_CARD.html

Needs markdown-it-py (already in .venv-mlx; otherwise `pip install markdown-it-py`). The page has
no external dependencies: styles are inline, it follows the system light/dark setting, and it has
a table of contents built from the ## and ### headings. The output depends only on the Markdown,
so rebuilding an unchanged card changes nothing. Rerun after editing MODEL_CARD.md.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from markdown_it import MarkdownIt

CSS = """
:root{--bg:#fbfaf7;--card:#ffffff;--ink:#1c1b18;--mute:#66645d;--line:#e6e3db;--accent:#1f5fbf;--code:#f2f0ea;--th:#f6f4ef}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141413;--card:#1c1c1a;--ink:#ecebe6;--mute:#a09e97;--line:#34332f;--accent:#7aa7ff;--code:#26251f;--th:#22221f}}
:root[data-theme=dark]{--bg:#141413;--card:#1c1c1a;--ink:#ecebe6;--mute:#a09e97;--line:#34332f;--accent:#7aa7ff;--code:#26251f;--th:#22221f}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:16px}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.layout{display:grid;grid-template-columns:260px minmax(0,1fr);gap:32px;max-width:1440px;margin:0 auto;padding:28px 16px 64px}
nav.toc{position:sticky;top:16px;align-self:start;max-height:calc(100vh - 32px);overflow:auto;font-size:13px;border-right:1px solid var(--line);padding-right:16px}
nav.toc b{display:block;margin-bottom:8px;color:var(--mute);text-transform:uppercase;letter-spacing:.05em;font-size:11px}
nav.toc a{display:block;color:var(--ink);text-decoration:none;padding:3px 0}
nav.toc a.l3{padding-left:14px;color:var(--mute)}
nav.toc a:hover{color:var(--accent)}
details.toc-m{display:none;margin-bottom:16px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
details.toc-m a{display:block;color:var(--ink);text-decoration:none;padding:2px 0;font-size:14px}
details.toc-m a.l3{padding-left:14px;color:var(--mute)}
main{min-width:0}
h1{font-size:28px;line-height:1.25;margin:0 0 16px}
h2{font-size:22px;margin:40px 0 12px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:18px;margin:28px 0 10px}
h4{font-size:15px;margin:22px 0 8px;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}
p,li{max-width:80ch}
a{color:var(--accent)}
code{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
td code,th code{white-space:nowrap;overflow-wrap:normal}
pre{background:var(--code);padding:12px;border-radius:8px;overflow:auto}
pre code{padding:0;background:none}
.tw{overflow-x:auto;margin:12px 0 18px;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:var(--th);font-weight:600;white-space:nowrap}
tr:last-child td{border-bottom:none}
blockquote{margin:12px 0;padding:4px 14px;border-left:3px solid var(--line);color:var(--mute)}
main > p > strong:first-child{color:var(--ink)}
footer{margin-top:48px;color:var(--mute);font-size:12px;border-top:1px solid var(--line);padding-top:12px}
@media (max-width:900px){.layout{grid-template-columns:1fr;gap:0}nav.toc{display:none}details.toc-m{display:block}h1{font-size:23px}}
@media print{nav.toc,details.toc-m{display:none}.layout{display:block}h2{break-before:auto}}
"""


def slugify(text: str, seen: dict[str, int]) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"
    n = seen.get(s, 0)
    seen[s] = n + 1
    return s if n == 0 else f"{s}-{n}"


def build(src: Path, out: Path) -> None:
    md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False}).enable("table")
    body = md.render(src.read_text())

    seen: dict[str, int] = {}
    toc: list[tuple[int, str, str]] = []

    def add_id(m: re.Match) -> str:
        level, inner = int(m.group(1)), m.group(2)
        text = html.unescape(re.sub(r"<[^>]+>", "", inner))
        hid = slugify(text, seen)
        if level in (2, 3):
            toc.append((level, hid, text))
        return f'<h{level} id="{hid}">{inner}</h{level}>'

    body = re.sub(r"<h([1-4])>(.*?)</h\1>", add_id, body, flags=re.S)
    # category entries ("**12. `label` — Name**") get an anchor named after the label, and table
    # cells that hold just that label link to it
    anchors: set[str] = set()

    def add_anchor(m: re.Match) -> str:
        anchors.add(m.group(2))
        return f'<p id="cat-{m.group(2)}"><strong>{m.group(1)}<code>{m.group(2)}</code>'

    body = re.sub(r"<p><strong>(\d+\. )<code>([a-z_]+)</code>", add_anchor, body)
    body = re.sub(r"<td><code>([a-z_]+)</code></td>",
                  lambda m: f'<td><a href="#cat-{m.group(1)}"><code>{m.group(1)}</code></a></td>' if m.group(1) in anchors else m.group(0),
                  body)
    body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
    title_m = re.search(r"<h1[^>]*>(.*?)</h1>", body, flags=re.S)
    title = html.unescape(re.sub(r"<[^>]+>", "", title_m.group(1))) if title_m else "Model card"

    links = "".join(f'<a class="l{lvl}" href="#{hid}">{html.escape(t)}</a>' for lvl, hid, t in toc)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FINRA 2210 Model Card</title>
<meta name="description" content="{html.escape(title)}">
<style>{CSS}</style></head>
<body><div class="layout">
<nav class="toc" aria-label="Contents"><b>Contents</b>{links}</nav>
<main>
<details class="toc-m"><summary>Contents</summary>{links}</details>
{body}
<footer>Generated from {html.escape(src.name)} by docs/build_model_card_html.py. Edit the Markdown, then rerun the script.</footer>
</main></div></body></html>
"""
    out.write_text(page)
    print(f"wrote {out} ({len(page):,} bytes, {len(toc)} contents entries)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="MODEL_CARD.md")
    ap.add_argument("--out", default=None, help="default: <src> with .html")
    args = ap.parse_args()
    src = Path(args.src)
    build(src, Path(args.out) if args.out else src.with_suffix(".html"))


if __name__ == "__main__":
    main()
