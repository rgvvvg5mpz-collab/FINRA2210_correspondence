"""Live confusion matrices for an eval, refreshed every 10 scored rows.

  .venv/bin/python evaluation/live_confusion.py                       # fine-tuned model on OOD, http://localhost:8765
  .venv/bin/python evaluation/live_confusion.py --preds runs/base-mlx/eval_ood.preds.jsonl --port 8766
  .venv/bin/python evaluation/live_confusion.py --gold data/splits/test.jsonl   # fine-tuned model on test

Reads a predictions file written by finra2210.evaluate (<eval out>.preds.jsonl, or a gzipped copy
such as results/predictions/*.preds.jsonl.gz) and the gold split, and serves a page that polls every
3 s. Without --preds it uses the running or finished eval in runs/phi4mini-2210-mlx, and in a fresh
clone (no runs/) the published v0.1.0 predictions in results/predictions/. Only the first multiple
of 10 rows is scored, so the numbers move in steps of 10. Unparsable predictions get their own
column. Category precision / recall are shown at four levels (exact, rule level, (d)(1) family,
broad group), computed by finra2210.evaluate.Scorer so they match eval_*.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from finra2210.analysis import analyze, load_gold, read_preds  # noqa: E402

STEP = 10


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live OOD Confusion</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e4e3de;--good:31,138,76;--bad:200,64,52;--warn:#b7791f;--bar:#2f6fde}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--card:#1f1f1d;--ink:#ecebe6;--mute:#9b9a94;--line:#33322f;--good:70,190,120;--bad:240,105,90;--warn:#e0a84a;--bar:#6b9cff}}
:root[data-theme=dark]{--bg:#151514;--card:#1f1f1d;--ink:#ecebe6;--mute:#9b9a94;--line:#33322f;--good:70,190,120;--bad:240,105,90;--warn:#e0a84a;--bar:#6b9cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0 0 2px}h2{font-size:14px;margin:0 0 10px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--mute);margin-bottom:14px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin:6px 0 16px}.bar>div{height:100%;background:var(--bar);transition:width .6s}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.kpi b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.kpi span{color:var(--mute);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,400px),1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:4px 5px;text-align:right;white-space:nowrap}th{font-weight:600;color:var(--mute);font-size:12px}
td.lab,th.lab{text-align:left}.cm td.v{text-align:center;border-radius:4px;min-width:44px}
.cm td.v small{display:block;font-size:10px;opacity:.75}
.corner{font-size:11px;color:var(--mute);text-align:left}
.cats td,.cats th{border-bottom:1px solid var(--line)}.lo{color:rgb(var(--bad));font-weight:600}
.flash{animation:f 1s}@keyframes f{from{background:rgba(var(--good),.25)}to{background:transparent}}
.foot{color:var(--mute);font-size:12px;margin-top:14px}
</style></head><body><main>
<h1>Live OOD confusion matrices</h1>
<div class="sub" id="sub">waiting for data…</div>
<div class="bar"><div id="prog" style="width:0"></div></div>
<div class="kpis" id="kpis"></div>
<div class="grid">
 <div class="card"><h2>Compliant vs violation</h2><div id="cm_comp"></div></div>
 <div class="card"><h2>Communication type</h2><div id="cm_ctype"></div></div>
 <div class="card"><h2>Severity</h2><div id="cm_sev"></div></div>
 <div class="card"><h2>Category precision / recall by level</h2><div id="levels"></div></div>
 <div class="card"><h2>Accuracy by OOD axis / difficulty</h2><div id="axes"></div></div>
</div>
<div class="card" style="margin-top:12px"><h2>Broad categories (a hit = any category from the right group)</h2><div id="groups"></div></div>
<div class="card" style="margin-top:12px"><h2>Violation categories by broad group (gold count, hits, misses, false alarms)</h2><div id="cats"></div></div>
<div class="card" style="margin-top:12px"><h2>Most common category swaps (missed → predicted instead)</h2><div id="swaps"></div></div>
<div class="foot">Rows = gold, columns = prediction; cell shade is the share of its row. Updates every 10 scored rows.</div>
</main><script>
const pct=x=>x==null?"–":(100*x).toFixed(1)+"%", f2=x=>x==null?"–":x.toFixed(2);
const short=s=>s.replace("_communication","").replace("institutional","institutional");
function cm(m){
  let h='<table class="cm"><tr><th class="corner">gold ↓ / pred →</th>'+m.cols.map(c=>`<th>${short(c)}</th>`).join("")+'</tr>';
  m.rows.forEach((r,i)=>{const tot=m.cells[i].reduce((a,b)=>a+b,0)||1;
    h+=`<tr><th class="lab">${short(r)}</th>`+m.cells[i].map((v,j)=>{const s=v/tot, diag=m.cols[j]===r;
      const col=diag?`rgba(var(--good),${.12+.6*s})`:(v?`rgba(var(--bad),${.08+.6*s})`:"transparent");
      return `<td class="v" style="background:${col}">${v}<small>${v?pct(s):""}</small></td>`}).join("")+"</tr>"});
  return h+"</table>"}
let last=-1;
async function tick(){
  let d; try{d=await (await fetch("api",{cache:"no-store"})).json()}catch(e){document.getElementById("sub").textContent="server not reachable";return}
  const s=d.stats, total=d.total;
  document.getElementById("prog").style.width=(100*s.n/total)+"%";
  const eta=d.rate_per_min?` · ${d.rate_per_min.toFixed(0)} rows/min · ETA ${Math.round((total-d.lines)/d.rate_per_min)} min`:"";
  document.getElementById("sub").textContent=`${d.label} — scored ${s.n.toLocaleString()} of ${total.toLocaleString()} (file has ${d.lines})${eta} · checked ${new Date().toLocaleTimeString()}`;
  if(s.n===last) return; last=s.n;
  const k=[["Compliant accuracy",pct(s.accuracy)],["Comm-type accuracy",pct(s.ctype_accuracy)],["Severity accuracy",pct(s.severity_accuracy)],["Category F1 (exact)",f2(s.category_levels[0].f1)],["Category F1 (rule level)",f2(s.category_levels[1].f1)],["Category F1 ((d)(1) family)",f2(s.category_levels[2].f1)],["Category F1 (broad group)",f2(s.category_levels[3].f1)],["Valid JSON",pct(s.parse_rate)]];
  document.getElementById("kpis").innerHTML=k.map(([a,b])=>`<div class="kpi flash"><b>${b}</b><span>${a}</span></div>`).join("");
  document.getElementById("cm_comp").innerHTML=cm(s.compliant);
  document.getElementById("cm_ctype").innerHTML=cm(s.ctype);
  document.getElementById("cm_sev").innerHTML=cm(s.severity);
  document.getElementById("levels").innerHTML='<table class="cats"><tr><th class="lab">level</th><th>precision</th><th>recall</th><th>F1</th></tr>'+
    s.category_levels.map(l=>`<tr><td class="lab">${l.level}<br><small style="color:var(--mute)">${l.note}</small></td><td>${f2(l.precision)}</td><td>${f2(l.recall)}</td><td><b>${f2(l.f1)}</b></td></tr>`).join("")+
    `<tr><td class="lab">Cite string written by the model<br><small style="color:var(--mute)">rule_cite_micro</small></td><td>${f2(s.rule_cite.precision)}</td><td>${f2(s.rule_cite.recall)}</td><td><b>${f2(s.rule_cite.f1)}</b></td></tr></table>`;
  const ax=Object.entries(s.axes).map(([a,v])=>`<tr><td class="lab">${a}</td><td>${v.n}</td><td>${pct(v.accuracy)}</td></tr>`).join("");
  const df=Object.entries(s.difficulty).map(([a,v])=>`<tr><td class="lab">${a}</td><td>${v.n}</td><td>${pct(v.accuracy)}</td></tr>`).join("");
  document.getElementById("axes").innerHTML=`<table><tr><th class="lab">axis</th><th>n</th><th>accuracy</th></tr>${ax}<tr><th class="lab">difficulty</th><th></th><th></th></tr>${df}</table>`;
  const lo=x=>x!=null&&x<.6?"lo":"";
  document.getElementById("groups").innerHTML='<table class="cats"><tr><th class="lab">broad group / category labels</th><th>gold</th><th>hit</th><th>missed</th><th>false alarm</th><th>precision</th><th>recall</th><th>F1</th></tr>'+
    s.broad_groups.map(g=>`<tr><td class="lab" style="white-space:normal"><b>${g.title}</b><br><small style="color:var(--mute)">${g.categories.join(", ")}</small></td><td>${g.support}</td><td>${g.hits}</td><td>${g.support-g.hits}</td><td>${g.false_alarms}</td><td>${g.support||g.false_alarms?f2(g.precision):"–"}</td><td class="${g.support?lo(g.recall):""}">${g.support?f2(g.recall):"–"}</td><td class="${g.support?lo(g.f1):""}"><b>${g.support?f2(g.f1):"–"}</b></td></tr>`).join("")+"</table>";
  let prev=null;
  document.getElementById("cats").innerHTML='<table class="cats"><tr><th class="lab">category</th><th>gold</th><th>hit</th><th>missed</th><th>false alarm</th><th>precision</th><th>recall</th><th>F1</th></tr>'+
    s.categories.map(c=>{const head=c.group!==prev?`<tr><th class="lab" colspan="8" style="padding-top:12px">${s.group_titles[c.group]||"not in taxonomy"}</th></tr>`:"";prev=c.group;
      return head+`<tr><td class="lab">${c.category}</td><td>${c.gold}</td><td>${c.tp}</td><td>${c.fn}</td><td>${c.fp}</td><td>${f2(c.precision)}</td><td class="${lo(c.recall)}">${f2(c.recall)}</td><td class="${lo(c.f1)}">${f2(c.f1)}</td></tr>`}).join("")+"</table>";
  document.getElementById("swaps").innerHTML=s.swaps.length?'<table class="cats"><tr><th class="lab">gold category missed</th><th class="lab">predicted instead</th><th>count</th></tr>'+s.swaps.map(w=>`<tr><td class="lab">${w.missed}</td><td class="lab">${w.predicted}</td><td>${w.count}</td></tr>`).join("")+"</table>":"none yet";
}
tick(); setInterval(tick,3000);
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None,
                    help="predictions file (.jsonl or .jsonl.gz); default: the running / finished eval in "
                         "runs/phi4mini-2210-mlx, or the published v0.1.0 predictions in results/predictions/")
    ap.add_argument("--gold", default="data/splits/ood.jsonl")
    ap.add_argument("--label", default=None)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    gold, order = load_gold(Path(args.gold))
    if args.preds is None:
        split = Path(args.gold).stem
        live = Path(f"runs/phi4mini-2210-mlx/eval_{split}.preds.jsonl")
        published = Path(f"results/predictions/fine_tuned_{split}.preds.jsonl.gz")
        args.preds = str(live if live.exists() or not published.exists() else published)
    preds_path = Path(args.preds)
    run = preds_path.name.split(".")[0] if preds_path.parent.name == "predictions" else preds_path.parent.name
    label = args.label or f"{run} on {Path(args.gold).stem}"
    history: list[tuple[float, int]] = []
    cache = {"key": None, "stats": None}

    def api() -> dict:
        preds = read_preds(preds_path)
        now = time.time()
        if not history or history[-1][1] != len(preds):
            history.append((now, len(preds)))
        recent = [h for h in history if now - h[0] < 900]
        rate = None
        if len(recent) >= 2 and recent[-1][1] > recent[0][1]:
            rate = (recent[-1][1] - recent[0][1]) / ((recent[-1][0] - recent[0][0]) / 60)
        key = len(preds) // STEP
        if cache["key"] != key:
            cache["key"], cache["stats"] = key, analyze(preds, gold, step=STEP)
        return {"label": label, "total": len(order), "lines": len(preds), "rate_per_min": rate, "stats": cache["stats"]}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api"):
                body, ctype = json.dumps(api()).encode(), "application/json"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"serving {label} at http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
