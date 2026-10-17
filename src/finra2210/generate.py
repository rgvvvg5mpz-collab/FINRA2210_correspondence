"""Generate labeled FINRA 2210 samples with Claude Fable 5.1.

Two modes:
  live   — concurrent async calls, good for pilots (fast feedback, full price).
  batch  — Message Batches API, 50% price, up to 24h. Use for the 10K / 5K runs.

Both modes write raw teacher output to <out>.raw.jsonl (one line per request,
with scenarios + parsed samples + usage) so postprocess.py can validate, dedup
and assemble. Everything is resumable: completed scenario_ids are skipped.

Usage:
  python -m finra2210.generate --n 48 --mode live --out data/pilot
  python -m finra2210.generate --n 10000 --mode batch --out data/in_domain
  python -m finra2210.generate --n 5000 --mode batch --ood --out data/ood
  python -m finra2210.generate --mode batch --collect --out data/in_domain   # poll + fetch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .prompts import TEACHER_SYSTEM, build_user_message
from .scenarios import make_scenarios
from .schema import SAMPLE_BATCH_SCHEMA

MODEL = "claude-fable-5-1"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _params(scenarios: list[dict], effort: str, max_tokens: int) -> dict:
    return dict(
        model=MODEL,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": TEACHER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": build_user_message(scenarios)}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SAMPLE_BATCH_SCHEMA}},
    )


def _extract(message) -> tuple[list[dict] | None, str]:
    """Return (samples, error)."""
    if message.stop_reason == "refusal":
        cat = getattr(getattr(message, "stop_details", None), "category", None)
        return None, f"refusal:{cat}"
    if message.stop_reason == "max_tokens":
        return None, "max_tokens"
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        return json.loads(text)["samples"], ""
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return None, f"parse:{e}"


def _usage(message) -> dict:
    u = message.usage
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "model": message.model,
    }


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _done_ids(raw_path: Path) -> set[str]:
    done = set()
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("samples"):
                done.update(s["scenario_id"] for s in rec["samples"])
    return done


def _append(raw_path: Path, rec: dict) -> None:
    with raw_path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# live mode
# ---------------------------------------------------------------------------

async def _run_live(chunks: list[list[dict]], raw_path: Path, effort: str, max_tokens: int, concurrency: int) -> None:
    client = anthropic.AsyncAnthropic(max_retries=4, timeout=600)
    sem = asyncio.Semaphore(concurrency)
    totals = {"in": 0, "out": 0, "cache_read": 0, "ok": 0, "fail": 0}

    async def one(i: int, scen: list[dict]) -> None:
        async with sem:
            t0 = time.time()
            try:
                # Fable: thinking is always on; refusal fallback routed server-side.
                msg = await client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **_params(scen, effort, max_tokens)
                )
            except anthropic.APIStatusError as e:
                _append(raw_path, {"scenarios": scen, "samples": None, "error": f"api:{e.status_code}:{e.message[:200]}"})
                totals["fail"] += 1
                print(f"[{i}] API error {e.status_code}: {e.message[:120]}", file=sys.stderr)
                return
            samples, err = _extract(msg)
            u = _usage(msg)
            _append(raw_path, {"scenarios": scen, "samples": samples, "error": err, "usage": u})
            totals["in"] += u["input_tokens"]; totals["out"] += u["output_tokens"]; totals["cache_read"] += u["cache_read"]
            totals["ok" if samples else "fail"] += 1
            print(f"[{i}] {len(samples) if samples else 0}/{len(scen)} samples  {time.time()-t0:5.1f}s  "
                  f"in={u['input_tokens']} cached={u['cache_read']} out={u['output_tokens']} {err}", file=sys.stderr)

    await asyncio.gather(*(one(i, c) for i, c in enumerate(chunks)))
    # Fable list price: $10 / $50 per MTok; cache reads $0.25 / MTok (fallback models are cheaper)
    cost = totals["in"] * 10 / 1e6 + totals["out"] * 50 / 1e6 + totals["cache_read"] * 0.25 / 1e6
    print(f"\nlive run: {totals['ok']} ok / {totals['fail']} failed requests; "
          f"uncached_in={totals['in']} cache_read={totals['cache_read']} out={totals['out']} "
          f"≈ ${cost:,.2f} at list price", file=sys.stderr)


# ---------------------------------------------------------------------------
# batch mode
# ---------------------------------------------------------------------------

def _submit_batch(chunks: list[list[dict]], out: Path, effort: str, max_tokens: int) -> None:
    client = anthropic.Anthropic()
    requests = [
        Request(custom_id=f"req-{i:05d}", params=MessageCreateParamsNonStreaming(**_params(c, effort, max_tokens)))
        for i, c in enumerate(chunks)
    ]
    batch_ids = []
    # API limit is 100K requests / 256 MB per batch; stay well under with 2K requests per batch.
    for part in _chunks(requests, 2000):
        b = client.messages.batches.create(requests=part)
        batch_ids.append(b.id)
        print(f"submitted batch {b.id} with {len(part)} requests", file=sys.stderr)
    state = {"batch_ids": batch_ids, "chunks": chunks, "effort": effort}
    (out.with_suffix(".batch.json")).write_text(json.dumps(state))
    print(f"state saved to {out.with_suffix('.batch.json')}; run again with --collect", file=sys.stderr)


def _collect_batch(out: Path, raw_path: Path, poll_s: int) -> None:
    client = anthropic.Anthropic()
    state = json.loads(out.with_suffix(".batch.json").read_text())
    chunk_by_id = {f"req-{i:05d}": c for i, c in enumerate(state["chunks"])}
    totals = {"in": 0, "out": 0, "cache_read": 0, "ok": 0, "fail": 0}
    for bid in state["batch_ids"]:
        while True:
            b = client.messages.batches.retrieve(bid)
            rc = b.request_counts
            print(f"{bid}: {b.processing_status} processing={rc.processing} ok={rc.succeeded} err={rc.errored}", file=sys.stderr)
            if b.processing_status == "ended":
                break
            time.sleep(poll_s)
        for r in client.messages.batches.results(bid):
            scen = chunk_by_id[r.custom_id]
            if r.result.type == "succeeded":
                samples, err = _extract(r.result.message)
                u = _usage(r.result.message)
                _append(raw_path, {"scenarios": scen, "samples": samples, "error": err, "usage": u})
                totals["in"] += u["input_tokens"]; totals["out"] += u["output_tokens"]; totals["cache_read"] += u["cache_read"]
                totals["ok" if samples else "fail"] += 1
            else:
                _append(raw_path, {"scenarios": scen, "samples": None, "error": f"batch:{r.result.type}"})
                totals["fail"] += 1
    cost = (totals["in"] * 10 / 1e6 + totals["out"] * 50 / 1e6 + totals["cache_read"] * 0.25 / 1e6) * 0.5
    print(f"\nbatch collect: {totals['ok']} ok / {totals['fail']} failed; out_tokens={totals['out']} ≈ ${cost:,.2f} (batch price)", file=sys.stderr)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=48, help="number of samples to generate")
    ap.add_argument("--seed", type=int, default=2210)
    ap.add_argument("--ood", action="store_true", help="sample OOD scenarios instead of in-domain")
    ap.add_argument("--mode", choices=["live", "batch"], default="live")
    ap.add_argument("--collect", action="store_true", help="batch mode: poll and fetch results of a submitted batch")
    ap.add_argument("--per-request", type=int, default=8, help="samples per teacher request")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--poll", type=int, default=120, help="batch poll interval seconds")
    ap.add_argument("--out", required=True, help="output stem, e.g. data/in_domain")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out.with_suffix(".raw.jsonl")

    if args.mode == "batch" and args.collect:
        _collect_batch(out, raw_path, args.poll)
        return

    scenarios = make_scenarios(args.n, args.seed, args.ood)
    done = _done_ids(raw_path)
    todo = [s for s in scenarios if s["scenario_id"] not in done]
    print(f"{len(scenarios)} scenarios, {len(done)} already done, {len(todo)} to generate", file=sys.stderr)
    if not todo:
        return
    chunks = _chunks(todo, args.per_request)

    if args.mode == "live":
        asyncio.run(_run_live(chunks, raw_path, args.effort, args.max_tokens, args.concurrency))
    else:
        _submit_batch(chunks, out, args.effort, args.max_tokens)


if __name__ == "__main__":
    main()
