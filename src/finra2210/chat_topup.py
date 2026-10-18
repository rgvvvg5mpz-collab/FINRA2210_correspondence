"""Chat top-up: SMS / instant-message / text-blast scenarios across every category that can occur in chat.

  python -m finra2210.chat_topup --stats                                   # plan only: print the count tables
  python -m finra2210.chat_topup --out data/chunks_chat/in_domain --per-chunk 24

The deployed reviewer sees chat, but only ~19% of the round-1 + sparse top-up data is chat, and the
retail-only categories almost never appear there (a text blast is the only retail chat channel in the
base sampler). Every spec here is an ordinary in-domain spec (scenarios.sample_in_domain) whose
channel, communication type, recipients, tone, length and violation_categories are forced into a
chat CELL (channel x communication type). Violation specs carry a target category (first in
violation_categories); compliant specs carry a chat-specific near-miss. Each spec also carries a
`chat_angle` (group chat, thread, emoji, off-channel request, one-liner, ...) and a `chat_format`.

brokercheck_link_missing is website_page only, so it has no chat target.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from . import taxonomy as T
from .scenarios import _fit_product, _near_miss, _pick_categories, _weighted_dict, sample_in_domain

COMPLIANT_FRAC = 0.40
ONE_LINER_FRAC = 0.20   # of violation specs: one short message that carries 2+ violations

# (channel, communication_type) -> number of specs
CELLS = {
    ("sms_text", "correspondence"): 600,
    ("instant_message", "correspondence"): 500,
    ("text_message_blast", "retail_communication"): 550,
    ("sms_text", "retail_communication"): 150,          # same text to >25 clients, one at a time or as a group text
    ("instant_message", "retail_communication"): 150,   # client group chat / broadcast channel with >25 retail members
    ("sms_text", "institutional_communication"): 250,
    ("instant_message", "institutional_communication"): 200,
}

# target (primary) category -> number of violation specs; sums to the violation slots in CELLS
TARGETS = {
    # retail only -> retail chat cells
    "unapproved_retail_communication": 100,
    "unsupported_comparison": 80,
    "recommendation_disclosure": 80,
    "fund_ranking_disclosure": 50,
    # institutional only
    "institutional_forwarding": 60,
    # correspondence + retail
    "member_name_missing": 60,
    "fund_performance_disclosure": 60,
    "tax_claim_incomplete": 50,
    "testimonial_disclosure": 50,
    "past_recommendations_reference": 50,
    "third_party_content_adoption": 50,
    "options_communication": 40,
    # all three types
    "performance_projection": 100,
    "promissory_or_guarantee": 100,
    "not_fair_and_balanced": 100,
    "exaggerated_or_unwarranted_claim": 80,
    "false_or_misleading_statement": 70,
    "misleading_in_context": 70,
    "buried_disclosure": 50,
    "hypothetical_or_backtest_misuse": 50,
    "product_specific_omission": 50,
    "finra_endorsement_implied": 40,
}

# categories never added as a natural extra (they need a setup of their own)
_NO_EXTRA = {"brokercheck_link_missing", "unapproved_retail_communication", "institutional_forwarding",
             "options_communication", "third_party_content_adoption"}

_PRODUCT_FOR = {"options_communication": "listed options / covered calls"}

_PROFILE = {
    "correspondence": [("a single existing retail client", (1, 1)), ("a prospect who is not yet a customer", (1, 1)),
                       ("a small group chat of retail clients (family members, a couple, a few friends)", (2, 12)),
                       ("a retail client list just under the threshold, texted one at a time", (18, 25))],
    "retail_communication": [("the rep's or branch's retail client list (mass distribution over the threshold)", (26, 2000))],
    "institutional_communication": [("a single institutional investor (pension CIO, bank treasury, insurance GA)", (1, 3)),
                                    ("an RIA who serves retail clients", (1, 2))],
}
_RETAIL_PROFILE = {
    "text_message_blast": "the rep's or branch's full retail client list, sent as a text blast through the firm's texting platform",
    "sms_text": "more than 25 retail clients this month, the same text sent one at a time or as a large group text",
    "instant_message": "a client group chat / broadcast channel with more than 25 retail members",
}

CHAT_FORMATS = {
    "sms_text": ["a single text", "a thread of 3-8 texts with timestamps (client lines give context; the rep's lines are under review)",
                 "two or three back-to-back texts from the rep"],
    "instant_message": ["a single IM", "an IM thread of 3-8 messages with names and timestamps",
                        "a group chat excerpt with several participants and timestamps"],
    "text_message_blast": ["a single blast text with opt-out line", "a blast text plus one follow-up reply to a client who answered it"],
}

ANGLES = {
    "all": [
        "emoji and texting abbreviations (lol, tbh, fyi, 🚀📈) carry part of the meaning",
        "the rep answers a client's direct question ('is this safe?', 'will it go back up?', 'should I buy more?')",
        "a link or screenshot is pasted in with one line of the rep's own commentary",
        "voice-to-text typos and no punctuation, as if dictated while driving",
        "a quick reply that is fine on its own but misleading given the earlier messages in the thread",
        "the rep reacts to a market move the same day (urgent, short)",
    ],
    "correspondence": [
        "client asks to move the conversation to the rep's personal cell / WhatsApp / Signal or to delete the thread (off-channel request); the rep's reply is part of the message",
        "small family group chat (spouses, adult kids) about one shared decision",
        "follow-up text after a meeting summarizing what was discussed",
        "prospecting text to someone the rep met at an event",
    ],
    "retail_communication": [
        "text blast promoting a product, event or idea to the whole book",
        "the rep treats a text sent individually to 30+ clients as one-off correspondence",
        "a client group chat / broadcast channel the rep runs for 'VIP' clients",
        "short-code style message with a link to more information and a STOP opt-out",
    ],
    "institutional_communication": [
        "IM on a trading / sales chat with a bank, insurer or pension desk (indicative levels, color, axes)",
        "wholesaler texting an RIA about a product the RIA will use with retail clients",
        "chat with an institutional client that drifts into material the recipient will pass along",
    ],
}

NEAR_MISS_CHAT = {
    "all": [
        "client asks 'is this guaranteed?' and the rep says no, names the main risk in one line and offers a call",
        "emoji and casual abbreviations, but no performance claim, promise or recommendation",
        "brief reassurance after a market drop that does not predict a rebound",
        "rep declines to text a return number and offers to send full standardized performance by email",
        "rep shares a news link with neutral framing ('FYI, worth a read, happy to discuss') and no endorsement of its claims",
    ],
    "correspondence": [
        "client asks to switch to WhatsApp / personal cell; the rep keeps the conversation on the firm's approved, archived platform",
        "group chat reply that corrects a family member's overstatement ('it's not guaranteed, here's why')",
        "routine logistics text (meeting time, forms to sign, transfer status) with no claims",
    ],
    "retail_communication": [
        "text blast that names the firm, shows a principal approval code and a STOP opt-out, and is balanced",
        "logistics-only blast (office closure, tax-form timing) that makes no recommendation and promotes no product",
        "group-chat post that was approved by a principal before use (approval reference shown) and stays balanced",
    ],
    "institutional_communication": [
        "IM to a bank or pension desk with indicative levels, marked institutional-only, with no projection of returns",
        "wholesaler declines an RIA's request for a client-ready version and points to the separately approved retail piece",
    ],
}

_CHAT_TONES = ["casual / texting", "casual / texting", "terse / rushed", "friendly-professional", "salesy",
               "urgent (market event)", "apologetic (complaint response)"]
_CHAT_LENGTHS = [("one-liner (under 160 characters)", 0.30), ("short (2-4 short messages)", 0.45), ("medium (a thread of 5-8 messages)", 0.25)]


def _cell_fits(cat: str, cell: tuple[str, str]) -> bool:
    return cell[1] in T.VIOLATION_CATEGORIES[cat]["applies_to"]


def assign_targets(rng: random.Random) -> list[tuple[tuple[str, str], str | None]]:
    """(cell, target) jobs: violation slots filled most-constrained category first, then compliant slots."""
    slots = {c: n - round(n * COMPLIANT_FRAC) for c, n in CELLS.items()}
    assert sum(slots.values()) == sum(TARGETS.values()), (sum(slots.values()), sum(TARGETS.values()))
    jobs: list[tuple[tuple[str, str], str | None]] = []
    order = sorted(TARGETS, key=lambda c: (len(T.VIOLATION_CATEGORIES[c]["applies_to"]), c))
    for cat in order:
        for _ in range(TARGETS[cat]):
            open_cells = [c for c in slots if slots[c] > 0 and _cell_fits(cat, c)]
            cell = rng.choices(open_cells, weights=[slots[c] for c in open_cells], k=1)[0]
            slots[cell] -= 1
            jobs.append((cell, cat))
    assert not any(slots.values()), slots
    jobs += [(c, None) for c, n in CELLS.items() for _ in range(round(n * COMPLIANT_FRAC))]
    rng.shuffle(jobs)
    return jobs


def _shape(rng: random.Random, spec: dict, cell: tuple[str, str]) -> None:
    channel, ctype = cell
    spec["channel"] = channel
    spec["communication_type"] = ctype
    if ctype == "retail_communication":
        spec["recipient_type"] = "retail"
        spec["recipient_profile"] = _RETAIL_PROFILE[channel]
        spec["recipient_count_30d"] = rng.randint(26, 2000)
    else:
        profile, (lo, hi) = rng.choice(_PROFILE[ctype])
        spec["recipient_type"] = "institutional" if ctype == "institutional_communication" else "retail"
        spec["recipient_profile"] = profile
        spec["recipient_count_30d"] = rng.randint(lo, hi)
    if ctype != "retail_communication" and spec["sender_role"] == "firm marketing department":
        spec["sender_role"] = "registered representative"
    if ctype == "institutional_communication" and spec["sender_role"].startswith("client service"):
        spec["sender_role"] = "wholesaler / internal sales desk"
    spec["tone"] = rng.choice(_CHAT_TONES)
    spec["length"] = _weighted_dict(rng, dict(_CHAT_LENGTHS))
    formats = CHAT_FORMATS[channel]
    if spec["length"].startswith("one-liner"):
        formats = [f for f in formats if f.startswith("a single")]
    spec["chat_format"] = rng.choice(formats)


def _angle(rng: random.Random, pool: dict, ctype: str) -> str:
    return rng.choice(pool["all"] + pool[ctype] * 2)


def make_violation(rng: random.Random, idx: int, cell: tuple[str, str], cat: str) -> dict:
    spec = sample_in_domain(rng, idx)
    spec["scenario_id"] = f"ct-{idx:06d}"
    spec["compliant"] = False
    spec["near_miss_topic"] = None
    spec["difficulty"] = _weighted_dict(rng, T.DIFFICULTY_MIX)
    _shape(rng, spec, cell)
    weights = {k: w for k, w in T.VIOLATION_WEIGHTS.items() if k != cat and k not in _NO_EXTRA}
    one_liner = rng.random() < ONE_LINER_FRAC
    extra = _pick_categories(rng, "hard" if one_liner else spec["difficulty"], spec["communication_type"], weights, spec["channel"])
    if one_liner:
        spec["violation_categories"] = [cat] + extra[:max(1, len(extra) - 1)]
        spec["length"] = "one-liner (under 160 characters)"
        spec["chat_format"] = "a single message"
        spec["chat_angle"] = "a single short message that carries two or more distinct violations at once"
    else:
        spec["violation_categories"] = [cat] + extra[: len(extra) - 1]
        spec["chat_angle"] = _angle(rng, ANGLES, spec["communication_type"])
    if cat in _PRODUCT_FOR:
        spec["product"] = _PRODUCT_FOR[cat]
    else:
        _fit_product(rng, spec)
    return spec


def make_near_miss(rng: random.Random, idx: int, cell: tuple[str, str]) -> dict:
    spec = sample_in_domain(rng, idx)
    spec["scenario_id"] = f"ct-{idx:06d}"
    spec["compliant"] = True
    spec["violation_categories"] = []
    spec["difficulty"] = rng.choice(["easy", "medium", "hard", "hard"])
    _shape(rng, spec, cell)
    ctype = spec["communication_type"]
    if rng.random() < 0.6:
        spec["near_miss_topic"] = _angle(rng, NEAR_MISS_CHAT, ctype)
    else:
        spec["near_miss_topic"] = _near_miss(rng, ctype, spec["channel"])
        if spec["length"].startswith("one-liner"):  # disclosure-heavy topics do not fit in one line
            spec["length"] = "short (2-4 short messages)"
            spec["chat_format"] = rng.choice([f for f in CHAT_FORMATS[spec["channel"]] if not f.startswith("a single")] or CHAT_FORMATS[spec["channel"]])
    spec["chat_angle"] = f"compliant chat near-miss: {spec['near_miss_topic']}"
    if "logistics" in spec["near_miss_topic"]:
        spec["product"] = "account services / logistics (no product)"
    _fit_product(rng, spec)
    return spec


def make_chat_topup(seed: int) -> list[dict]:
    rng = random.Random(seed)
    jobs = assign_targets(rng)
    return [make_near_miss(rng, i, cell) if cat is None else make_violation(rng, i, cell, cat)
            for i, (cell, cat) in enumerate(jobs)]


def stats(specs: list[dict]) -> dict:
    cell = Counter(f"{s['channel']} / {s['communication_type']}" for s in specs)
    target = Counter(s["violation_categories"][0] for s in specs if not s["compliant"])
    any_cat = Counter(c for s in specs for c in s["violation_categories"])
    return {
        "n": len(specs),
        "compliant": sum(s["compliant"] for s in specs),
        "by_cell": dict(sorted(cell.items())),
        "by_target_category": dict(target.most_common()),
        "by_any_category": dict(any_cat.most_common()),
        "n_violations_per_spec": dict(sorted(Counter(len(s["violation_categories"]) for s in specs).items())),
        "one_liners_2plus": sum(1 for s in specs if s.get("chat_angle", "").startswith("a single short message")),
        "difficulty": dict(Counter(s["difficulty"] for s in specs)),
        "length": dict(Counter(s["length"] for s in specs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-chunk", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2212)
    ap.add_argument("--stats", action="store_true", help="print the plan tables and exit without writing chunks")
    args = ap.parse_args()

    specs = make_chat_topup(args.seed)
    if args.stats or not args.out:
        print(json.dumps(stats(specs), indent=1))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    for i in range(0, len(specs), args.per_chunk):
        (out / f"chunk_{i // args.per_chunk:04d}.json").write_text(json.dumps(specs[i : i + args.per_chunk], indent=1))
        n_chunks += 1
    print(f"{len(specs)} chat top-up scenarios -> {n_chunks} chunks in {out}")


if __name__ == "__main__":
    main()
