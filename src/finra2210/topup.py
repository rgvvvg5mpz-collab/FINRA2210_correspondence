"""Targeted top-up scenarios for categories that are too sparse in the round-1 train set.

  python -m finra2210.topup --out data/chunks_topup/in_domain --per-chunk 24

Each scenario is an ordinary in-domain spec (scenarios.sample_in_domain) whose communication
type, recipients, channel and violation_categories are then forced so the target category is the
primary violation. A minority are compliant near-misses on the same topic. Every spec also carries
an `angle` so agents spread over the different ways the category shows up in practice.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import taxonomy as T
from .scenarios import _fit_product, _pick_categories, _weighted, _weighted_dict, sample_in_domain

_PUBLIC = "the general public / all retail visitors (website, social, seminar, ad)"
_BOOK = "the rep's or branch's full retail client list (mass distribution over the threshold)"
_INST = "a single institutional investor (pension CIO, bank treasury, insurance GA)"
_RIA = "an RIA who serves retail clients"

# (n violation scenarios, n compliant near-misses) per target category
PLAN = {
    "brokercheck_link_missing": (104, 16),
    "institutional_forwarding": (92, 16),
    "unapproved_retail_communication": (92, 16),
}

ANGLES = {
    "brokercheck_link_missing": [
        "a registered rep's profile / bio page with no BrokerCheck reference at all",
        "the firm's homepage or landing page for retail visitors that never mentions BrokerCheck",
        "a team page listing several advisors, none with a BrokerCheck reference or link",
        "a service or product landing page (retirement planning, 529s, rollovers) aimed at retail visitors, no BrokerCheck",
        "BrokerCheck is named in the text but there is no hyperlink to it",
        "a BrokerCheck link exists only deep in a long footer of legal text, so it is not readily apparent",
        "the page links to the SEC's IAPD site or the firm's own 'disclosures' page instead of FINRA BrokerCheck",
        "a branch office page (address, hours, advisors, services) with no BrokerCheck reference",
    ],
    "institutional_forwarding": [
        "explicit invitation to share the material with the recipient's retail clients",
        "a client-friendly one-pager written for end investors, sent to an adviser who serves retail households",
        "suggests the recipient white-label the piece (swap in their own logo) for their client mailings",
        "the recipient said earlier that they pass everything along to their clients, and the sender knows it",
        "an 'institutional use only' legend that the body of the message contradicts by inviting wider sharing",
        "a bank treasury or trust contact is encouraged to put the piece in front of the bank's retail customers",
        "an insurance GA is asked to pass the piece to its agents to use with their retail customers",
        "subtle: offers to send 'a few hundred copies' or a clean PDF 'for your client meetings'",
    ],
    "unapproved_retail_communication": [
        "the text or metadata shows it already went out (e.g. 'sent to the whole book this morning') with no principal approval",
        "the rep sent the same template one-at-a-time to more than 25 clients this month, treating it as correspondence",
        "used before compliance finished its review ('compliance is still looking at it, but we went ahead')",
        "the rep materially edited a previously approved piece (new numbers, new claims) and used the edited version",
        "'approved' only by someone who is not a registered principal (sales assistant, marketing coordinator, the rep)",
        "the rep wrote and distributed a newsletter or blast without ever submitting it for review",
    ],
}

# angles that only make sense on some channels
_UNAPPROVED_CHANNEL_ANGLES = {
    "social_media_post": "a static post on the rep's business social account, published without principal approval (not an interactive forum reply)",
    "seminar_deck_or_script": "a seminar deck or script presented to the public before a principal signed off",
    "seminar_invitation": "a seminar invitation mailed or posted before a principal approved it",
    "website_page": "a page the rep added to the firm-hosted or personal business website without principal approval",
    "radio_or_podcast_script": "a script read on air / on a podcast without prior principal approval",
    "video_script": "a video posted publicly without prior principal approval",
    "print_ad": "a print ad that ran before a principal approved it",
    "blog_article": "a blog article the rep published without principal approval",
}

NEAR_MISS = {
    "brokercheck_link_missing": [
        "rep profile page with a readily apparent BrokerCheck reference and hyperlink next to the rep's name, plus FINRA/SIPC membership",
        "firm homepage with the BrokerCheck reference and hyperlink in the header area, describing services without performance claims",
        "website page with the BrokerCheck hyperlink, FINRA/SIPC membership and the required relationship disclosure for an affiliated RIA",
        "retail product page that discusses a product with balanced risks and carries a prominent BrokerCheck reference and hyperlink",
    ],
    "institutional_forwarding": [
        "institutional message to an adviser marked 'for institutional use only — not for distribution to retail investors', with no invitation to share and balanced risks",
        "the recipient asks to forward the piece to clients; the sender declines and offers a separately approved retail version instead",
        "wholesaler note to an RIA that keeps itself institutional and points to separate, principal-approved client-facing materials",
        "institutional communication to a pension CIO that discusses expected spread/rate scenarios without projecting the performance of any investment",
        "bank treasury email about a bond offering, for the treasurer's own investment decision, with an institutional-only legend",
    ],
    "unapproved_retail_communication": [
        "mass communication that shows a registered principal approved it before first use (approval code / date / approver in the footer)",
        "retail communication that makes no investment recommendation and promotes no product or service (e.g. office move, holiday hours, tax-form timing), so (b)(1) pre-approval is not required",
        "rep explains to clients that the attached piece was approved by the firm's principal and is being used unchanged",
    ],
}
_FORUM_NEAR_MISS = "reply in an online interactive forum (live comment thread), which is exempt from (b)(1) pre-approval but supervised, with no performance claims"
_NO_PROMO_CHANNELS = {"mass_email_campaign", "text_message_blast", "form_letter", "client_newsletter", "website_page", "social_media_post"}


def _no_texting(spec: dict) -> None:
    if spec["channel"] in ("letter", "website_page", "print_ad", "form_letter") and spec["tone"] == "casual / texting":
        spec["tone"] = "friendly-professional"


def _set_retail(rng: random.Random, spec: dict, profile: str) -> None:
    spec["communication_type"] = "retail_communication"
    spec["recipient_type"] = "retail"
    spec["recipient_profile"] = profile
    if profile == _PUBLIC:
        spec["recipient_count_30d"] = rng.randint(200, 50000)
        spec["channel"] = _weighted(rng, T.RETAIL_PUBLIC_CHANNELS)
    else:
        spec["recipient_count_30d"] = rng.randint(26, 2000)
        spec["channel"] = _weighted(rng, T.RETAIL_BOOK_CHANNELS)


def _set_institutional(rng: random.Random, spec: dict) -> None:
    profile = rng.choice([_INST, _RIA])
    spec["communication_type"] = "institutional_communication"
    spec["recipient_type"] = "institutional"
    spec["recipient_profile"] = profile
    spec["recipient_count_30d"] = rng.randint(1, 3) if profile == _INST else rng.randint(1, 2)
    spec["channel"] = _weighted(rng, T.IN_DOMAIN_CHANNELS)
    if spec["sender_role"] == "firm marketing department":
        spec["sender_role"] = rng.choice(["registered representative", "wholesaler / internal sales desk"])


def _shape(rng: random.Random, spec: dict, cat: str) -> None:
    if cat == "brokercheck_link_missing":
        _set_retail(rng, spec, _PUBLIC)
        spec["channel"] = "website_page"
    elif cat == "institutional_forwarding":
        _set_institutional(rng, spec)
    else:
        _set_retail(rng, spec, rng.choice([_PUBLIC, _BOOK]))
    _no_texting(spec)


def make_violation(rng: random.Random, idx: int, cat: str) -> dict:
    spec = sample_in_domain(rng, idx)
    spec["scenario_id"] = f"tu-{idx:06d}"
    spec["compliant"] = False
    spec["near_miss_topic"] = None
    spec["difficulty"] = _weighted_dict(rng, T.DIFFICULTY_MIX)
    _shape(rng, spec, cat)
    weights = {k: w for k, w in T.VIOLATION_WEIGHTS.items() if k != cat}
    weights.pop("unapproved_retail_communication", None)  # never a natural extra on these specs
    extra = _pick_categories(rng, spec["difficulty"], spec["communication_type"], weights, spec["channel"])
    spec["violation_categories"] = [cat] + extra[: len(extra) - 1]
    angles = list(ANGLES[cat])
    if cat == "unapproved_retail_communication" and spec["channel"] in _UNAPPROVED_CHANNEL_ANGLES:
        angles = [_UNAPPROVED_CHANNEL_ANGLES[spec["channel"]]] * 2 + angles[:1] + angles[2:]
    if cat == "institutional_forwarding" and spec["recipient_profile"] == _RIA:
        angles = [a for a in angles if "bank" not in a and "insurance GA" not in a]
    if cat == "institutional_forwarding" and spec["recipient_profile"] == _INST:
        angles = [a for a in angles if "adviser who serves" not in a]
    spec["angle"] = rng.choice(angles)
    _fit_product(rng, spec)
    return spec


def make_near_miss(rng: random.Random, idx: int, cat: str) -> dict:
    spec = sample_in_domain(rng, idx)
    spec["scenario_id"] = f"tu-{idx:06d}"
    spec["compliant"] = True
    spec["violation_categories"] = []
    spec["difficulty"] = rng.choice(["medium", "hard"])
    _shape(rng, spec, cat)
    topics = list(NEAR_MISS[cat])
    if cat == "unapproved_retail_communication":
        if spec["channel"] == "social_media_post":
            topics.append(_FORUM_NEAR_MISS)
        if spec["channel"] not in _NO_PROMO_CHANNELS:
            topics = [t for t in topics if not t.startswith("retail communication that makes no")]
    if cat == "institutional_forwarding" and spec["recipient_profile"] == _RIA:
        topics = [t for t in topics if "pension CIO" not in t and "bank treasury" not in t]
    if cat == "institutional_forwarding" and spec["recipient_profile"] == _INST:
        topics = [t for t in topics if "RIA" not in t and "adviser" not in t]
    spec["near_miss_topic"] = rng.choice(topics)
    spec["angle"] = f"compliant near-miss for {cat}"
    if spec["near_miss_topic"].startswith("retail communication that makes no"):
        spec["product"] = "account services / logistics (no product)"
    _fit_product(rng, spec)
    return spec


def make_topup(seed: int) -> list[dict]:
    rng = random.Random(seed)
    jobs = [(cat, False) for cat, (nv, _) in PLAN.items() for _ in range(nv)]
    jobs += [(cat, True) for cat, (_, nc) in PLAN.items() for _ in range(nc)]
    rng.shuffle(jobs)
    return [(make_near_miss if comp else make_violation)(rng, i, cat) for i, (cat, comp) in enumerate(jobs)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-chunk", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2211)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = make_topup(args.seed)
    n_chunks = 0
    for i in range(0, len(scenarios), args.per_chunk):
        (out / f"chunk_{i // args.per_chunk:04d}.json").write_text(json.dumps(scenarios[i : i + args.per_chunk], indent=1))
        n_chunks += 1
    print(f"{len(scenarios)} top-up scenarios -> {n_chunks} chunks in {out}")


if __name__ == "__main__":
    main()
