"""Scenario sampling: in-domain and OOD specs that the teacher must realize.

Diversity comes from here, not from the teacher's imagination. Every sample
is pinned to difficulty, compliance target, violation categories, channel,
product, sender, recipient profile, tone and length before Fable sees it.
"""

from __future__ import annotations

import random

from . import taxonomy as T


def _weighted(rng: random.Random, pairs) -> str:
    items, weights = zip(*pairs)
    return rng.choices(items, weights=weights, k=1)[0]


def _weighted_dict(rng: random.Random, d: dict[str, float]) -> str:
    items = [k for k, w in d.items() if w > 0]
    weights = [d[k] for k in items]
    return rng.choices(items, weights=weights, k=1)[0]


def _pick_categories(rng: random.Random, difficulty: str, ctype: str, weights: dict[str, float], channel: str = "") -> list[str]:
    allowed = {k: w for k, w in weights.items() if w > 0 and ctype in T.VIOLATION_CATEGORIES[k]["applies_to"]}
    if channel != "website_page":
        allowed.pop("brokercheck_link_missing", None)
    n = {"easy": 1, "medium": rng.choice([1, 1, 2]), "hard": rng.choice([1, 2, 2, 3])}[difficulty]
    cats: list[str] = []
    while len(cats) < n and len(cats) < len(allowed):
        c = _weighted_dict(rng, allowed)
        if c not in cats:
            cats.append(c)
    return cats


_NEAR_MISS_TOPICS = [
    "quotes historical fund returns with full standardized performance, max sales charge and expense ratio",
    "gives a strong opinion about a sector with a reasonable basis and balancing risks",
    "explains a hypothetical compounding illustration, clearly labeled hypothetical, not tied to a product",
    "compares two products and discloses the material differences (cost, liquidity, guarantees, tax)",
    "mentions muni income being federally tax-exempt and correctly notes state tax / AMT may apply",
    "relays a client testimonial with all three 2210(d)(6) disclosures",
    "recommends a stock in correspondence (no (d)(7) obligation for correspondence) with balanced risks",
    "discusses a past market move without implying it will recur",
    "uses words like 'safe' or 'guaranteed' only where literally true (FDIC-insured brokered CD, fixed annuity guarantee by insurer, with the issuer-backing caveat)",
    "states the firm is a FINRA/SIPC member correctly without implying endorsement",
    "summarizes a research report price target with the report's basis and risk language",
    "responds to a complaint factually without admitting or promising anything improper",
    "routine logistics: forms, meeting times, account transfers, beneficiary updates, tax documents",
    "institutional communication to a pension CIO that discusses expected spread/rate scenarios without projecting the performance of any investment",
    "declines to predict the market when the client asks for a forecast",
    "describes product risks at length (surrender charges, liquidity, leverage) in a plain-English way",
    "compares two products in correspondence — (d)(2) does not apply, but (d)(1)(A) balance is met by stating the key differences anyway",
    "retail stock recommendation that carries every (d)(7) disclosure (market-making, financial interest, underwriting, price, offer of supporting info)",
    "retail fund promotion with standardized performance, sales charge, expense ratio and a note that it will be filed with FINRA",
    "social media post that names the firm, links to finra.org / BrokerCheck, and makes no claims about performance",
    "seminar invitation that describes the topic and the free dinner without promising outcomes or using 'FINRA approved'",
    "website page with the BrokerCheck hyperlink, FINRA/SIPC membership and the required relationship disclosure for an affiliated RIA",
    "uses a fund ranking with the full Rule 2212 disclosures",
]

_RETAIL_ONLY_TOPICS = {
    "retail stock recommendation that carries every (d)(7) disclosure (market-making, financial interest, underwriting, price, offer of supporting info)",
    "retail fund promotion with standardized performance, sales charge, expense ratio and a note that it will be filed with FINRA",
    "social media post that names the firm, links to finra.org / BrokerCheck, and makes no claims about performance",
    "seminar invitation that describes the topic and the free dinner without promising outcomes or using 'FINRA approved'",
    "website page with the BrokerCheck hyperlink, FINRA/SIPC membership and the required relationship disclosure for an affiliated RIA",
    "uses a fund ranking with the full Rule 2212 disclosures",
}
_CORRESPONDENCE_ONLY_TOPICS = {
    "recommends a stock in correspondence (no (d)(7) obligation for correspondence) with balanced risks",
    "compares two products in correspondence — (d)(2) does not apply, but (d)(1)(A) balance is met by stating the key differences anyway",
}


_INSTITUTIONAL_ONLY_TOPICS = {
    "institutional communication to a pension CIO that discusses expected spread/rate scenarios without projecting the performance of any investment",
}


# near-miss topics that only make sense on one channel
_CHANNEL_TOPICS = {
    "social media post that names the firm, links to finra.org / BrokerCheck, and makes no claims about performance": "social_media_post",
    "seminar invitation that describes the topic and the free dinner without promising outcomes or using 'FINRA approved'": "seminar_invitation",
    "website page with the BrokerCheck hyperlink, FINRA/SIPC membership and the required relationship disclosure for an affiliated RIA": "website_page",
}


def _near_miss(rng: random.Random, ctype: str, channel: str | None = None) -> str:
    pool = [t for t in _NEAR_MISS_TOPICS
            if not (ctype != "retail_communication" and t in _RETAIL_ONLY_TOPICS)
            and not (ctype != "correspondence" and t in _CORRESPONDENCE_ONLY_TOPICS)
            and not (ctype != "institutional_communication" and t in _INSTITUTIONAL_ONLY_TOPICS)
            and not (channel is not None and t in _CHANNEL_TOPICS and _CHANNEL_TOPICS[t] != channel)]
    return rng.choice(pool)


def _channel_for(rng: random.Random, ctype: str, rec_desc: str) -> str:
    if ctype != "retail_communication":
        return _weighted(rng, T.IN_DOMAIN_CHANNELS)
    if rec_desc.startswith("the general public"):
        return _weighted(rng, T.RETAIL_PUBLIC_CHANNELS)
    return _weighted(rng, T.RETAIL_BOOK_CHANNELS)


def sample_in_domain(rng: random.Random, idx: int) -> dict:
    difficulty = _weighted_dict(rng, T.DIFFICULTY_MIX)
    compliant = rng.random() < T.COMPLIANT_RATE
    rec_i = rng.choices(range(len(T.IN_DOMAIN_RECIPIENT_SCENARIOS)), weights=T.IN_DOMAIN_RECIPIENT_WEIGHTS, k=1)[0]
    rec_desc, (lo, hi), rtype, ctype = T.IN_DOMAIN_RECIPIENT_SCENARIOS[rec_i]
    channel = _channel_for(rng, ctype, rec_desc)
    tone = rng.choice(T.TONES)
    if channel in ("letter", "website_page", "print_ad", "form_letter") and tone == "casual / texting":
        tone = "friendly-professional"
    sender = _weighted(rng, T.IN_DOMAIN_SENDERS)
    if sender == "firm marketing department" and ctype != "retail_communication":
        sender = "registered representative"
    spec = {
        "scenario_id": f"id-{idx:06d}",
        "distribution": "in_domain",
        "difficulty": difficulty,
        "compliant": compliant,
        "communication_type": ctype,
        "violation_categories": [] if compliant else _pick_categories(rng, difficulty, ctype, T.VIOLATION_WEIGHTS, channel),
        "near_miss_topic": _near_miss(rng, ctype, channel) if compliant and difficulty != "easy" else None,
        "channel": channel,
        "product": _weighted(rng, T.IN_DOMAIN_PRODUCTS),
        "sender_role": sender,
        "recipient_profile": rec_desc,
        "recipient_count_30d": rng.randint(lo, hi),
        "recipient_type": rtype,
        "tone": tone,
        "length": _weighted(rng, T.LENGTHS),
    }
    if compliant and difficulty == "easy":
        spec["near_miss_topic"] = "none — routine content with no claims"
    _fit_product(rng, spec)
    return spec


# (d)(7)(A) excludes funds and variable products; (d)(5) is only about open-end fund performance.
_SECURITY_PRODUCTS = ["individual stocks", "bonds / CDs / fixed income", "municipal bonds"]
_FUND_PRODUCTS = ["mutual funds", "ETFs", "target-date / asset allocation funds"]


def _fit_product(rng: random.Random, spec: dict) -> None:
    cats = spec["violation_categories"]
    if "recommendation_disclosure" in cats and spec["product"] not in _SECURITY_PRODUCTS:
        spec["product"] = rng.choice(_SECURITY_PRODUCTS)
    elif ("fund_performance_disclosure" in cats or "fund_ranking_disclosure" in cats) and spec["product"] not in _FUND_PRODUCTS:
        spec["product"] = rng.choice(_FUND_PRODUCTS[:2])
    topic = spec.get("near_miss_topic") or ""
    if ("fund performance" in topic or "fund promotion" in topic or "fund ranking" in topic) and spec["product"] not in _FUND_PRODUCTS:
        spec["product"] = rng.choice(_FUND_PRODUCTS[:2])
    elif "stock recommendation" in topic or "recommends a stock" in topic or "price target" in topic:
        spec["product"] = "individual stocks"
    elif "muni income" in topic:
        spec["product"] = "municipal bonds"


_AI_TWIST_COMPLIANT = "AI-assisted draft reviewed and approved by a principal before use — RN 24-09"


def sample_ood(rng: random.Random, idx: int) -> dict:
    base = sample_in_domain(rng, idx)
    base["scenario_id"] = f"ood-{idx:06d}"
    base["distribution"] = "ood"
    n_axes = rng.choice([1, 2, 2, 3])
    axes = rng.choices(list(T.OOD_AXIS_WEIGHTS), weights=list(T.OOD_AXIS_WEIGHTS.values()), k=n_axes)
    shifts: dict[str, str] = {}
    for ax in dict.fromkeys(axes):  # dedupe, keep order
        shifts[ax] = rng.choice(T.OOD_AXES[ax])
    base["ood_shifts"] = shifts
    if "channel" in shifts:
        base["channel"] = shifts["channel"]
    if "product" in shifts:
        base["product"] = shifts["product"]
        # OOD products unlock product-specific categories
        weights = dict(T.VIOLATION_WEIGHTS)
        weights["product_specific_omission"] = 0.15
        weights["options_communication"] = 0.12 if "options" in shifts["product"] else 0.0
        weights["third_party_content_adoption"] = 0.05
        if not base["compliant"]:
            base["violation_categories"] = _pick_categories(rng, base["difficulty"], base["communication_type"], weights, base["channel"])
    if "audience" in shifts:
        aud = shifts["audience"]
        if "institutional investor who will forward" in aud or "plan sponsor" in aud:
            base["recipient_type"] = "institutional"
            base["recipient_count_30d"] = rng.randint(1, 3)
            base["communication_type"] = "institutional_communication"
            if not base["compliant"]:
                base["violation_categories"] = ["institutional_forwarding"] + [
                    c for c in base["violation_categories"] if c != "institutional_forwarding"
                ][:1]
        elif "family office with $40M" in aud:
            base["recipient_type"] = "retail"
            base["recipient_count_30d"] = 1
            base["communication_type"] = "correspondence"
        elif "registered person at another member" in aud:
            base["recipient_type"] = "institutional"
            base["recipient_count_30d"] = 1
            base["communication_type"] = "institutional_communication"
        base["recipient_profile"] = aud
    if "style" in shifts:
        base["tone"] = shifts["style"]
    if "firm_type" in shifts:
        base["firm_type"] = shifts["firm_type"]
    if "regulatory_twist" in shifts:
        # a compliant message cannot have been sent unreviewed; show the supervised version instead
        if base["compliant"] and "without review" in shifts["regulatory_twist"]:
            shifts["regulatory_twist"] = _AI_TWIST_COMPLIANT
        base["regulatory_twist"] = shifts["regulatory_twist"]
    if "channel" in shifts and "thread" in shifts["channel"]:
        base["length"] = "long (3-6 paragraphs)"
    # a retail-facing channel cannot carry an institutional-only communication
    if base["communication_type"] == "institutional_communication" and _retail_facing(base["channel"]):
        base["communication_type"] = "retail_communication"
        base["recipient_type"] = "retail"
        base["recipient_count_30d"] = rng.randint(200, 50000)
        base["recipient_profile"] = "the general public / all retail visitors"
    # audience shifts may have changed the communication type after categories were picked;
    # drop categories that cannot apply to the final type and re-pick if none remain
    if not base["compliant"]:
        ok = [c for c in base["violation_categories"]
              if base["communication_type"] in T.VIOLATION_CATEGORIES[c]["applies_to"]]
        base["violation_categories"] = ok or _pick_categories(
            rng, base["difficulty"], base["communication_type"], T.VIOLATION_WEIGHTS, base["channel"])
    # audience/product shifts may have changed the communication type after the near-miss
    # topic was chosen; re-pick so the topic is always valid for the final type
    if base["compliant"] and base["near_miss_topic"] and base["difficulty"] != "easy":
        base["near_miss_topic"] = _near_miss(rng, base["communication_type"], base["channel"])
    return base


_RETAIL_FACING = ("retail user", "Reddit/forum", "website_page", "social_media_post", "blog_article",
                  "print_ad", "radio_or_podcast_script", "video_script", "seminar_invitation")


def _retail_facing(channel: str) -> bool:
    return any(k in channel for k in _RETAIL_FACING)


def make_scenarios(n: int, seed: int, ood: bool) -> list[dict]:
    rng = random.Random(seed)
    fn = sample_ood if ood else sample_in_domain
    return [fn(rng, i) for i in range(n)]
