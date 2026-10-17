"""Prompt construction for the teacher model and for the student (Phi-4-mini).

Teacher side: a stable system prompt (rule digest + difficulty guide + output
contract) that is prompt-cached, plus a per-request user message carrying a
list of scenario specs to realize.

Student side: the exact instruction/response format the fine-tuned SLM sees.
"""

from __future__ import annotations

import json

from .taxonomy import DIFFICULTY_GUIDE, RULE_DIGEST, VIOLATION_CATEGORIES

_CATEGORY_TABLE = "\n".join(
    f"  {name:<36} cite {spec['rule']:<24} applies to: {', '.join(spec['applies_to'])}\n      {spec['desc']}"
    for name, spec in VIOLATION_CATEGORIES.items()
)

TEACHER_SYSTEM = f"""\
You are a senior FINRA Advertising Regulation examiner and a former broker-dealer Chief Compliance
Officer. You are producing a labeled training corpus for a small language model that will act as a
first-pass reviewer of broker-dealer correspondence under FINRA Rule 2210.

{RULE_DIGEST}

VIOLATION CATEGORIES YOU MAY USE (category -> rule cite you MUST use for it)
{_CATEGORY_TABLE}

{DIFFICULTY_GUIDE}

WRITING THE COMMUNICATIONS
- Write each communication as the registered person actually would: real greeting, body, sign-off,
  signature block with the (fictional) member firm name and the usual footer if that is normal for
  the channel. Texts and IMs are short and informal; letters have letterhead-style headers.
- Use fictional people and fictional member firms (e.g. "Harbor Point Securities LLC",
  "Northwind Financial Group, member FINRA/SIPC"). Fictional fund names are fine; well-known
  tickers (SPY, VTI, BND, AGG) and real indexes (S&P 500) are fine. Never name a real brokerage,
  real advisor, or real client.
- Vary names, cities, dollar amounts, dates (2023-2026), and voice. Do not reuse phrasing across
  samples in the same batch. Avoid template-like structure.
- Compliant samples must NOT be sanitized to the point of being obviously safe. Many of them must
  discuss performance, products, and recommendations correctly, with the disclosures 2210 actually
  requires, so the student cannot learn "mentions returns => violation".
- Violation samples must contain the violation in the text itself, quotable verbatim. Do not make
  the violation depend only on facts outside the message unless the metadata carries that fact
  (e.g. recipient_count_30d = 60 makes a template a retail communication).
- Keep the rest of the message realistic. Real violations sit inside otherwise normal emails.

LABELING RULES
- communication_type is determined by the metadata: retail recipients and recipient_count_30d <= 25
  => correspondence; retail and > 25 => retail_communication; only institutional investors =>
  institutional_communication (unless institutional_forwarding applies, in which case label the
  type the content effectively is and cite institutional_forwarding).
- compliant = true  <=> violations is empty and severity = "none".
- Each violation's excerpt must be copied verbatim from the communication (<= 200 chars). One entry
  per category; cite the rule the table assigns to that category.
- severity: low = technical/disclosure gap with little investor harm; medium = misleading content
  a reviewer would send back; high = promissory/guarantee/projection/fraud-adjacent or a violation
  likely to be referred to Enforcement.
- required_actions: 1-3 concrete reviewer actions. Compliant => ["No action required"] or a
  recordkeeping/filing reminder when one genuinely applies (e.g. fund retail communication must be
  filed within 10 business days even if compliant).
- rationale: 2-4 sentences in the voice of a compliance reviewer, citing paragraphs.
- Only cite categories that the communication_type supports (see applies-to column). In
  particular (d)(7) recommendation disclosure applies ONLY to retail communications; (d)(3),
  (d)(4), (d)(5), (d)(6) apply to correspondence and retail communications but NOT institutional.
- generator_notes: say what makes the sample sit at its difficulty and, for compliant samples, what
  tempting-but-wrong finding a weak model might raise.

OUTPUT
Return JSON matching the provided schema with exactly one sample per scenario, in the same order,
echoing each scenario_id. Do not add commentary outside the JSON.
"""


def build_user_message(scenarios: list[dict]) -> str:
    """One request realizes a list of scenario specs."""
    return (
        f"Generate {len(scenarios)} labeled samples, one per scenario below. Follow each scenario's "
        "difficulty, compliance target, violation categories (for violation samples these are the "
        "categories that MUST appear; you may add at most one more if it arises naturally), channel, "
        "product, sender, recipient profile, tone and length. For compliant scenarios the "
        "'near_miss_topic' tells you what risky-looking topic the message should handle correctly.\n\n"
        "SCENARIOS:\n" + json.dumps(scenarios, indent=2)
    )


# ---------------------------------------------------------------------------
# Student (Phi-4-mini-instruct) chat format
#
# Each training example is a realistic chat: a rep, supervisor or compliance
# analyst asks for a review in their own words, states the distribution context
# conversationally, and pastes the communication. The assistant answers in
# prose (verdict, findings with excerpts and rule cites, what to do) and closes
# with a compact ```json verdict block so the output stays machine-parseable.
# A slice of examples carries a follow-up turn answered from the label.
# ---------------------------------------------------------------------------

import hashlib
import random

STUDENT_SYSTEM = (
    "You are a FINRA Rule 2210 communications reviewer assisting a broker-dealer's compliance "
    "team. When a user shares a communication, determine whether it is correspondence, a retail "
    "communication or an institutional communication, identify any violations of Rule 2210 (and "
    "closely related FINRA rules) with the offending excerpt and rule paragraph, rate severity, and "
    "say what should happen next. Be direct and specific; do not invent facts not in the message. "
    "End every review with a ```json block containing the verdict: communication_type, compliant, "
    "violations [{category, rule, excerpt, explanation}], severity, required_actions, rationale."
)

_CTYPE_WORDS = {
    "correspondence": "correspondence",
    "retail_communication": "a retail communication",
    "institutional_communication": "an institutional communication",
}

_CHANNEL_WORDS = {
    "email": "email", "letter": "letter", "sms_text": "text message", "instant_message": "instant message",
    "mass_email_campaign": "mass email", "client_newsletter": "client newsletter", "form_letter": "form letter",
    "text_message_blast": "text blast", "social_media_post": "social media post", "website_page": "website page",
    "blog_article": "blog post", "seminar_invitation": "seminar invitation", "seminar_deck_or_script": "seminar script",
    "radio_or_podcast_script": "radio script", "print_ad": "print ad", "video_script": "video script",
}


def _rng(sample: dict) -> random.Random:
    return random.Random(int(hashlib.sha1(sample["id"].encode()).hexdigest()[:12], 16))


def _channel_word(sample: dict) -> str:
    return _CHANNEL_WORDS.get(sample["channel"], sample["channel"])


def _context_sentence(sample: dict, rng: random.Random) -> str:
    m = sample["metadata"]
    n = m["recipient_count_30d"]
    who = m["recipient_description"]
    sender = m["sender_role"]
    firm = m["firm"]
    ch = _channel_word(sample)
    if m["recipient_type"] == "institutional":
        opts = [
            f"This {ch} is from a {sender} at {firm} and goes only to {who} ({n} recipient{'s' if n != 1 else ''}, all institutional).",
            f"Context: {sender}, {firm}. Audience is {who} — institutional only, {n} in the last 30 days.",
            f"Sender is a {sender} with {firm}. Recipients: {who}. No retail investors on this one ({n} recipients).",
        ]
    elif n <= 25:
        opts = [
            f"A {sender} at {firm} wants to send this {ch} to {who} — that's {n} {'person' if n == 1 else 'people'} in the last 30 days, all retail.",
            f"From a {sender} ({firm}). Going to {who}; {n} retail recipient{'s' if n != 1 else ''} this month.",
            f"Context: {firm}, {sender}. Recipient: {who}. Count over the trailing 30 days is {n}.",
            f"This is {'an' if ch[0] in 'aeiou' else 'a'} {ch} from one of our {sender}s at {firm} to {who} ({n} retail recipient{'s' if n != 1 else ''}).",
        ]
    else:
        opts = [
            f"Marketing wants to push this {ch} from {firm} ({sender}) to {who} — roughly {n:,} retail recipients.",
            f"Submitted for pre-use review: {ch}, {firm}, prepared by {sender}. Audience: {who}, about {n:,} retail investors.",
            f"This {ch} will reach {who} (~{n:,} retail investors in 30 days). Sender: {sender} at {firm}.",
        ]
    s = rng.choice(opts)
    if m.get("subject") and rng.random() < 0.6:
        s += f' Subject line: "{m["subject"]}".'
    return s


_ASKS = [
    "Can you review this under Rule 2210 before it goes out?",
    "Is this compliant with FINRA 2210? Flag anything I need to fix.",
    "Compliance check please — what category is this and does it pass?",
    "Please do a 2210 review of the message below.",
    "Does this need to be changed before we send it?",
    "What problems, if any, does a FINRA reviewer see here?",
    "Run this through a 2210 review and tell me what you'd cite.",
    "Quick review — anything in here that violates the communications rules?",
    "Is this OK to send as-is? If not, tell me what's wrong and what rule it hits.",
    "I'm the supervising principal on this one. Give me your findings and the required action.",
    "Classify this communication and tell me whether it's approvable.",
    "Pre-use review request. Verdict, cites, and next steps please.",
    "Our rep drafted this. What would Advertising Regulation say?",
    "Look this over for 2210 issues and give me the verdict JSON at the end.",
    "Review the following and explain your reasoning like you would in a review log.",
]


def student_input(sample: dict, rng: random.Random | None = None) -> str:
    rng = rng or _rng(sample)
    ctx = _context_sentence(sample, rng)
    ask = rng.choice(_ASKS)
    body = sample["communication"]
    layouts = [
        f"{ask}\n\n{ctx}\n\n{body}",
        f"{ctx}\n\n{ask}\n\n---\n{body}\n---",
        f"{ask}\n\n{ctx}\n\n\"\"\"\n{body}\n\"\"\"",
        f"{ctx}\n\n{body}\n\n{ask}",
    ]
    return rng.choice(layouts)


def _verdict_json(label: dict) -> str:
    return "```json\n" + json.dumps(label, ensure_ascii=False, indent=1) + "\n```"


def render_answer(sample: dict, rng: random.Random | None = None) -> str:
    rng = rng or _rng(sample)
    label = sample["label"]
    m = sample["metadata"]
    n = m["recipient_count_30d"]
    ctype = label["communication_type"]
    lines = []

    # 1. classification sentence
    if ctype == "correspondence":
        lines.append(rng.choice([
            f"This is **correspondence** under 2210(a)(2): {n} retail recipient{'s' if n != 1 else ''} in the trailing 30 days is within the 25-investor threshold, so no principal pre-approval or FINRA filing is required, but it is subject to supervisory review under Rule 3110(b) and to the content standards in 2210(d).",
            f"Classification: correspondence ({n} retail recipient{'s' if n != 1 else ''} ≤ 25 in 30 days). That means 3110(b) post-use review rather than pre-approval, and no filing — but every content standard in 2210(d) still applies.",
        ]))
    elif ctype == "retail_communication":
        lines.append(rng.choice([
            f"This is a **retail communication** under 2210(a)(5): it is distributed or made available to more than 25 retail investors ({n:,} here), so it needs an appropriately qualified registered principal's approval before first use (2210(b)(1)) and must meet all of 2210(d), including the retail-only provisions.",
            f"Classification: retail communication — {n:,} retail investors is well over the 25-in-30-days threshold. Principal pre-approval under 2210(b)(1) is required, and filing with Advertising Regulation may apply depending on content.",
        ]))
    else:
        lines.append(rng.choice([
            "This is an **institutional communication** under 2210(a)(3): it goes only to institutional investors, so (b)(3) written procedures govern review and the retail-only provisions ((d)(2), (d)(3)–(d)(7)) do not apply. The general standards in 2210(d)(1) still do, and the firm may not treat it as institutional if it has reason to believe it will reach retail investors.",
            "Classification: institutional communication (institutional recipients only). Reviewed under the firm's (b)(3) procedures; 2210(d)(1) fair-and-balanced and no-projection standards apply in full even though the retail disclosure provisions do not.",
        ]))

    # 2. verdict + findings
    if label["compliant"]:
        lines.append("")
        lines.append(rng.choice(["**Verdict: compliant.** ", "**No violations found.** ", "**This passes.** "]) + label["rationale"])
    else:
        sev = label["severity"]
        lines.append("")
        lines.append(f"**Verdict: not compliant (severity {sev}).** I'd cite the following:")
        for i, v in enumerate(label["violations"], 1):
            lines.append(f"{i}. **{v['rule']}** — {v['category'].replace('_', ' ')}. Excerpt: \"{v['excerpt']}\" {v['explanation']}")
        lines.append("")
        lines.append(label["rationale"])

    # 3. actions
    acts = label["required_actions"]
    lines.append("")
    if len(acts) == 1:
        lines.append(f"**Next step:** {acts[0]}")
    else:
        lines.append("**Next steps:**")
        lines.extend(f"- {a}" for a in acts)

    lines.append("")
    lines.append(_verdict_json(label))
    return "\n".join(lines)


# Follow-up turns answerable from the label alone.
def _followup(sample: dict, rng: random.Random) -> tuple[str, str] | None:
    label = sample["label"]
    m = sample["metadata"]
    n = m["recipient_count_30d"]
    ctype = label["communication_type"]
    choices = []

    if not label["compliant"]:
        fix_lines = []
        for v in label["violations"]:
            fix_lines.append(f"- Remove or rewrite \"{v['excerpt']}\". {v['explanation']} That resolves the {v['rule']} finding.")
        choices.append((
            rng.choice(["What exactly do I need to change?", "How do I fix it?", "Give me the edits that would make this approvable."]),
            "Here is what has to change, finding by finding:\n" + "\n".join(fix_lines)
            + "\n\nAfter those edits, re-read the whole message for balance — fixing the flagged sentence is not enough if the overall tone still "
            "oversells the benefit without the risk. " + (f"Then: {'; '.join(label['required_actions'])}." if label["required_actions"] else ""),
        ))
        worst = label["violations"][0]
        choices.append((
            rng.choice(["Which rule is the most serious one here?", "If you had to pick one cite, which is it?"]),
            f"{worst['rule']} ({worst['category'].replace('_', ' ')}). The excerpt \"{worst['excerpt']}\" is the finding a FINRA examiner would lead with: {worst['explanation']} Severity on this communication is {label['severity']}.",
        ))

    if ctype == "correspondence":
        choices.append((
            rng.choice(["Why is this correspondence and not a retail communication?", "Does this need principal approval first?", "Do we have to file this with FINRA?"]),
            f"It is correspondence: it is distributed to {n} retail investor{'s' if n != 1 else ''} within a 30-calendar-day period, which is 25 or fewer, so it meets the 2210(a)(2) definition. "
            "Correspondence does not require registered-principal approval before use and is excluded from the filing requirements (2210(c)(7)(J)); it is supervised under Rule 3110(b) and 3110.06–.09, typically through risk-based post-use review, and must be retained under 4511 / SEA 17a-4. "
            "Watch the count: if the same template reaches more than 25 retail investors in any 30-day window it becomes a retail communication and pre-approval applies.",
        ))
    elif ctype == "retail_communication":
        choices.append((
            rng.choice(["Does this need to be filed with FINRA?", "What's the approval and filing path for this?", "Who has to sign off on this?"]),
            f"It is a retail communication ({n:,} retail investors), so an appropriately qualified registered principal must approve it before the earlier of first use or filing (2210(b)(1)(A)). "
            + ("Based on the required actions: " + "; ".join(label["required_actions"]) + ". " if label["required_actions"] else "")
            + "Filing is content-driven: retail communications that promote or recommend a registered investment company, a public DPP, a CMO or a registered structured product must be filed with Advertising Regulation within 10 business days of first use (2210(c)(3)); fund rankings not generally published need pre-use filing (2210(c)(2)); a member in its first year must pre-file (2210(c)(1)). Keep the approval record with the approving principal's name and date under 2210(b)(4).",
        ))
    else:
        choices.append((
            rng.choice(["Can the recipient forward this to their own clients?", "What if the pension fund shares this with plan participants?"]),
            "Not without changing its status. Under 2210(a)(4) the firm may not treat a communication as institutional if it has reason to believe it will be forwarded or made available to any retail investor; plan participants are retail investors, not institutional ones. "
            "If forwarding is likely, the piece has to be reviewed as a retail communication — principal pre-approval, the retail-only content standards, and filing where applicable — and the firm's (b)(3) procedures should include a legend or instruction against onward distribution.",
        ))

    if label["compliant"]:
        choices.append((
            rng.choice(["Anything you'd still tighten even though it passes?", "Is there anything borderline here?"]),
            "It passes, so these are optional. " + sample.get("generator_notes", "").strip()
            + (" " if sample.get("generator_notes") else "")
            + "The message keeps the required disclosures in the body rather than a footnote, which is what 2210(d)(1)(C) wants; keep that placement if the text is edited.",
        ))

    return rng.choice(choices) if choices else None


def to_chat_messages(sample: dict, followup_rate: float = 0.30) -> list[dict]:
    rng = _rng(sample)
    msgs = [
        {"role": "system", "content": STUDENT_SYSTEM},
        {"role": "user", "content": student_input(sample, rng)},
        {"role": "assistant", "content": render_answer(sample, rng)},
    ]
    if rng.random() < followup_rate:
        fu = _followup(sample, rng)
        if fu:
            msgs.append({"role": "user", "content": fu[0]})
            msgs.append({"role": "assistant", "content": fu[1]})
    return msgs
