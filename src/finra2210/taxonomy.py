"""FINRA Rule 2210 taxonomy used to drive synthetic data generation.

Everything the generator and validator need to agree on lives here:
the rule digest shown to the teacher model, the violation categories and
the rule cites they map to, difficulty definitions, and the scenario axes
(in-domain and out-of-distribution) that are sampled to force diversity.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Rule digest (shown verbatim to the teacher model as grounding)
# ---------------------------------------------------------------------------

RULE_DIGEST = """\
FINRA RULE 2210 — COMMUNICATIONS WITH THE PUBLIC (working digest)

(a) DEFINITIONS
  (a)(2) Correspondence: any written (including electronic) communication distributed or made
         available to 25 OR FEWER retail investors within any 30-calendar-day period.
  (a)(3) Institutional communication: distributed or made available ONLY to institutional
         investors (excludes a member's internal communications). A member may NOT treat a
         communication as institutional if it has reason to believe it will be forwarded or
         made available to any retail investor (a)(4) / (b)(3).
  (a)(4) Institutional investor: bank, savings and loan, insurance company, registered
         investment company, SEC- or state-registered investment adviser; any other entity or
         natural person with total assets of at least $50 million; governmental entity;
         employee benefit / qualified plan with at least 100 participants (NOT the individual
         participants); a FINRA member or registered person; a person acting solely on behalf
         of an institutional investor.
  (a)(5) Retail communication: distributed or made available to MORE THAN 25 retail investors
         within any 30-calendar-day period. Includes websites, social media posts, seminars
         materials, newsletters, scripted presentations, form letters, mass emails.
  (a)(6) Retail investor: any person other than an institutional investor, whether or not
         they have an account with the member.
  Counting note: the 25/30-day threshold is per communication across recipients. The same
  template sent one-at-a-time to 30 people in a month is a retail communication.

(b) APPROVAL, REVIEW AND RECORDKEEPING
  (b)(1) Retail communications: an appropriately qualified registered principal must approve
         BEFORE the earlier of use or filing. Exceptions include: content already filed by
         another member with a FINRA review letter; posts on an online interactive electronic
         forum (still supervised under Rule 3110); communications that make no financial or
         investment recommendation and do not promote a product or service (subject to
         3110 supervision).
  (b)(2) Correspondence: NO pre-use principal approval required; subject to supervision and
         review under Rule 3110(b) and 3110.06–.09 (risk-based review is permitted).
  (b)(3) Institutional communications: written procedures for principal review; must be
         designed to ensure the communication is not forwarded to retail investors.
  (b)(4) Recordkeeping: retain per SEA Rule 17a-4(b)(4) — copy, dates of first and last use,
         name of approving principal (or preparer if no approval required), and sources of
         statistics/charts; see also FINRA Rule 4511. Applies to ALL three categories.

(c) FILING WITH FINRA
  Retail communications concerning registered investment companies (mutual funds, ETFs,
  closed-end funds, UITs, variable products), public direct participation programs, CMOs,
  security futures and certain other products must be filed with FINRA Advertising Regulation
  within 10 business days of first use (some require pre-use filing: e.g. fund rankings not
  generally published, bond fund volatility ratings). Correspondence and institutional
  communications are EXCLUDED from filing. New members must pre-file retail communications
  for their first year.

(d) CONTENT STANDARDS
  (d)(1) General standards — apply to ALL communications (correspondence, retail, institutional):
    (A) Must be fair and balanced and provide a sound basis for evaluating the facts about
        any security, industry or service. No material fact or qualification may be omitted
        if the omission would cause the communication to be misleading. Benefits must be
        accompanied by the relevant risks.
    (B) No false, exaggerated, unwarranted, promissory or misleading statement or claim.
        A member may not publish or circulate a communication it knows or has reason to know
        contains an untrue statement of material fact or is otherwise false or misleading.
        "Guaranteed", "can't lose", "risk-free", "safe" for securities with market risk, and
        unsupported superlatives ("best", "#1") fall here.
    (C) Information may be placed in a legend or footnote only if that placement would not
        inhibit an investor's understanding.
    (D) Members must ensure that statements are clear and not misleading within the context in
        which they are made, and that they provide balanced treatment of risks and potential
        benefits. Communications must be consistent with the risks of fluctuating prices and
        the uncertainty of dividends, rates of return and yield inherent to investments.
    (E) Members must consider the nature of the audience and provide details and explanations
        appropriate to that audience.
    (F) Communications may NOT predict or project performance, imply that past performance
        will recur, or make any exaggerated or unwarranted claim, opinion or forecast.
        Permitted exceptions: (i) a hypothetical illustration of mathematical principles that
        does not predict the performance of an investment or strategy (e.g. the effect of
        compounding at an assumed rate, clearly labeled hypothetical); (ii) an investment
        analysis tool meeting Rule 2214; (iii) a price target in a research report that has a
        reasonable basis, discloses valuation methods and risks (Rule 2241). NOTE: FINRA has
        proposed (SR-FINRA-2023-016) an exception for projected performance in institutional
        communications and communications to qualified purchasers; it is NOT in the current
        rule text. Treat any projection, including one sent to an institutional investor, as a
        (d)(1)(F) violation unless it fits exceptions (i)–(iii).
  (d)(2) Comparisons: any comparison in RETAIL COMMUNICATIONS between investments or services
        must disclose all material differences, including (as applicable) investment
        objectives, costs and expenses, liquidity, safety, guarantees or insurance, fluctuation
        of principal or return, and tax features. (d)(2) does NOT apply to correspondence or
        institutional communications; a misleading comparison there is cited under (d)(1)(A)
        (omitted material qualification) or (d)(1)(B).
  (d)(3) Disclosure of member's name: retail communications and CORRESPONDENCE must
        prominently disclose the name of the member firm (the FINRA member, or the name
        under which it does business). Nonmember affiliate names may appear if the
        relationship is reflected.
  (d)(4) Tax considerations: in retail communications and correspondence, any reference to
        "tax-free" or "tax-exempt" income must indicate which income taxes apply or do not
        (e.g. muni income may be subject to state tax or AMT) unless the income is free from
        all applicable taxes. (d)(4)(B): income or returns may not be characterized as
        tax-free or exempt when tax is merely deferred (e.g. payable on withdrawal).
        (d)(4)(C): comparative illustrations of tax-deferred vs taxable compounding must use
        identical amounts and assumed gross rates (max 10%), actual federal tax rates, and
        disclose that the assumed rate is not guaranteed, the effect of losses, and that
        ordinary income rates apply to withdrawals.
  (d)(5) Fees, expenses and standardized performance: retail communications and
        CORRESPONDENCE that present performance of a non-money-market open-end fund must
        disclose standardized performance per SEC Rule 482 / 34b-1 (1-, 5-, 10-year or
        since-inception average annual total returns as of the most recent calendar
        quarter-end), the maximum sales charge, and the total annual operating expense
        ratio (gross of waivers) — presented at least as prominently as any non-standardized
        performance. Money market funds: current yield per Rule 482 with required
        disclosures (yield fluctuates, not insured/guaranteed, possible to lose money).
  (d)(6) Testimonials: a retail communication or correspondence containing a testimonial
        about investment advice or performance must prominently disclose (A) that it may not
        be representative of other customers' experience, (B) that it is no guarantee of
        future performance or success, and (C) if more than $100 in value was paid, that it
        is a paid testimonial. A testimonial on a technical aspect of investing must come
        from someone with the knowledge and experience to form a valid opinion.
  (d)(7) Recommendations (RETAIL COMMUNICATIONS only): must have a reasonable basis, and
        must disclose, if applicable, that the member is making a market in the security, that
        the member or its officers/partners have a financial interest (and its nature), and
        that the member managed or co-managed a public offering of the issuer within the past
        12 months; must provide or offer to furnish supporting information, and for a
        corporate equity give the price at the time of the recommendation. (d)(7)(A) does
        not apply to correspondence, institutional communications, 2241/2242 research
        reports, or communications that recommend only registered investment companies or
        variable insurance products (reasonable basis still required).
        (d)(7)(C) applies to RETAIL COMMUNICATIONS AND CORRESPONDENCE: neither may refer,
        directly or indirectly, to past specific recommendations that were or would have been
        profitable, unless a list of ALL recommendations of that type over at least the prior
        year is provided or offered, with name/date/nature/price for each and the legend
        "it should not be assumed that recommendations made in the future will be profitable
        or will equal the performance of the securities in this list."
  (d)(8) BrokerCheck: each member website must include a readily apparent reference and
        hyperlink to BrokerCheck on the initial page intended for retail investors and on any
        page with a professional profile of a registered person who serves retail investors.
  (d)(9) Prospectuses filed with the SEC are not subject to (d), except Rule 482
        advertisements and FWPs that must be filed, which are.

(e) USE OF FINRA'S NAME: a member may state it is a FINRA member, but may not state or imply
    that FINRA, any other regulator, or SIPC endorses, indemnifies, guarantees or has approved
    the member's business practices, selling methods, any class of securities, or any
    specific security. Reference to Advertising Regulation review is limited to "Reviewed by
    FINRA" / "FINRA Reviewed". "FINRA-regulated so your money is safe" and "FINRA-approved
    strategy" are violations. A website indication of FINRA membership must hyperlink to
    www.finra.org nearby.

(f) PUBLIC APPEARANCES: unscripted seminars, interviews, call-in shows and live chats are not
    retail communications but must satisfy (d)(1), and (d)(4)–(d)(6) where applicable; a
    person making a recommendation must disclose any financial interest in the security and
    any other actual, material conflict. Scripts and slides used ARE retail communications.

(g) Any violation of an SEC rule governing communications (e.g. Rule 482, 34b-1, 156) by a
    member is also a violation of Rule 2210.

RELATED RULES REVIEWERS CITE WITH 2210
  Rule 3110(b)     supervision and review of correspondence / internal communications
  Rule 4511 / SEA 17a-4(b)(4)   books and records; retention of communications
  Rule 2111 / Reg BI   suitability / best interest when a communication makes a recommendation
  Rule 2212  use of investment company rankings;  Rule 2213  bond fund volatility ratings
  Rule 2214  investment analysis tools;  Rule 2215  security futures;  Rule 2216  CMOs
  Rule 2220  options communications (options promotional material needs ODD delivery & special
             approval);  Rule 2330  variable annuity recommendations; Rule 2241 research reports
  Rule 2020  manipulative / deceptive devices; Rule 2010  high standards of commercial honor
  Rule 3240  borrowing from customers;  Rule 2040/2210 referral & finder compensation claims
  Regulatory Notice 17-18  social media, text/instant messaging, chatbots, hyperlinks
    (adoption / entanglement of third-party content), native advertising; Notice 19-31
    (FAQ on fund performance/hyperlink disclosures); Notice 24-09 (generative AI — rules are
    technology-neutral; firm is responsible for AI-generated communications).
"""

# ---------------------------------------------------------------------------
# Violation taxonomy
# ---------------------------------------------------------------------------

# category -> (primary rule cite, short description, which communication types it can apply to)
VIOLATION_CATEGORIES: dict[str, dict] = {
    "performance_projection": {
        "rule": "2210(d)(1)(F)",
        "desc": "Predicts or projects performance, implies past performance will recur, or gives an unwarranted forecast.",
        "applies_to": ["correspondence", "retail_communication", "institutional_communication"],
    },
    "promissory_or_guarantee": {
        "rule": "2210(d)(1)(B)",
        "desc": "Promissory, guaranteed, 'can't lose', 'risk-free', 'safe' language for an investment with market risk.",
        "applies_to": ["correspondence", "retail_communication", "institutional_communication"],
    },
    "exaggerated_or_unwarranted_claim": {
        "rule": "2210(d)(1)(B)",
        "desc": "Unsupported superlatives or claims ('best fund', '#1 advisor', 'never had a losing year') without substantiation.",
        "applies_to": ["correspondence", "retail_communication", "institutional_communication"],
    },
    "false_or_misleading_statement": {
        "rule": "2210(d)(1)(B)",
        "desc": "Materially untrue statement of fact (fees, insurance coverage, product features, credentials).",
        "applies_to": ["correspondence", "retail_communication", "institutional_communication"],
    },
    "not_fair_and_balanced": {
        "rule": "2210(d)(1)(A)",
        "desc": "Benefits presented without the corresponding risks; material qualification omitted; cherry-picked facts.",
        "applies_to": ["correspondence", "retail_communication", "institutional_communication"],
    },
    "misleading_in_context": {
        "rule": "2210(d)(1)(D)",
        "desc": "Individually true statements that mislead in combination or in the overall context.",
        "applies_to": ["correspondence", "retail_communication", "institutional_communication"],
    },
    "buried_disclosure": {
        "rule": "2210(d)(1)(C)",
        "desc": "Material risk or qualification relegated to a footnote/legend in a way that inhibits understanding.",
        "applies_to": ["correspondence", "retail_communication", "institutional_communication"],
    },
    "unsupported_comparison": {
        "rule": "2210(d)(2)",
        "desc": "Retail communication compares investments or services without disclosing material differences (objectives, costs, liquidity, safety, guarantees, tax). In correspondence/institutional, cite not_fair_and_balanced instead.",
        "applies_to": ["retail_communication"],
    },
    "member_name_missing": {
        "rule": "2210(d)(3)",
        "desc": "Retail communication or correspondence does not prominently disclose the member firm's name.",
        "applies_to": ["correspondence", "retail_communication"],
    },
    "tax_claim_incomplete": {
        "rule": "2210(d)(4)",
        "desc": "'Tax-free' / 'tax-exempt' claim without indicating which taxes apply (state, AMT, capital gains).",
        "applies_to": ["correspondence", "retail_communication"],
    },
    "fund_performance_disclosure": {
        "rule": "2210(d)(5)",
        "desc": "Fund performance shown without standardized 1/5/10-year returns as of quarter-end, max sales charge, and expense ratio, or non-standardized returns more prominent than standardized.",
        "applies_to": ["correspondence", "retail_communication"],
    },
    "testimonial_disclosure": {
        "rule": "2210(d)(6)",
        "desc": "Testimonial about advice or performance without 'may not be representative' / 'no guarantee' / paid-testimonial disclosures.",
        "applies_to": ["correspondence", "retail_communication"],
    },
    "recommendation_disclosure": {
        "rule": "2210(d)(7)",
        "desc": "Retail communication recommends a security (not a fund/variable product) without disclosing market-making, financial interest, or recent underwriting, without offering supporting information, or without the price for a corporate equity.",
        "applies_to": ["retail_communication"],
    },
    "past_recommendations_reference": {
        "rule": "2210(d)(7)(C)",
        "desc": "Retail communication or correspondence refers to past specific profitable recommendations ('I told clients to buy NVDA at $40') without the complete one-year list and required legend.",
        "applies_to": ["correspondence", "retail_communication"],
    },
    "brokercheck_link_missing": {
        "rule": "2210(d)(8)",
        "desc": "Member website page intended for retail investors, or a registered person's profile page, lacks a readily apparent BrokerCheck reference and hyperlink (website_page channel only).",
        "applies_to": ["retail_communication"],
    },
    "fund_ranking_disclosure": {
        "rule": "2212",
        "desc": "Retail communication uses an investment company ranking/rating without the Rule 2212 disclosures (ranking entity, category, number of funds, period, criteria, and that past performance is no guarantee).",
        "applies_to": ["retail_communication"],
    },
    "finra_endorsement_implied": {
        "rule": "2210(e)",
        "desc": "States or implies FINRA or SIPC endorses, guarantees or has approved the firm, its practices, or a security.",
        "applies_to": ["correspondence", "retail_communication", "institutional_communication"],
    },
    "unapproved_retail_communication": {
        "rule": "2210(b)(1)",
        "desc": "Content that is really a retail communication (template to >25 retail investors, seminar deck, mass email, social post) used without registered-principal pre-approval — the text or metadata must show it was already used/sent, not merely submitted for review.",
        "applies_to": ["retail_communication"],
    },
    "institutional_forwarding": {
        "rule": "2210(a)(4) / 2210(b)(3)",
        "desc": "Communication treated as institutional although the member has reason to believe it will reach retail investors (e.g. plan participants, an adviser's retail clients, 'feel free to share with your clients').",
        "applies_to": ["institutional_communication"],
    },
    "hypothetical_or_backtest_misuse": {
        "rule": "2210(d)(1)(F)",
        "desc": "Backtested or hypothetical results presented as if achievable, or an illustration that in substance projects an investment's performance.",
        "applies_to": ["correspondence", "retail_communication", "institutional_communication"],
    },
    "options_communication": {
        "rule": "2220 / 2210(d)(1)",
        "desc": "Options discussed with retail investors in a way that omits risks, implies limited risk, or promotes options before ODD delivery.",
        "applies_to": ["correspondence", "retail_communication"],
    },
    "product_specific_omission": {
        "rule": "2210(d)(1)(A)",
        "desc": "Product-specific material features omitted (surrender charges on annuities, liquidity of non-traded REITs/private placements, leverage reset in leveraged ETFs, call risk on structured notes, CMO prepayment risk under 2216).",
        "applies_to": ["correspondence", "retail_communication", "institutional_communication"],
    },
    "third_party_content_adoption": {
        "rule": "2210(d)(1) / RN 17-18",
        "desc": "Forwarding or linking third-party content (article, ranking, chart) that the member has adopted or become entangled with and that itself violates the content standards.",
        "applies_to": ["correspondence", "retail_communication"],
    },
}

RULE_CITES = sorted({v["rule"] for v in VIOLATION_CATEGORIES.values()})

COMMUNICATION_TYPES = ["correspondence", "retail_communication", "institutional_communication"]
SEVERITIES = ["none", "low", "medium", "high"]
DIFFICULTIES = ["easy", "medium", "hard"]

DIFFICULTY_GUIDE = """\
DIFFICULTY DEFINITIONS (label every sample with exactly one)

easy
  - Violation samples: a single, blatant violation a first-week compliance analyst would catch
    on sight (explicit guarantee, a numeric return forecast, "FINRA approved", "best fund in
    America", an obvious untrue fee statement). The rest of the message is unremarkable.
  - Compliant samples: routine, low-risk content — scheduling, account logistics, document
    requests, generic market commentary with no claims, a thank-you note. Nothing that even
    resembles a claim.

medium
  - Violation samples: one or two violations that require reading the whole message, not just
    spotting a keyword. Examples: an implied projection ("positioned to deliver strong results
    once rates fall"), benefits of a product listed with the risks omitted, a comparison that
    skips material differences, fund performance quoted without standardized returns, a
    "tax-free" claim that ignores state tax, a template that the metadata reveals went to 40
    clients (so it is an unapproved retail communication).
  - Compliant samples: messages that discuss products, performance, or recommendations and get
    it RIGHT — historical returns with full 2210(d)(5) disclosure, a hypothetical compounding
    illustration properly labeled, balanced pros and cons, correct muni tax language. These
    must look superficially similar to violation samples so the model cannot rely on topic.

hard
  - Violation samples: borderline or compound cases requiring judgment — statements that are
    individually true but misleading together (d)(1)(D); a material qualification buried in a
    footnote (d)(1)(C); an institutional communication the sender should know will be
    forwarded to retail; a backtest dressed up as an illustration; a forwarded third-party
    article the rep has adopted; interaction between the 25-recipient threshold and (d)(7)
    recommendation disclosures; hedged language ("should", "we expect", "on track to") that
    still constitutes a projection; a correct-looking performance table with the wrong as-of
    date or non-standardized returns given more prominence.
  - Compliant samples: messages that LOOK aggressive or sloppy but are actually compliant —
    strong opinions with a reasonable basis and balance, a price target inside a 2241-compliant
    research excerpt, a properly labeled hypothetical compounding illustration, a testimonial
    with all three disclosures, casual texting tone with no substantive claim, a complaint
    response that is blunt but accurate, a retail stock recommendation that carries every
    (d)(7) disclosure. Also messages where a tempting-but-wrong cite exists (e.g. a
    correspondence that recommends a stock — (d)(7)(A) does NOT apply to correspondence; a
    comparison in correspondence — (d)(2) does NOT apply, only (d)(1) balance does).

RETAIL COMMUNICATIONS (retail broker-dealer slice)
  A large share of samples are retail communications: mass email campaigns, newsletters, form
  letters to a rep's whole book, static social media posts, website and landing pages, blog
  articles, seminar invitations and decks/scripts, radio/podcast scripts, print ads, text
  blasts. Write them in the native format of the channel (a social post is short and has
  hashtags; a website page has headings and a footer; a seminar script has speaker lines).
  The review posture for retail communications is PRE-USE: the sample is what a principal
  sees before approval. required_actions therefore use "Approve for use" / "Return for
  revision" / "Route to Advertising Regulation filing within 10 business days of first use"
  (fund, ETF, variable product, DPP, CMO, structured product content) / "Pre-file 10 business
  days before use" (fund rankings not generally published; new member first year). Only use
  the unapproved_retail_communication category when the text or metadata shows the material
  was already distributed without approval.
"""

# ---------------------------------------------------------------------------
# Scenario axes — in-domain
# ---------------------------------------------------------------------------

# correspondence / institutional channels
IN_DOMAIN_CHANNELS = [
    ("email", 0.58),
    ("letter", 0.15),
    ("sms_text", 0.15),
    ("instant_message", 0.12),
]

# retail communication channels, by recipient profile
RETAIL_BOOK_CHANNELS = [          # distributed to the rep's or firm's client list (> 25)
    ("mass_email_campaign", 0.45),
    ("client_newsletter", 0.25),
    ("form_letter", 0.15),
    ("text_message_blast", 0.15),
]
RETAIL_PUBLIC_CHANNELS = [        # made available to the public
    ("social_media_post", 0.26),
    ("website_page", 0.20),
    ("blog_article", 0.12),
    ("seminar_invitation", 0.12),
    ("seminar_deck_or_script", 0.12),
    ("radio_or_podcast_script", 0.08),
    ("print_ad", 0.06),
    ("video_script", 0.04),
]

IN_DOMAIN_PRODUCTS = [
    ("mutual funds", 0.16),
    ("ETFs", 0.14),
    ("individual stocks", 0.14),
    ("bonds / CDs / fixed income", 0.10),
    ("401(k) rollover / IRA", 0.10),
    ("529 plans / education savings", 0.05),
    ("target-date / asset allocation funds", 0.07),
    ("fixed annuities", 0.04),
    ("municipal bonds", 0.05),
    ("general market commentary (no specific product)", 0.08),
    ("account services / logistics (no product)", 0.07),
]

IN_DOMAIN_SENDERS = [
    ("registered representative", 0.50),
    ("financial advisor (dual-registered)", 0.18),
    ("branch manager", 0.07),
    ("client service associate (unregistered, sending on rep's behalf)", 0.06),
    ("wholesaler / internal sales desk", 0.04),
    ("compliance or operations staff", 0.04),
    ("firm marketing department", 0.11),
]

IN_DOMAIN_RECIPIENT_SCENARIOS = [
    # (description, count range, recipient_type, expected communication_type)
    ("a single existing retail client", (1, 1), "retail", "correspondence"),
    ("a handful of retail clients (BCC)", (2, 12), "retail", "correspondence"),
    ("a book of retail clients just under the threshold", (18, 25), "retail", "correspondence"),
    ("a prospect who is not yet a customer", (1, 1), "retail", "correspondence"),
    ("the rep's or branch's full retail client list (mass distribution over the threshold)", (26, 2000), "retail", "retail_communication"),
    ("the general public / all retail visitors (website, social, seminar, ad)", (200, 50000), "retail", "retail_communication"),
    ("a single institutional investor (pension CIO, bank treasury, insurance GA)", (1, 3), "institutional", "institutional_communication"),
    ("an RIA who serves retail clients", (1, 2), "institutional", "institutional_communication"),
]
IN_DOMAIN_RECIPIENT_WEIGHTS = [0.28, 0.12, 0.05, 0.05, 0.16, 0.22, 0.07, 0.05]

TONES = [
    "formal", "friendly-professional", "casual / texting", "salesy", "apologetic (complaint response)",
    "terse / rushed", "educational / explanatory", "urgent (market event)",
]

LENGTHS = [("short (1-3 sentences)", 0.30), ("medium (1-2 paragraphs)", 0.50), ("long (3-6 paragraphs)", 0.20)]

# ---------------------------------------------------------------------------
# Scenario axes — out-of-distribution (each OOD sample shifts on 1-3 axes)
# ---------------------------------------------------------------------------

OOD_AXES: dict[str, list[str]] = {
    "channel": [
        "LinkedIn direct message", "X/Twitter DM", "WhatsApp message", "Slack/Teams message to a client",
        "firm chatbot transcript (bot speaking to a retail user)", "voicemail transcript",
        "video-call auto-generated summary sent to client", "multi-turn email thread (3-6 replies)",
        "Reddit/forum reply by a registered person", "webinar follow-up with attached slide text",
        "handwritten note scanned (OCR artifacts)", "Zelle/Venmo-style payment memo field",
    ],
    "product": [
        "listed options / covered calls", "crypto ETFs / spot bitcoin ETFs", "private placements (Reg D)",
        "structured notes / market-linked CDs", "variable annuities", "leveraged / inverse ETFs",
        "non-traded REITs", "BDCs / interval funds", "SPACs", "margin / securities-based lending",
        "CMOs / mortgage-backed securities", "security futures", "closed-end funds at a discount",
        "direct indexing / SMAs", "employee stock plan / concentrated position", "I bonds vs brokered CDs",
    ],
    "audience": [
        "institutional investor who will forward to retail", "401(k) plan sponsor (institutional) vs plan participants (retail)",
        "family office with $40M (NOT institutional — under $50M)", "a registered person at another member firm",
        "a non-English-speaking client (message in Spanish, same standards apply)", "a minor's custodian / UTMA",
        "deceased client's estate / beneficiary", "a journalist asking for comment",
    ],
    "style": [
        "heavy slang and emoji", "non-native English with grammar errors", "extremely long and rambling",
        "all caps / shouting", "copy-pasted marketing boilerplate with a personal line on top",
        "legalistic and over-disclaimed (but still violating or compliant?)", "sarcastic", "bullet-point fragments",
        "formatted as a table of numbers", "contains a URL to third-party content",
    ],
    "firm_type": [
        "bank-affiliated broker-dealer (FDIC confusion risk)", "insurance-company broker-dealer",
        "crypto-adjacent broker-dealer", "independent BD with 1099 reps", "robo-advisor hybrid",
        "new member firm in first year (pre-filing requirement)",
    ],
    "regulatory_twist": [
        "public appearance (live Q&A) follow-up — 2210(f)", "fund ranking from a third party — Rule 2212",
        "bond fund volatility rating — Rule 2213", "investment analysis tool output — Rule 2214",
        "research report excerpt with price target — Rule 2241", "SIPC coverage misstatement",
        "Reg BI / 2111 interplay with a recommendation", "recordkeeping failure implied by the message (e.g. 'delete this after reading', 'texting from my personal phone')",
        "borrowing from a customer — Rule 3240", "AI-generated draft sent without review — RN 24-09",
    ],
}

OOD_AXIS_WEIGHTS = {"channel": 0.28, "product": 0.28, "audience": 0.12, "style": 0.14, "firm_type": 0.08, "regulatory_twist": 0.10}

# ---------------------------------------------------------------------------
# Target label mix
# ---------------------------------------------------------------------------

DIFFICULTY_MIX = {"easy": 0.30, "medium": 0.40, "hard": 0.30}
COMPLIANT_RATE = 0.45  # fraction of samples that are compliant

# categories sampled for violation samples, weighted to roughly what review desks actually see
VIOLATION_WEIGHTS = {
    "performance_projection": 0.13,
    "promissory_or_guarantee": 0.10,
    "exaggerated_or_unwarranted_claim": 0.09,
    "false_or_misleading_statement": 0.07,
    "not_fair_and_balanced": 0.13,
    "misleading_in_context": 0.05,
    "buried_disclosure": 0.03,
    "unsupported_comparison": 0.07,          # retail only (applies_to filter)
    "member_name_missing": 0.04,
    "tax_claim_incomplete": 0.04,
    "fund_performance_disclosure": 0.07,
    "testimonial_disclosure": 0.04,
    "recommendation_disclosure": 0.05,       # retail only
    "past_recommendations_reference": 0.02,
    "brokercheck_link_missing": 0.02,        # website_page only (sampler filter)
    "fund_ranking_disclosure": 0.02,         # retail only
    "finra_endorsement_implied": 0.03,
    "unapproved_retail_communication": 0.02,
    "institutional_forwarding": 0.02,
    "hypothetical_or_backtest_misuse": 0.02,
    "options_communication": 0.00,   # in-domain: 0, OOD product axis turns these on
    "product_specific_omission": 0.01,
    "third_party_content_adoption": 0.00,
}
