# Instructions for a generation agent

You are writing a labeled training corpus of broker-dealer communications reviewed under
FINRA Rule 2210. You were given a SPLIT (`in_domain` — this is a targeted top-up, there is no OOD part) and a CHUNK id (e.g. `chunk_0012`).
Working directory: /Users/saxena/Desktop/ClaudeCode_GIT/FINRA2210_correspondence

1. Read `data/chunks_topup/BRIEF.md` in full: role (senior FINRA Advertising Regulation examiner),
   Rule 2210 digest, category -> rule cite table, difficulty definitions, retail communications
   guidance, writing and labeling rules. Follow it exactly.

2. Read `data/chunks_topup/<SPLIT>/<CHUNK>.json`: 24 scenario specs. Produce exactly one sample per
   scenario, same order, echoing each scenario_id.

   RESUME: if `data/agent_out_topup/<SPLIT>/<CHUNK>.json` already exists, a previous agent was cut
   off part-way. Do NOT rewrite it. Run the checker (step 4) first: keep every sample it reports
   as ok, fix the samples it reports problems for (e.g. a missing generator_notes key), and write
   only the scenarios it reports as MISSING, appending them in scenario order. Read the existing
   samples so your new ones do not reuse their names, firms or phrasing.
   - TOP-UP: every scenario carries an `angle`: the concrete way the target category shows up
     (or, for compliant ones, the near-miss it narrowly avoids). Build the message around it.
     The FIRST entry of violation_categories is the target category and must be the clearest,
     best-evidenced violation in the message, with an excerpt that shows it:
       * brokercheck_link_missing (2210(d)(8)): website_page only; the page is intended for retail
         investors (rep profile, firm/branch/team page, service page). Write the page text, nav and
         footer as rendered text (links as [text](url)). The excerpt is the part of the page where the
         BrokerCheck reference/hyperlink should be (e.g. the rep's name/credentials block or footer).
       * institutional_forwarding (2210(a)(4) / 2210(b)(3)): the recipient really is institutional,
         but the text gives the member reason to believe it will reach retail investors.
       * unapproved_retail_communication (2210(b)(1)): the text or metadata (subject,
         recipient_description) must show the piece was ALREADY used/sent/posted without
         registered-principal approval; merely submitting it for review is not a violation.
     Compliant near-misses must look like they are on the edge of that category and still be clean.
   - OOD scenarios carry an `ood_shifts` object (channel / product / audience / style /
     firm_type / regulatory_twist). Those shifts are the whole point of the sample: a chatbot
     transcript reads like a chatbot transcript, a Spanish-language message is in Spanish, an
     options or structured-note message shows the product's real features, a regulatory_twist
     shows up concretely in the text.

3. Write `data/agent_out_topup/<SPLIT>/<CHUNK>.json` as `{"samples": [...]}`. Each sample has exactly
   these keys: scenario_id, difficulty, channel, metadata {sender_role, firm, recipient_description,
   recipient_count_30d (integer), recipient_type, subject}, communication, label
   {communication_type, compliant, violations: [{category, rule, excerpt, explanation}], severity,
   required_actions: [..], rationale}, generator_notes.

   Hard constraints (the checker enforces them):
   - metadata.recipient_count_30d, metadata.recipient_type, difficulty, compliant and
     label.communication_type must equal the scenario's values; channel must be the scenario's channel.
   - Violation scenarios: every category in violation_categories must appear (at most one natural
     extra); each rule cite is the one the brief's table assigns; each excerpt is a verbatim
     substring of communication (<= 200 chars); one entry per category; only categories that
     apply to the communication_type ((d)(2) and (d)(7)(A) are retail_communication only;
     (d)(3)-(d)(6) never apply to institutional_communication).
   - Compliant scenarios: violations = [], severity = "none", and the message must handle the
     near_miss_topic correctly so it is not trivially safe.
   - communication is the full text as sent, in the native format of the channel (social post,
     website page, seminar script, newsletter, transcript, etc.), 40-6000 chars.

   Quality: write like the specified sender in the specified channel, tone and length. Keep labels
   tight: rationale 2-3 sentences, each explanation 1-2 sentences, generator_notes ONE sentence.
   Fictional people and firms only; vary names, firms, cities, amounts, dates (2023-2026) and
   voice; no two samples share phrasing or structure. Valid JSON (escape newlines as \n).
   Write the file in two passes of 12 samples (write, then edit to append the rest) so progress
   is saved.

4. Run the checker:
   `cd /Users/saxena/Desktop/ClaudeCode_GIT/FINRA2210_correspondence && PYTHONPATH=src .venv/bin/python -m finra2210.check data/chunks_topup/<SPLIT>/<CHUNK>.json data/agent_out_topup/<SPLIT>/<CHUNK>.json`
   Fix every reported problem by editing your output file and re-run until it reports
   "0 with problems, 0 missing". Then run it once more adding
   `--emit data/agent_out_topup/<SPLIT>/<CHUNK>.raw.jsonl --teacher <TEACHER>`
   where <TEACHER> is the model id you were told you are running as.

5. ISOLATION: up to 20 agents run in parallel and share the scratchpad. If you write any helper
   script or temp file, put it ONLY in `data/agent_work/topup/<SPLIT>/<CHUNK>/` (create it) and give it
   no generic name. Never run, edit or read helper files you did not create, and never write to
   any output file other than your own `<CHUNK>.json` / `<CHUNK>.raw.jsonl`.

6. Do not modify anything under src/ or data/chunks_topup/. Do not run other commands. Reply with
   ONE line: chunk id, samples written, checker iterations, any scenario you could not satisfy and why.
