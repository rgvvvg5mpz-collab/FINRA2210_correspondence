# Board intro video: narration script

`FINRA2210_Board_Intro.mp4` · 1920×1080, 30 fps, 2:56 · narrated by the macOS voice **Ava (Premium)** (`say`), frames rendered from `FINRA2210_Board_Deck.html?video=1`.

| # | On screen | Narration |
|---|---|---|
| 1 | Title card (0:00) | A first-pass reviewer for FINRA Rule 2210. A short introduction for the board. |
| 2 | Slide 1 · Title (0:05) | Every email, text, post and web page a broker-dealer sends to the public must meet FINRA Rule 2210, and today compliance teams read each one by hand. We have built a small model that does the first pass: fine-tuned on 17,000 labeled messages, and running on the firm's own hardware. No message leaves the firm. |
| 3 | Slide 3 · What it does (0:25) | Here it is on a real text. A rep tells three clients a fund is a can't-lose 7% a year. The model calls it correspondence, flags it as not compliant, names the promissory claim with its rule paragraph, quotes the exact words, rates it high severity, and proposes next steps. The SME decides, and here would add one finding the model missed: the firm's name is absent. |
| 4 | Slide 2 · Taxonomy explorer (0:50) | It looks for 23 kinds of violation in 5 groups, from misleading content to missing disclosures, each pinned to one paragraph of the rule. |
| 5 | Slide 4 · Results (0:59) | On 1,186 messages it had never seen, the verdict was right 96.5% of the time, and the communication type 99.8%. On 5,800 deliberately unfamiliar messages, the verdict held at 95%. Naming the exact category is harder, at 80%, and that is the focus of the next round. |
| 6 | Slide 5 · Architecture (1:21) | It was built as a data factory: archived rule text, a taxonomy, scenarios sampled in code, 20 parallel Claude agents writing the messages and labels, and a checker that rejected anything with a wrong citation or an invented quote. Zero rejects, zero duplicates, and a fixed test set for every future version. |
| 7 | Slide 9 · Evaluation (1:42) | The untrained base model gets the verdict right 39% of the time. Fine-tuning takes it to 96%. Most of the remaining errors are a neighbouring category under the same rule. |
| 8 | Slide 11 · Serving (1:53) | In production it runs two ways. Real time, at the review desk: an estimated 1–2 seconds per message on one H100 GPU. Batched, for look-backs: a million messages overnight on one GPU, for a few cents per thousand before any enterprise discount. These are estimates, not measurements yet. |
| 9 | Slide 12 · AWS deployment (2:14) | On AWS it lives inside the firm's private network. A SageMaker endpoint serves the model, verdicts are kept in write-once storage for Rule 17a-4, and every expert decision becomes training data. A starter deployment is about $1,000 a month. |
| 10 | Slide 13 · Governance and roadmap (2:30) | Governance is built in. An SME makes every decision. Every verdict carries the model version and the rule-digest version it was judged by. A weekly watch flags changes to FINRA's rules. And no version ships unless it beats the current one on 44 checks. |
| 11 | Closing card (2:46) | We ask the board for three decisions: the scope of a supervised pilot, who reviews the labels, and access and licensing for the repository. Thank you. |

## Rebuilding

1. Render frames: open the deck with `?video=1` in headless Chrome at 1600×900 and screenshot the slides listed above (hides the progress bar and tap hints).
2. Narration: `say -v "Ava (Premium)" -o segNN.aiff -f segNN.txt` for each row; the spoken text spells out numbers and cites ("twenty-two ten").
3. Assemble with ffmpeg: one still per segment held for its narration plus 0.35 s of silence each side, 0.5 s crossfades, audio at 1.03× tempo, H.264 CRF 18 + AAC 192k.

Pricing and latency figures spoken in the video are the deck's estimates (w/o enterprise discount) and carry the same caveats as the deck.
