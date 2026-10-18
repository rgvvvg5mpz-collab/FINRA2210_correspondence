# Docs

| File | What it is |
|---|---|
| [CHAT_TOPUP_PLAN.md](CHAT_TOPUP_PLAN.md) | the planned 2,400-item chat top-up (SMS, IM, text blasts): why, what, cost, expected effect, steps; not generated yet |
| [build_model_card_html.py](build_model_card_html.py) | renders `MODEL_CARD.md` into the standalone `MODEL_CARD.html` |

Rebuild the HTML model card after editing `MODEL_CARD.md` (needs `markdown-it-py`, in
`requirements-mlx.txt`):

```bash
.venv-mlx/bin/python docs/build_model_card_html.py
```

The output depends only on the Markdown, so rebuilding an unchanged card changes nothing.

Other documentation lives next to what it describes: the top-level [README](../README.md), the
[model card](../MODEL_CARD.md), and the READMEs in [data/](../data/README.md),
[training/](../training/README.md), [evaluation/](../evaluation/README.md),
[results/](../results/README.md) and [retraining/](../retraining/README.md).
