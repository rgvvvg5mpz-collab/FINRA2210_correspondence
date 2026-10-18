# Training summary: `runs/phi4mini-2210-mlx`

- Status: finished; last logged iteration 6,636 of 6,636 (batch 4).
- Best validation loss: 0.287 at iteration 6,636.
- Last validation loss: 0.287 at iteration 6,636; first: 2.185.
- Overfitting check: validation loss is at or within 0.005 of its minimum; no sign of overfitting.
- Train loss (mean of last 10 reports): 0.248; gap to last validation loss: +0.039.
- Epoch boundary at iteration 3,318: train loss 0.315 in the 200 iterations before vs 0.280 after (second pass over seen examples).
- Loss spike at iteration 320: 4.935 vs neighbour median 0.496; check that validation loss recovered afterwards.

## Validation loss

| Iteration | Loss | Seconds |
|---|---|---|
| 1 | 2.185 | 341 |
| 400 | 0.563 | 358 |
| 800 | 0.393 | 356 |
| 1,200 | 0.366 | 355 |
| 1,600 | 0.345 | 354 |
| 2,000 | 0.334 | 355 |
| 2,400 | 0.327 | 436 |
| 2,800 | 0.316 | 399 |
| 3,200 | 0.313 | 441 |
| 3,600 | 0.307 | 349 |
| 4,000 | 0.302 | 373 |
| 4,400 | 0.299 | 357 |
| 4,800 | 0.294 | 321 |
| 5,200 | 0.291 | 356 |
| 5,600 | 0.290 | 353 |
| 6,000 | 0.289 | 354 |
| 6,400 | 0.288 | 340 |
| 6,636 | 0.287 | 354 |

![loss curve](loss_curve.png)
