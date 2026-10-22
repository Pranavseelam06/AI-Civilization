# Experiment 14: sampling versus argmax

Date: October 22, 2026.

Compared the same frozen 1,000-update Experiment 14 carrying model on 30 matched
world seeds, 600000–600029, with a 500-tick horizon. No training, rewards, model
weights, or action masks changed. Sampling draws from the categorical model
probabilities; argmax takes the largest masked logit without drawing randomness,
resolving exact ties to the lowest action index. Initial worlds match; trajectories
subsequently diverge. These are previously inspected evaluation worlds, not a new
held-out test set. Each world has one sampled-policy seed (world seed + 10000).

| Metric | Sample | Argmax |
| --- | ---: | ---: |
| Mean final survivors / 12 | **10.47** | 5.87 |
| Mean capped lifespan, ticks | **490.3** | 404.2 |
| Worlds with all 12 alive | **12 / 30** | 0 / 30 |
| Pooled food-energy waste | 0% | 0% |
| Mean rejected movement fraction | **6.13%** | 14.89% |
| Total contested movement requests | 8,448 | 15,005 |
| Total stored meals | 5,758 | 4,386 |

Argmax had more survivors in two worlds, tied one, and had fewer in 27, with a
paired mean difference of −4.60 survivors. The sample mode exactly reproduced
all prior per-episode Experiment 14 evaluation metrics. Checkpoint SHA-256 was
verified unchanged after evaluation.

In this sample, always taking the highest percentage performed substantially
worse. All rejected moves were simultaneous destination conflicts: the local
mask cannot foresee other agents selecting the same square. Deterministic
selection can repeat the same conflicting actions; greater conflict counts
are observed here, but this test does not isolate the cause of every death.
Sampling remains the default. Training still samples as before; neither mode
requires retraining. This is not evidence that argmax is worse for all models.

## Compare or choose a mode

```bash
python -m learning.evaluate --food-carrying --checkpoints checkpoints/experiment_14.pt \
  --action-selection both --episodes 30 --start-seed 600000 \
  --output-dir runs/experiment_14_action_selection_repeat
```

Use `--action-selection sample` or `--action-selection argmax` for one mode.
Random and heuristic baselines run once and are unaffected by this option.
CSV rows, corner samples, metadata, and `summary.json` identify selection mode;
the summary includes paired final-survivor differences when comparing both.
Existing output is never overwritten. Recorded results are local in
`runs/experiment_14_sample_vs_argmax`.

In the viewer, select PPO and use **PPO action choice**, then Reset world to
apply it. Sampling is the default. To launch directly with argmax:

```bash
python -m visualization.sim3d --checkpoint experiment_14.pt --action-selection argmax
```
