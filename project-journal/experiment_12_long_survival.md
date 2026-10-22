# Experiment 12: longer survival audit

Date: October 22, 2026.

Evaluated the unchanged `experiment_12_best.pt` checkpoint (700 training updates)
on 30 new worlds, seeds 500000–500029, with the horizon extended from 500 to
2,000 ticks; no retraining, action overrides, or reward changes were performed.
Each world used a stochastic policy seed equal to its world seed plus 10000.
The checkpoint file's SHA-256 hash was unchanged after evaluation.

| Tick | Mean survivors out of 12 |
| --- | ---: |
| 500 | 11.37 |
| 1,000 | 9.47 |
| 1,500 | 8.73 |
| 2,000 | 8.20 |

These are checkpoints along the same 30 trajectories, not different samples.

The mean lifespan capped at 2,000 ticks was 1,653.6 ticks; surviving agents'
eventual death times are unknown. There were 114 deaths among 360 agents,
including 95 after tick 500. No world retained all 12 agents through tick 2,000.
The agents consumed 22,651 food units with zero observed wasted food energy.

The policy sustains much of its population beyond its training horizon, but
survival continues to decline. Efficient eating alone does not guarantee
long-term survival. This audit does not isolate why agents die; it cannot
separate food scarcity, movement decisions, and competition as causes.
A useful next diagnostic is to inspect local food availability before deaths.

## Reproduce

```bash
python -m experiments.long_survival --output-dir runs/experiment_12_long_survival_repeat
```

The output includes `episodes.csv`, per-agent `deaths.csv`, `milestones.csv`,
`protocol.json`, and `summary.json` in the local output directory. Raw runs stay
local; this report is public. The audit script has tests for exact death timing,
extinction, later zero-population milestones, and horizon-censored survivors.
