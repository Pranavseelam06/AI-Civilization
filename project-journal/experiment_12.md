# Experiment 12 — More training, unchanged reward

Trained the Experiment 8 setup from scratch for 1,000 PPO updates, saving every
100 updates, with seed 123 and 128 world ticks per update.
The 200-update checkpoint exactly matches the original Experiment 8 weights,
confirming that this extends the same training trajectory.

The reward, model, world, and PPO settings stayed unchanged: food reward is
`1 - 2 * energy_before_eating / 100`, death is −10, and surviving the episode is +5.

## Validation: choosing a checkpoint

All stages were evaluated on the same ten validation worlds, seeds 200000–200009.
Selection maximized mean final survivors, with lower pooled food-energy waste,
longer survival time, then earlier update as tie-breakers.

| Training updates | Mean final survivors / 12 | Food energy wasted |
| ---: | ---: | ---: |
| 100 | 1.40 | 10.37% |
| 200 | 4.70 | 0.62% |
| 300 | 11.30 | 0.08% |
| 400 | 6.40 | 0.00% |
| 500 | 9.00 | 0.00% |
| 600 | 8.30 | 0.00% |
| 700 | 11.60 | 0.00% |
| 800 | 8.30 | 0.00% |
| 900 | 11.20 | 0.00% |
| 1000 | 7.10 | 0.00% |

**700 updates was selected before running the final test**, and its unchanged
weights were copied to `checkpoints/experiment_12_best.pt` for easy viewer access.
The 1,000-update checkpoint is retained separately as `checkpoints/experiment_12.pt`.
All ten stages remain under `checkpoints/experiment_12/`.

## Independent test: checking the fixed selection

The selected checkpoint, original Experiment 8, and final 1,000-update checkpoint
were tested on 30 separate worlds, seeds 300000–300029; these seeds were not used
for training or checkpoint selection.

| Model | Training updates | Mean final survivors / 12 | Mean survival ticks | Food energy wasted |
| --- | ---: | ---: | ---: | ---: |
| Experiment 8 | 200 | 4.50 | 387.4 | 0.48% |
| Selected Experiment 12 | 700 | 11.23 | 491.1 | 0.00% |
| Final Experiment 12 | 1000 | 6.80 | 427.8 | 0.00% |

Longer training helped substantially at the selected stage, but performance was
not monotonic: the final model was worse than the selected intermediate one.
No food-energy waste was observed for the selected checkpoint in this test;
that is not a guarantee for all future settings.
This is one training seed, so repeat independent training seeds before making
a general claim about reliability.

## Watch the selected model

```sh
python -m visualization.sim3d --checkpoint experiment_12_best.pt
```

## Reproduce training

```sh
python -m learning.train --updates 1000 --ticks 128 --save-every 100 --log-every 100 --checkpoint checkpoints/experiment_12_new.pt
```

The new checkpoint name avoids overwriting the completed experiment.
Raw training metrics, validation results, locked selection, and independent test
results are saved locally under `runs/experiment_12_training/`,
`runs/experiment_12_validation/`, and `runs/experiment_12_test/`; the predeclared
protocol is `runs/experiment_12_protocol.json`.
