# Experiment 13 — Independent training-seed replication

Repeated Experiment 8's unchanged setup with three new independent training
seeds: **1001, 10001, and 100001**.
Each run trained for 1,000 updates (128 world ticks per update), saving every
100 updates, using the same reward version 5, six-action actor-critic, PPO
settings, and simulation rules as Experiment 12.

## Protocol fixed before evaluation

- Validation: ten matched worlds, seeds **350000–350009**.
- Select each run's checkpoint by highest mean final survivors; break ties by
  lower food-energy waste, longer mean survival, then earlier update.
- Freeze the selections before testing.
- Test: thirty different matched worlds, seeds **400000–400029**.
- Compare with original Experiment 8 and the selected Experiment 12 checkpoint.
- Select the overall replication checkpoint using validation results only.

The training seeds, validation seeds, and test seeds were recorded in
`runs/experiment_13_protocol.json` before training started.
All thirty stage files were checked for the correct seed, reward version,
actual update count, and world-tick count.

## Validation-selected stages

| Training seed | Selected update | Mean survivors / 12 | Food energy wasted |
| ---: | ---: | ---: | ---: |
| 1001 | 800 | 10.70 | 0.000% |
| 10001 | 500 | 10.60 | 0.000% |
| 100001 | 400 | 10.10 | 0.014% |

The overall validation choice was seed **1001**, update **800**, copied unchanged
to `checkpoints/experiment_13_best.pt`; this choice was not revised after testing.
Each seed's selected copy is also available separately.

## Independent test results

| Model | Selected update | Mean final survivors / 12 | Mean survival ticks | Food energy wasted |
| --- | ---: | ---: | ---: | ---: |
| Experiment 8, seed 123 | 200 | 4.00 | 375.3 | 0.410% |
| Experiment 12, seed 123 | 700 | 11.57 | 495.9 | 0.000% |
| Experiment 13, seed 1001 | 800 | 11.07 | 490.1 | 0.000% |
| Experiment 13, seed 10001 | 500 | 10.60 | 482.5 | 0.000% |
| Experiment 13, seed 100001 | 400 | 9.93 | 475.6 | 0.006% |

**All three new selected training seeds outperformed Experiment 8** on these
matched test worlds, averaging **10.53 survivors**
with an across-training-seed standard deviation of **0.57**.
This supports repeatability of effective, food-efficient foraging in the current
world setup, beyond the original favorable training seed.

The original Experiment 12 checkpoint still had the highest test mean; the
replication does not require every new seed to beat that selected reference.
Performance also fluctuates across training stages, so checkpoint selection
remains part of the procedure.

These are three new training seeds, not ninety independent worlds: each model
was evaluated on the same thirty test worlds, and all stage selection used a
separate shared validation set.
The result is limited to the current map size, food dynamics, population, and
observation setup; other task settings still need separate tests.

## Watch the validation-selected replication model

```sh
python -m visualization.sim3d --checkpoint experiment_13_best.pt
```

The stronger original reference remains available:

```sh
python -m visualization.sim3d --checkpoint experiment_12_best.pt
```

## Artifacts and tooling

- Training: `runs/experiment_13_seed1001_training/`,
  `runs/experiment_13_seed10001_training/`, and
  `runs/experiment_13_seed100001_training/`.
- Validation episodes, stage summaries, and locked selections:
  `runs/experiment_13_validation/`.
- Independent test episodes, metadata, and summary:
  `runs/experiment_13_test/`.
- Intermediate checkpoints: `checkpoints/experiment_13_seed*/update_*.pt`.
- Selected copies: `checkpoints/experiment_13_seed*_best.pt`.
- Reusable selector: `python -m experiments.select_stages --help`.

The selector verifies matched distinct validation seeds and refuses to overwrite
checkpoint outputs; all selected copies were checked against their source bytes.
The test suite contains **143 passing tests**.
