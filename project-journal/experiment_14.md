# Experiment 14: one-slot food carrying

Updated: October 22, 2026. The viewer now uses the final **1,000-update** model.

## Task and training

One carried-food slot adds PICKUP as action 6 and a 102nd observation value
indicating stored food. EAT uses inventory first, otherwise ground food.
PICKUP earns no reward or energy; the unchanged hunger-based food reward
applies only when eating. All actions cost a tick and one energy, carried
food is lost on death, and no forced eating rule is used.

Both training runs used seed 123, 128 ticks per PPO update, the default 32×32
world with 12 agents and a 500-tick horizon, and unchanged PPO parameters.
The initial run trained for 200 updates (25,600 world ticks / 201,979 agent
transitions). At the user's request, training was extended to 1,000 total
updates (128,000 world ticks / 1,337,875 agent transitions), saving every 100.

The trainer cannot resume the exact world state from a checkpoint, so the
longer run reproduced training from the same seed. Its update-200 weights
and transition count exactly matched the original before further training.
The final update 1,000 was designated for use before evaluation, without
selecting a stage on test results. The network has 102 inputs, two 128-unit
hidden layers, seven outputs, and action masking (version 3).

## Evaluation

Both models were evaluated on the same 30 worlds, seeds 600000–600029, for
500 ticks. Initial world and policy RNG seeds match; subsequent trajectories
can diverge. These worlds were held out from training, but reused for this
training-duration comparison after inspecting the initial results, so this
is not a fresh independent confirmation of the extension.

| Metric | Random | Carrying heuristic | 200 updates | 1,000 updates |
| --- | ---: | ---: | ---: | ---: |
| Mean survivors / 12 | 0.00 | 9.70 | 4.63 | **10.47** |
| Mean capped lifespan (ticks) | 153.2 | 466.7 | 391.7 | **490.3** |
| Total food picked up | 460 | 3,323 | 4,764 | 6,011 |
| Total stored meals consumed | 431 | 3,170 | 4,674 | 5,758 |
| Total meals consumed | 865 | 6,116 | 4,688 | 5,997 |
| Food-energy waste | 11.45% | 0.00% | 4.34% | **0.00%** |
| Carried food lost on death | 29 | 0 | 19 | 2 |
| Food still carried at horizon | 0 | 153 | 71 | 251 |

The 1,000-update model improved final survivor counts in 29 paired worlds,
tied in one, and lost none compared with the 200-update model, averaging
5.83 additional survivors. All 12 agents survived in 12 of the 30 worlds.
Its mean survival exceeds the carrying heuristic on this sample, with zero
observed food-energy waste. This supports longer training for this seed, but
one training seed does not establish robustness. Inventory usage alone does
not establish strategic long-term storage; holding durations were not measured.
Earlier Experiments 1–13 used different action/observation rules, so their
historical scores are not a controlled comparison with this carrying task.

Waste is pooled wasted energy divided by 25 times food consumed; lifespans
are capped at 500 ticks. Random samples seven actions without masking; the
heuristic uses its documented hand-written carrying rules.

## Run the updated model

```bash
python -m visualization.sim3d --checkpoint experiment_14.pt
```

Restart the server and refresh the page to reload weights. Selecting Experiment
14 automatically enables carrying. Click an agent to inspect its inventory and
seven action probabilities; a red dot means it carries food.

The original 200-update model is preserved locally at
`checkpoints/archive/experiment_14_update_0200.pt`; the original training and
evaluation records remain intact. Extension snapshots are in
`checkpoints/experiment_14_extended/update_*.pt`. No extra stages are shown in
the experiment menu, and the requested 1,000-update checkpoint is the active one.

## Reproduce

Use unused output paths to preserve existing results:

```bash
python -m learning.train --food-carrying --updates 1000 --ticks 128 --seed 123 \
  --save-every 100 --log-every 100 --checkpoint checkpoints/experiment_14_repeat.pt \
  --run-dir runs/experiment_14_repeat_training
python -m learning.evaluate --food-carrying --checkpoints checkpoints/experiment_14_repeat.pt \
  --episodes 30 --start-seed 600000 --output-dir runs/experiment_14_repeat_evaluation
```

Original local records: `runs/experiment_14_protocol.json`,
`runs/experiment_14_training`, `runs/experiment_14_evaluation`.
Extended local records: `runs/experiment_14_extended_protocol.json`,
`runs/experiment_14_extended_training`, `runs/experiment_14_extended_evaluation`.
Protocols retain checkpoint hashes and the predeclared setups; evaluations
include per-episode metrics and summaries. Model weights and raw runs stay
local; this report and the experiment log retain the public findings.
