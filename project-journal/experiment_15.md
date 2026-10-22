# Experiment 15: ten-tick carried-food decay

Date: October 22, 2026.

## Rules

Only carried food decays. Pickup sets its freshness timer to 10 without aging
it on the pickup tick. Each subsequent tick permits an action before decreasing
the timer: eating at freshness 1 is valid; otherwise reaching zero removes the
food without energy or reward. For a pickup completed at tick 20, actions
completing at ticks 21–30 may consume it, and uneaten food expires at the end
of tick 30. Attempting PICKUP with a full slot does not reset freshness.
Ground food, regrowth, wall observations, collision rules, and the existing
hunger-based eating reward remain unchanged. No spoilage penalty or forced
EAT rule is added. Inventory is still lost on death, counted separately from
spoilage. The inspector displays remaining freshness, and replays preserve it.

Decay is optional and defaults off. Experiments 1–14 retain their original
rules. Experiment 15 adds a 103rd observation value: remaining freshness
normalized by the food lifetime. It retains seven actions. The checkpoint
selects the matching decay rules automatically; incompatible modes are rejected.

## Training

A fresh network trained with seed 123 for 1,000 PPO updates of 128 world ticks,
using the same PPO parameters as Experiment 14, a 32×32 world, 12 agents, and
a 500-tick horizon. The run produced 128,000 world ticks and 1,319,025 agent
transitions, saving checkpoints every 100 updates. The final 1,000-update model
was selected before evaluation; no test-based stage selection was performed.
Training and evaluation both use sampled actions, not argmax.

## Evaluation on 30 new worlds

World seeds: 700000–700029. All variants used the same decay task and initial
world/policy seeds; trajectories subsequently diverge. These seeds were not
used for training. The heuristic retains its original 75-energy eating rule
and does not anticipate expiry, so it is a simple reference rather than an
optimized decay policy.

| Metric | Random | Heuristic | Experiment 15 |
| --- | ---: | ---: | ---: |
| Mean survivors / 12 | 0.03 | 7.83 | **8.67** |
| Mean capped lifespan (ticks) | 145.7 | 431.7 | **470.3** |
| Worlds with all 12 alive | 0 / 30 | 0 / 30 | 3 / 30 |
| Food picked up | 402 | 2,876 | 199 |
| Stored meals consumed | 313 | 1,203 | 190 |
| Total meals consumed | 746 | 5,439 | 5,642 |
| Carried food spoiled | 76 | 1,622 | 9 |
| Carried food lost on death | 13 | 0 | 0 |
| Food still carried at horizon | 0 | 51 | 0 |
| Pooled energy waste while eating | 11.55% | 0.00% | **0.00%** |

Food-energy waste measures unused energy when eating, not spoiled items;
spoilage is reported separately. Inventory conservation holds: pickups equal
stored meals plus spoilage plus inventory lost on death plus food still carried.

The model ate 5,452 meals directly from the ground and only 190 from inventory,
so it primarily uses direct eating under the short storage lifetime. Nine of
its 199 pickups spoiled. This supports task adaptation, not a claim that it
learned strategic storage. Its mean survival exceeds the simple heuristic on
this sample, but one training seed does not establish robustness. Experiment
14's historical 10.47-survivor score used different world seeds and no decay;
it is not a controlled comparison against this run.

## Run

```bash
python -m visualization.sim3d --checkpoint experiment_15.pt
```

Restart the server and refresh, or select Experiment 15 and reset. The viewer
automatically enables the 10-tick lifetime. Click an agent to inspect freshness;
the global Spoiled counter is separate from Wasted energy. For a baseline demo,
run `python -m visualization.sim3d --food-decay` or enable the decay checkbox
and reset a baseline world.

## Reproduce

Use unused paths to preserve existing checkpoints and evaluation results:

```bash
python -m learning.train --food-decay --updates 1000 --ticks 128 --seed 123 \
  --save-every 100 --log-every 100 --checkpoint checkpoints/experiment_15_repeat.pt \
  --run-dir runs/experiment_15_repeat_training
python -m learning.evaluate --food-decay --checkpoints checkpoints/experiment_15_repeat.pt \
  --episodes 30 --start-seed 700000 --output-dir runs/experiment_15_repeat_evaluation
```

Local protocol: `runs/experiment_15_protocol.json`; training metadata and
metrics: `runs/experiment_15_training`; evaluation episodes, metadata, corner
samples, and summary: `runs/experiment_15_evaluation`. Checkpoint SHA-256 was
verified unchanged across evaluation. The 174-test suite covers exact expiry,
last-tick consumption, no freshness refresh from full-slot pickup, stale food,
ground-food preservation, reset/death, replay, metrics, and model compatibility.
