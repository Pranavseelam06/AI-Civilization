# AI Civilization

[![Tests](https://github.com/Pranavseelam06/AI-Civilization/actions/workflows/tests.yml/badge.svg)](https://github.com/Pranavseelam06/AI-Civilization/actions/workflows/tests.yml)

AI Civilization is an experimental artificial-life project exploring whether agents can learn a communication protocol with no predefined symbol meanings that improves cooperation in a changing simulated world.

The project starts with a small ecosystem: agents observe their surroundings, move through a grid, collect food, and manage energy. Later stages introduce cooperative tasks and a limited signaling channel to study when learned communication becomes useful.

## Project status

Early development: a seeded grid simulator now supports movement, energy, death, food consumption, and regeneration. Random and heuristic policies, baseline evaluation, a live viewer, and saved replay playback are implemented. Behavioral tests cover these features; a shared-policy PPO training loop is available, while learned communication remains future work. Training effectiveness still requires held-out evaluation.

## Research goals

- Build a deterministic, reproducible simulation with local agent observations.
- Establish random and heuristic baselines before training policies.
- Train a compact shared policy for foraging and coordination.
- Compare learned communication with matched no-communication controls.
- Evaluate held-out maps, multiple training seeds, and message interventions.

The intended research outcome is evidence about task-specific learned signaling. Specialization, evolution, and persistent settlements are possible later extensions.

## Planned approach

The initial environment targets a 32 × 32 grid with 24 agents. Simulation and rendering remain separate so experiments can run headlessly. A later scout-and-gather task will give agents different information and capabilities, creating a measurable reason to communicate.

Planned technology includes Python and NumPy for simulation, Pygame for replay visualization, and PyTorch for learning. NumPy and PyYAML are currently declared dependencies; pytest and Pygame are optional development and viewer dependencies. PyTorch will be introduced with the learning stage.

## Repository structure

| Directory | Purpose |
| --- | --- |
| `civsim/` | World state, actions, observations, rewards, environment interface, and replay scaffolds |
| `configs/` | Simulation and experiment settings |
| `learning/` | Baseline policy scaffolds; trained policies are a later stage |
| `experiments/` | Evaluation and benchmark scaffolds |
| `visualization/` | Replay viewer scaffold |
| `tests/` | Behavioral test placeholders |
| `runs/` | Generated metrics and replays, ignored by Git |
| `checkpoints/` | Future saved policies, ignored by Git |

## Development setup

Python 3.10 or newer is required.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,viewer]'
python -m pytest
```

Run `python -m pytest` for behavioral tests. GitHub Actions runs the suite on pushes and pull requests using Python 3.10 and 3.14; the badge above reports its status. Tests run headlessly without requiring Pygame.

Run `python -m visualization.live_viewer` to watch the simulation or `python -m experiments.run_baselines` to compare random and heuristic policies. The live viewer prints the recording path; play it with `python -m visualization.pygame_viewer --replay runs/YOUR_FILENAME.jsonl`.

## Evaluation principles

Success is measured through reproducible experiments rather than visual appearance alone. Communication claims will require performance improvements over controls, sensitivity to message removal or shuffling, and evidence that signals carry task-relevant information. Negative results and limitations will be reported alongside successful experiments.

## 3D simulation viewer

With the project installed and your virtual environment active:

```sh
python -m visualization.sim3d
```

Open http://127.0.0.1:8001 and press **Start**. The viewer displays the complete
32 × 32 world with 12 agents, berry bushes, energy, death, and food regeneration.
Pause, step, adjust speed, or reset with a world seed and random/heuristic policy.
Drag to orbit, scroll to zoom, and right-drag to pan.

This uses the existing Python simulation; character appearances are cosmetic.
The browser downloads the free Three.js rendering library on loading, so an
internet connection is currently required. No API key or paid service is needed.
The static single-model preview remains available with
`python -m visualization.model_preview` on port 8000.

## Local PPO training

Install learning and test dependencies, then train headlessly:

```sh
python -m pip install -e '.[dev,learning]'
python -m learning.train --updates 20 --ticks 128
```

Each update collects fresh experience from one world before updating the shared
actor-critic network. Runs save configuration and loss/reward metrics to a unique
`runs/ppo_*` directory and learned weights plus optimizer state to `checkpoints/`.
CPU execution defaults to one thread. These are local operations without API fees.

To collect experience without updating weights:

```sh
python -m learning.train --collect-only --ticks 128 --output runs/rollout.npz
```

The implementation uses per-agent generalized advantage estimation, a clipped
PPO policy objective, value fitting, entropy regularization, and gradient clipping.
Deaths and finite episode endings have zero future value; ordinary rollout
cutoffs bootstrap the next observation without linking separate episodes. This is parameter-shared PPO with local inputs,
not a centralized-critic MAPPO system. See the [PPO paper](https://arxiv.org/abs/1707.06347).

Saved checkpoints do not include the live world state for exact training resumption.
The 3D viewer supports trained checkpoints as well as random/heuristic policies. A successful training run only verifies execution,
not improvement: compare trained checkpoints against baselines on held-out seeds.

### Watch a trained policy

```sh
python -m visualization.sim3d --checkpoint latest
```

Open http://127.0.0.1:8001 and press Start. To load a specific checkpoint, pass
`--checkpoint checkpoints/YOUR_FILENAME.pt`. The viewer also lists checkpoints
from the local `checkpoints/` folder: choose Trained PPO, select a checkpoint,
and Reset world. Refresh the page to discover files saved since it opened.
Policy actions are sampled from the loaded network with seed-controlled randomness.
Inference does not update model weights. The Pygame viewer retains its heuristic.

### Evaluate and diagnose a trained policy

```sh
python -m learning.evaluate --checkpoints latest --episodes 10
python -m learning.evaluate --checkpoints all --episodes 10
```

This compares checkpoints with random and heuristic baselines using matched seeds.
By default the evaluation range starts at 10000 or beyond a conservative bound
on recorded training seeds, whichever is greater. For checkpoints missing training
metadata, held-out status cannot be verified. Explicit `--start-seed` values need
to be checked against training seeds.

Each run saves `episodes.csv`, `metadata.json`, and (when corners are visited)
`corner_probabilities.csv` under a unique `runs/evaluation_*` directory.
Metrics include horizon-capped survival, food eaten, total reward, rejected move
reasons, and time near corners (within two cells of both edges). Intentional STAY
is reported separately and is not a rejected move. Corner traces contain up to
20 actual observations per episode, not an exhaustive or unbiased probability
summary. Diagnostic inspection does not change weights, sampling RNG, or rules.

### Legal-action masking

New PPO training runs mask moves outside the map, into obstacles, or into living
agents' pre-step cells. The mask comes only from the local observation. STAY
remains available; simultaneous contention for an empty cell still uses the
world's existing conflict rules. Rollouts save each mask so PPO computes its
probabilities using the same allowed actions as collection.

New checkpoints declare `action_masking: true`, and evaluation and the 3D viewer
apply it automatically. Older checkpoints retain their original unmasked behavior
for meaningful before/after comparisons. Train a fresh checkpoint to use masking;
this change does not rewrite existing weights. Random/heuristic baselines retain
their original behavior. Masking prevents invalid choices but does not itself
teach food-seeking.

### Controlled food-direction probes

```sh
python -m experiments.probe_food
```

Compares all saved checkpoints on synthetic, isolated interior observations with
no food or one adjacent food item north/south/east/west, at 25, 50, and 100 energy.
Console output shows the 50-energy cases; CSV saves all cases to `runs/food_probe_*`.
It measures changes from each action's no-food probability, without sampling
actions, changing weights, or advancing the world. These probes test local
food responsiveness; they do not establish full-world or multi-step performance.

### Current rewards

- Food reward when consumed: **1 − 2 × (energy before eating ÷ 100)**; Experiment 8’s objective is restored.
- Death: **−10**, once when it occurs.
- Alive at the configured episode end: **+5**, once.
- Ordinary living tick or already-dead agent: **0**.

At the final tick, eating and survival can combine for up to +6. Death on that tick
earns −10, not the survival bonus. With no horizon, there is no episode-end
survival bonus. Current training metadata records this as reward version 5 (restored to Experiment 8). Version 2 used +1 per consumed food unit; version 3 used energy restored ÷ 25.
Experiments 1–3 used the earlier +5-per-living-tick objective and are retained
as historical checkpoints. Experiment 4 uses the corrected objective.

### Test waiting versus exploration

```sh
python -m experiments.compare_exploration --checkpoint checkpoints/experiment_5.pt
```

Runs 30 matched seeds for normal PPO and a diagnostic variant that replaces a
sampled STAY with a random legal movement only when no food is visible. Agents
with no legal movement keep STAY. Food-visible choices and existing movement
choices are unchanged. The intervention uses a separate random generator.
Episode CSV, paired summaries, and metadata are saved in `runs/exploration_*`.
No checkpoint weights, rewards, or viewer behavior are changed. These are
evaluation interventions on one checkpoint, not retraining or evidence of
general improvement across independently trained policies.

### Explicit eating (Experiment 5 onward)

Agents now choose among six actions: STAY, NORTH, SOUTH, EAST, WEST, EAT.
Food is consumed only by EAT on the agent's current cell; there is no automatic
eating or coin flip. EAT restores up to 25 energy before the normal 1-energy
tick cost. Empty-cell or full-energy EAT is a no-op except for that tick cost.
Masks permit EAT only with food underfoot and energy below 100.

The heuristic eats at 75 energy or below; the neural network chooses its own
timing. Experiment 5 used +1 per consumed unit. Experiment 6 onward rewards actual
energy restored / 25, reducing the immediate reward for wasting food. World state, evaluation CSVs, and replay records
track energy restored and energy wasted at the 100-energy cap. The 3D viewer
shows the cumulative wasted energy.

```sh
python -m visualization.sim3d --checkpoint experiment_5.pt
```

Experiments 1–4 are historical five-action checkpoints trained with automatic
eating. They remain on disk, but cannot be loaded into the new six-action
world. The viewer and automatic evaluation/probe selection show only compatible
checkpoints. Past evaluation CSVs describe the old rules and should not be
compared directly against results from this changed task.

### Food efficiency reward (Experiment 6)

Eating at 50 energy restores 25 and earns +1; at 95 it restores 5 and earns
+0.2; at 99 it restores 1 and earns +0.04. Rewards use restored energy before
the normal tick cost. Death remains −10; surviving the episode remains +5 once.
Energy limits, food costs, regrowth, action masking, and the network are unchanged.
This reduces wasteful eating's immediate reward but does not guarantee
long-term conservation or cooperation between agents.

```sh
python -m visualization.sim3d --checkpoint experiment_6.pt
```

Experiments 5 and 6 share the six-action dynamics and can be compared on matched
world seeds. Evaluation always reports rewards under the current reward rule;
checkpoint metadata preserves the objective each policy was trained on.

Experiment 6 is an experimental checkpoint, not an established improvement.
On 30 matched evaluation seeds it underperformed Experiment 5 in both survival
and food-energy efficiency; keep Experiment 5 as the stronger reference.

### Signed food-efficiency reward (Experiment 7, version 4)

Consumed food earns `2 * (energy_restored / 25) - 1`. Restoring 25 energy
earns +1; restoring 5 earns −0.6; restoring 1 earns −0.92. No consumption earns
0, including EAT on empty land or at full energy. Rewards become negative when
less than half of a food unit is used. The minimum possible consumed-food gain
is 1 under current rules, so the lowest reachable food reward is −0.92.
Death remains −10 and end survival +5. Existing checkpoints retain their learned
weights; Experiment 6 was trained with version 3, not this new objective.

The signed-reward model is available as Experiment 7:

```sh
python -m visualization.sim3d --checkpoint experiment_7.pt
```

On 30 matched evaluation worlds, Experiment 7 wasted 16.8% of consumed food
energy versus Experiment 5's 48.5%, with mean survival of 392 versus 383 ticks,
but fewer final survivors (4.97 versus 5.70); these are single-training-seed results.

## Project journal

- [Project notes](project-journal/notes.md) — observations from building and watching the simulation.
- [Experiment log](project-journal/experiment_log.md) — one-sentence summaries of experiment changes and results.

These journal files are intended for the public repository; personal planning documents remain local.

### Hunger-based eating reward (Experiment 8, version 5)

Consumed food now earns `1 - 2 * (energy_before_eating / 100)`, replacing the
previous food reward rather than adding another bonus. At energy 20 it earns
+0.6; at 50 it earns 0; at 95 it earns −0.9. No food consumption earns zero,
even when an agent is hungry. Death stays −10 and episode survival stays +5.
There is no forced eating action or penalty for not eating below an energy
threshold; the policy still decides when to eat. This shaping may encourage
risky delays, so both survival and waste must be evaluated.

Experiment 8 is available with:

```sh
python -m visualization.sim3d --checkpoint experiment_8.pt
```

On 30 matched worlds it wasted about 0.5% of consumed food energy, versus 16.7%
for Experiment 7 and 48.6% for Experiment 5; final survivors averaged 4.50, 4.70,
and 6.10 respectively, so efficiency improved without improved survival.

### Smooth positive hunger reward (Experiment 9, version 6)

This reverted experiment used `0.1 + 0.9 * (1 - energy_before_eating / 100)` upon
successful consumption, replacing the signed hunger formula. Eating at energy
20 earns +0.82; at 50, +0.55; at 95, +0.145. No consumption earns zero.
Death remains −10 and episode-end survival +5, with no forced hunger threshold.
This tests whether always-positive eating rewards can balance survival and food
efficiency; it does not assume early eating or conservation is always optimal.

Experiment 9 is archived at `checkpoints/archive/experiment_9.pt`; this was a negative
result: on 30 matched worlds it averaged 2.37 survivors and 57.1% food-energy
waste, versus Experiment 8's 4.77 survivors and 0.5% waste, so it is not a
recommended replacement; it is excluded from normal viewer selection.

After Experiment 9, the code was restored to Experiment 8’s signed hunger
reward (version 5); Experiment 10 subsequently tests a zero-clipped variant.

### Zero-clipped hunger reward (Experiment 10, version 7)

Consumed food earns `max(0, 1 - 2 * (energy_before_eating / 100))`.
Below 50 energy it retains Experiment 8's positive food reward; at 50 or above
it earns zero rather than a penalty or bonus. Not eating also earns zero.
Death remains −10 and episode-end survival +5. Model, action masks, world
mechanics, and training duration are unchanged for the comparison.

```sh
python -m visualization.sim3d --checkpoint experiment_10.pt
```

On 30 matched worlds, Experiment 10 averaged 5.13 final survivors versus
Experiment 8's 3.70, but wasted 46.3% of food energy versus 0.7%; Experiment 5
averaged 6.37 survivors and 48.6% waste on those same seeds. The change improves
survival relative to Experiment 8 while giving up most of its food efficiency.
These are exploratory single-training-seed results, not a general guarantee.

### Low-energy missed-meal guidance (Experiment 11, version 8)

Food earns `energy_restored / 25`, and an agent below 30 energy receives
`-0.2 * (30 - energy_before_action) / 30` if food was underfoot before
its action but it did not choose EAT. The penalty gradually increases as
energy falls and is capped at −0.2. Missing actions count as STAY.

There is no penalty for searching without food, arriving on food this tick,
or choosing EAT. The check uses the pre-action state, not food that regrew or
became reachable after the decision. Death and episode-survival rewards are
unchanged; penalties may combine with death if the last available meal is skipped.
This explicitly introduces a hand-chosen guidance threshold at 30 energy.

```sh
python -m visualization.sim3d --checkpoint experiment_11.pt
```

On 30 matched worlds Experiment 11 averaged 3.67 survivors and 52.6% wasted
food energy, versus 2.17 survivors for Experiment 6 (same food reward without
the missed-meal penalty) and 4.03 survivors with 0.5% waste for Experiment 8.
It rarely skipped urgent available meals but did not solve the survival/efficiency
tradeoff; these are single-training-seed results.

### Active reference: Experiment 8

The active code is restored to Experiment 8's signed hunger reward, with no
missed-meal penalty; skipped-meal counters remain read-only diagnostics.
Experiments 9–11 are archived under `checkpoints/archive/`, so `latest`
selects Experiment 8 again. Earlier experiment descriptions above are historical.

```sh
python -m visualization.sim3d --checkpoint experiment_8.pt
```

### Longer training with periodic checkpoints (Experiment 12)

```sh
python -m learning.train --updates 1000 --ticks 128 --save-every 100 --log-every 100 --checkpoint checkpoints/experiment_12.pt
```

This keeps Experiment 8's reward and model unchanged and saves intermediate
stages to `checkpoints/experiment_12/update_0100.pt`, `update_0200.pt`, and so on.
Each file records its actual update count. The final checkpoint is separate.
Intermediate saves do not consume training randomness.

Choose a stage on validation seeds, then evaluate the fixed choice on a
different test-seed set rather than selecting the best model on test results.

Experiment 12's validation-selected 700-update checkpoint averaged 11.23 final
survivors out of 12 on 30 independent test worlds, versus Experiment 8's 4.50,
with no food-energy waste observed for the selected checkpoint; the final
1,000-update model averaged only 6.80 survivors.
See the [full stage-by-stage report](project-journal/experiment_12.md).

```sh
python -m visualization.sim3d --checkpoint experiment_12_best.pt
```

### Independent-seed replication (Experiment 13)

The same Experiment 8 setup is repeated with training seeds 1001, 10001,
and 100001, each for 1,000 updates with snapshots every 100. Each run selects
a checkpoint using validation seeds 350000–350009; the frozen selections are
then tested on separate seeds 400000–400029. No reward or PPO tuning is used.

`experiments.select_stages` provides reusable selection from evaluation CSVs,
checks matched distinct validation seeds, and records the selected paths and
criterion before final testing. It refuses to overwrite existing checkpoint
copies. Checkpoints are ranked by mean final survivors, then lower food waste,
longer survival, and earlier update.

All three new training seeds repeated the improvement over the original
200-update Experiment 8: selected models averaged 11.07, 10.60, and 9.93 final
survivors out of 12 on 30 independent test worlds, versus 4.00 for Experiment 8,
with at most 0.006% food-energy waste. The selected original Experiment 12
checkpoint averaged 11.57 on the same worlds.
See the [replication report](project-journal/experiment_13.md).

```sh
python -m visualization.sim3d --checkpoint experiment_13_best.pt
```

### Optional one-slot food carrying

Try the new mechanic with the heuristic in the 3D viewer:

```sh
python -m visualization.sim3d --food-carrying
```

Press Start, or enable **Food carrying · one slot** and reset a baseline world.
A red dot marks agents carrying food; the inspector shows their inventory.
`PICKUP` takes food from the current square into one slot, without feeding the
agent or awarding points. `EAT` uses carried food first, otherwise ground food,
restoring up to 25 energy. Both actions take one tick and its usual one-energy
cost. A full slot cannot collect more food, full-energy agents cannot eat,
and carried food is lost on death. No sharing or dropping is implemented yet.

Carrying is off by default. Experiments 1–13 use their original rules; selecting
an old PPO checkpoint automatically disables carrying. New carrying checkpoints
use seven actions and 102 observation values, including the inventory flag,
and appear separately in the viewer menu.

To train a **new** model for this task:

```sh
python -m learning.train --food-carrying --updates 200 --ticks 128 \
  --checkpoint checkpoints/food_carrying_1.pt --run-dir runs/food_carrying_1_training
```

Evaluate on new worlds and then view the model:

```sh
python -m learning.evaluate --food-carrying --checkpoints checkpoints/food_carrying_1.pt \
  --episodes 30 --output-dir runs/food_carrying_1_evaluation
python -m visualization.sim3d --checkpoint food_carrying_1.pt
```

The 200-update command is a starting run, not an established training budget for
learning storage. Rewards are unchanged: pickup earns nothing; consuming stored
food earns the existing hunger-based food reward. Only a short pipeline smoke
test has been run so far, so no learned carrying or survival improvement is claimed.

### Experiment 14: food carrying, updated to 1,000 updates

Extending the same seed and carrying setup from 200 to 1,000 updates raised mean
survivors from 4.63 to **10.47 out of 12** on the same 30 evaluation worlds and
reduced food-energy waste from 4.34% to **zero**. It improved survivor counts
in 29 worlds and tied in one, compared with the original 200-update model;
the carrying heuristic averaged 9.70 survivors. This is a single training seed
on reused evaluation worlds, not an independent replication.
See the [Experiment 14 report](project-journal/experiment_14.md).

```sh
python -m visualization.sim3d --checkpoint experiment_14.pt
```

Experiment 14 now loads the 1,000-update model and automatically enables
carrying. Restart the viewer to reload it; the original 200-update checkpoint
is preserved in the local checkpoint archive.
