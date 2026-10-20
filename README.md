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

- Food reward when consumed: **2 × (energy restored ÷ 25) − 1** (−1 to +1).
- Death: **−10**, once when it occurs.
- Alive at the configured episode end: **+5**, once.
- Ordinary living tick or already-dead agent: **0**.

At the final tick, eating and survival can combine for up to +6. Death on that tick
earns −10, not the survival bonus. With no horizon, there is no episode-end
survival bonus. Current training metadata records this as reward version 4. Version 2 used +1 per consumed food unit; version 3 used energy restored ÷ 25.
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

### Current signed food reward (version 4)

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
