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
Deaths zero future-value estimates; time-limit endings bootstrap final observations
without linking the next episode. This is parameter-shared PPO with local inputs,
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
