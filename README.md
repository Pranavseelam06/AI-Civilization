# AI Civilization

[![Tests](https://github.com/Pranavseelam06/AI-Civilization/actions/workflows/tests.yml/badge.svg)](https://github.com/Pranavseelam06/AI-Civilization/actions/workflows/tests.yml)

AI Civilization is an experimental artificial-life project exploring whether agents can learn a communication protocol with no predefined symbol meanings that improves cooperation in a changing simulated world.

The project starts with a small ecosystem: agents observe their surroundings, move through a grid, collect food, and manage energy. Later stages introduce cooperative tasks and a limited signaling channel to study when learned communication becomes useful.

## Project status

Early development: a seeded grid simulator now supports movement, energy, death, food consumption, and regeneration. Random and heuristic policies, baseline evaluation, a live viewer, and saved replay playback are implemented. Behavioral tests cover these features; model training and learned communication remain future work.

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
