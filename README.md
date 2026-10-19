# AI Civilization

AI Civilization is an experimental artificial-life project exploring whether agents can learn a communication protocol with no predefined symbol meanings that improves cooperation in a changing simulated world.

The project starts with a small ecosystem: agents observe their surroundings, move through a grid, collect food, and manage energy. Later stages introduce cooperative tasks and a limited signaling channel to study when learned communication becomes useful.

## Project status

Early development: the repository currently contains a Python scaffold, example configuration, TODO modules, and skipped test placeholders. The simulator, learning system, and replay viewer are not implemented yet. There are no experimental results to report.

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

Tests currently skip until their behavior is implemented. The experiment and viewer modules are placeholders and do not yet run simulations or display replays.

## Evaluation principles

Success is measured through reproducible experiments rather than visual appearance alone. Communication claims will require performance improvements over controls, sensitivity to message removal or shuffling, and evidence that signals carry task-relevant information. Negative results and limitations will be reported alongside successful experiments.
