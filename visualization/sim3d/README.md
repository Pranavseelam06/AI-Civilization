# 3D simulation viewer

Run from the project root with your virtual environment active:

```sh
python -m visualization.sim3d
```

Open http://127.0.0.1:8001 and press Start. Pause, advance a single tick,
adjust the requested tick rate, or reset with a seed and policy. Camera controls:
drag to orbit, scroll to zoom, right-drag to pan; Camera restores the overview.
Stop the server with Ctrl+C. The window can remain open at the episode end.

This is the existing Python World with its food, energy, death, regeneration,
and random/heuristic policies. The browser draws state and requests steps; it
does not duplicate the simulation rules. Seed 123 matches baseline world 123
and its policy RNG seed of 10123. Rendering never advances a tick on its own.
The simulation advances while this page is running; use one viewer tab per server.
All tabs on the same server share one simulation.

The 32 × 32 ground tiles and food bushes use instanced meshes. All 12 agents
cycle through the four character appearances for display only. Children and adults
have no different simulation capabilities. Dead agents disappear. Characters
slide between cells; the supplied GLBs contain no walking animations.

No paid service or API key is needed. The browser currently downloads pinned
Three.js 0.180.0 from jsDelivr, so initial loading requires internet. Models and
simulation are served locally. The original Pygame viewers and recording features
remain available; this 3D viewer does not yet record runs.

## Trained PPO

Install the learning extra (`python -m pip install -e '.[learning]'`).
Run `python -m visualization.sim3d --checkpoint latest` to start with the newest
`.pt` file in `checkpoints/`, or supply a specific checkpoint filename.
Alternatively choose Trained PPO and a checkpoint in the UI, then Reset world.
Refresh the page to update the checkpoint list after training. Baseline policies
still work without PyTorch installed. All inference runs locally on CPU without
gradients or optimizer updates. Use one browser tab per simulation server.

The experiment menu contains one entry for each Experiment 1–14. Experiments
12 and 13 use their validation-selected best checkpoints. Search by number
(e.g. `4`) above the controls and press Enter to load that experiment, then
press Start. Clear the search to see the full list. Experiments 1–4 load their
five-action networks with the original seeded 50% automatic-eating behavior;
later experiments use explicit EAT. This is inference only; no weights change.

### Inspecting agents

Click a person in the world, or choose an agent in the inspector dropdown.
Gold squares show the on-map portion of that agent's 5×5 field of view; a ring
marks the selected person. Dragging the camera does not select agents. Click
empty land or choose None to clear the selection. Pause or use Step to inspect
individual ticks. Dead agents remain available in the dropdown without an overlay.

The inspector shows current energy, the latest requested action and its movement
outcome, and (for PPO) the next-action probabilities from the current observation.
These probabilities refer to the next tick, not the observation used for the
latest action. Reading them does not sample actions, advance simulation RNG, or
update model weights. Heuristic/random policies have no neural model probabilities;
Experiments 1–4 show five probabilities with automatic eating. Finished episodes
show no next-action probabilities. These values do not represent verbal thoughts.

### Food carrying

Start with `python -m visualization.sim3d --food-carrying` to try the mechanic
using the heuristic, or enable the Food carrying checkbox and reset a baseline
world. The heuristic collects underfoot food above 75 energy when its slot is
empty and consumes stored food at 75 energy or below. This is a hand-written
demonstration; it is separate from PPO training. Random agents sample all seven
actions when carrying is enabled. Inventory appears in the inspector and agent
list; a red dot above a person means it carries one food unit.

A PPO checkpoint chooses its own carrying mode and disables the manual checkbox.
Existing Experiments 1–13 remain unchanged, and new carrying checkpoints appear
as separate Food carrying entries. See the root README for training and evaluation
commands. The existing models cannot gain this action without fresh training.

Experiment 14 is the trained carrying model, now updated to 1,000 updates: run
`python -m visualization.sim3d --checkpoint experiment_14.pt` or select Experiment 14
and reset. The model controls pickup and eating; no forced eating rule is used.

### Sample versus argmax

For PPO, choose **Sample · use percentages** or **Argmax · highest percentage**
and Reset world. The active choice appears in status and the inspector. This
only changes inference; it never trains or modifies weights. Sampling remains
the default. Exact argmax ties resolve to the first action in enum order.

```sh
python -m visualization.sim3d --checkpoint experiment_14.pt --action-selection argmax
```

The evaluation CLI supports `--action-selection sample`, `argmax`, or `both`;
`both` compares the same checkpoints on the same seeds, with baselines run once.

### Carried-food freshness / Experiment 15

```sh
python -m visualization.sim3d --checkpoint experiment_15.pt
```

Experiment 15 automatically enables 10-tick decay and a 103-value observation.
Pickup sets freshness to 10; each subsequent action can eat before that tick's
countdown. At zero, food disappears. Ground food never decays. Inventory and
remaining freshness appear in the inspector; Spoiled counts expired units,
while Wasted energy measures unused energy from actual meals. Experiments
1–14 keep their original observation layouts and non-decay rules.

For baseline policies, `--food-decay` or the decay checkbox enables this mode;
PPO checkpoints select their own lifetime. The CLI training and evaluation
commands likewise accept `--food-decay` and preserve old defaults.
