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

The experiment menu contains one entry for each Experiment 1–13. Experiments
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
