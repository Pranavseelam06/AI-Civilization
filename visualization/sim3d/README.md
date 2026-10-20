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
