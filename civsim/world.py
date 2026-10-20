"""World state and seeded initialization."""

from collections import Counter

import numpy as np

from civsim.actions import Action, MOVEMENT_DELTAS
from civsim.config import SimulationConfig
from civsim.terrain import create_grid, spawn_agents


class World:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.grid = create_grid(config.width, config.height)
        self.food = np.zeros_like(self.grid, dtype=bool)
        self.eating_rng = np.random.default_rng(0)
        self.agents = []
        self.tick = 0

    def reset(self, seed: int) -> None:
        """Create a fresh grid and population; the same seed repeats the positions."""
        self.grid = create_grid(self.config.width, self.config.height)
        self.agents = spawn_agents(self.grid, self.config.num_agents, seed)
        # Independent streams keep eating decisions separate from food placement.
        food_seed, eating_seed = np.random.SeedSequence(seed).spawn(2)
        food_rng = np.random.default_rng(food_seed)
        self.eating_rng = np.random.default_rng(eating_seed)
        # Start with food on about 10% of the land; at most one unit per cell.
        self.food = (food_rng.random(self.grid.shape) < 0.1) & (self.grid == 0)
        self.tick = 0

    def step(self, actions: dict[int, Action]) -> None:
        """Apply simultaneous moves. Missing or invalid actions become STAY.

        Cells occupied at the start of the tick cannot be entered, even if
        their occupant moves away. Contenders for the same empty cell all stay.
        Dead agents remain on their cells and block movement for now.
        After movement, living agents on food have a seeded 50% chance to eat.
        Eating restores 25 energy (capped at 100), then each living agent loses
        one energy, including when staying. Uneaten food remains for later ticks.
        """
        occupied = {(agent.x, agent.y) for agent in self.agents}
        destinations = {}
        height, width = self.grid.shape

        for agent in self.agents:
            if not agent.alive:
                continue
            try:
                action = Action(actions.get(agent.id, Action.STAY))
            except (ValueError, TypeError):
                action = Action.STAY
            dx, dy = MOVEMENT_DELTAS[action]
            x, y = agent.x + dx, agent.y + dy
            if (
                0 <= x < width
                and 0 <= y < height
                and self.grid[y, x] == 0
                and (x, y) not in occupied
            ):
                destinations[agent.id] = (x, y)

        counts = Counter(destinations.values())
        for agent in self.agents:
            if not agent.alive:
                continue
            destination = destinations.get(agent.id)
            if destination is not None and counts[destination] == 1:
                agent.x, agent.y = destination
            if self.food[agent.y, agent.x] and self.eating_rng.random() < 0.5:
                self.food[agent.y, agent.x] = False
                agent.energy = min(100, agent.energy + 25)
            agent.energy = max(0, agent.energy - 1)
            if agent.energy == 0:
                agent.alive = False
        self.tick += 1


# TODO: Replace the temporary eating coin flip with policy decisions later.
# TODO: Add food renewal when needed.
# TODO: Return useful transition results without rendering or policy decisions.
