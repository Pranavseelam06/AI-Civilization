"""World state and seeded initialization."""

from collections import Counter

import numpy as np

from civsim.actions import Action, MOVEMENT_DELTAS
from civsim.config import SimulationConfig
from civsim.rewards import calculate_reward
from civsim.terrain import create_grid, spawn_agents


class World:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.grid = create_grid(config.width, config.height)
        self.food = np.zeros_like(self.grid, dtype=bool)
        self.eating_rng = np.random.default_rng(0)
        self.regrowth_rng = np.random.default_rng(1)
        self.food_consumed = 0
        self.agents = []
        self.tick = 0

    def reset(self, seed: int) -> None:
        """Create a fresh grid and population; the same seed repeats the positions."""
        self.grid = create_grid(self.config.width, self.config.height)
        self.agents = spawn_agents(self.grid, self.config.num_agents, seed)
        # Independent streams keep eating decisions separate from food placement.
        food_seed, eating_seed, regrowth_seed = np.random.SeedSequence(seed).spawn(3)
        food_rng = np.random.default_rng(food_seed)
        self.eating_rng = np.random.default_rng(eating_seed)
        self.regrowth_rng = np.random.default_rng(regrowth_seed)
        self.food_consumed = 0
        # Start with food on about 10% of the land; at most one unit per cell.
        self.food = (food_rng.random(self.grid.shape) < 0.1) & (self.grid == 0)
        self.tick = 0

    def step(self, actions: dict[int, Action]) -> dict[int, int]:
        """Apply simultaneous moves. Missing or invalid actions become STAY.

        Cells occupied at the start of the tick cannot be entered, even if
        their occupant moves away. Contenders for the same empty cell all stay.
        Dead agents do not occupy cells or block movement.
        After movement, living agents on food have a seeded 50% chance to eat.
        Eating restores 25 energy (capped at 100), then each living agent loses
        one energy, including when staying. Uneaten food remains for later ticks.
        Return rewards by agent ID: death -10, eating +1, and a one-time +5
        for agents alive at the configured horizon. Ordinary survival earns 0.
        With horizon=None there is no end-survival bonus.
        Already-dead agents receive zero.
        """
        occupied = {(agent.x, agent.y) for agent in self.agents if agent.alive}
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
        rewards = {agent.id: 0 for agent in self.agents}
        for agent in self.agents:
            if not agent.alive:
                continue
            destination = destinations.get(agent.id)
            if destination is not None and counts[destination] == 1:
                agent.x, agent.y = destination
            ate_food = False
            if self.food[agent.y, agent.x] and self.eating_rng.random() < 0.5:
                ate_food = True
                self.food[agent.y, agent.x] = False
                self.food_consumed += 1
                agent.energy = min(100, agent.energy + 25)
            agent.energy = max(0, agent.energy - 1)
            if agent.energy == 0:
                agent.alive = False
            episode_end = (self.config.horizon is not None
                           and self.tick + 1 == self.config.horizon)
            rewards[agent.id] = calculate_reward(True, agent.alive, ate_food, episode_end)
        self.tick += 1
        if self.tick % 20 == 0:
            # Empty means no food and no living agent on walkable land.
            available = (self.grid == 0) & ~self.food
            for agent in self.agents:
                if agent.alive:
                    available[agent.y, agent.x] = False
            grows = self.regrowth_rng.random(self.grid.shape) < 0.01
            self.food |= available & grows
        return rewards


# TODO: Replace the temporary eating coin flip with policy decisions later.
# TODO: Return useful transition results without rendering or policy decisions.
