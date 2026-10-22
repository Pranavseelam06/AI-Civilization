"""World state and seeded initialization."""

from collections import Counter

import numpy as np

from civsim.actions import Action, MOVEMENT_DELTAS
from civsim.config import SimulationConfig
from civsim.rewards import calculate_reward
from civsim.terrain import create_grid, spawn_agents


class World:
    def __init__(self, config: SimulationConfig, *, automatic_eating=False):
        if automatic_eating and config.food_carrying:
            raise ValueError("Automatic eating cannot be combined with food carrying")
        self.automatic_eating = automatic_eating
        self.config = config
        self.grid = create_grid(config.width, config.height)
        self.food = np.zeros_like(self.grid, dtype=bool)
        self.regrowth_rng = np.random.default_rng(1)
        self.food_consumed = 0
        self.food_picked_up = 0
        self.food_spoiled = 0
        self.food_energy_gained = 0
        self.food_energy_wasted = 0
        self.missed_meals = 0
        self.agents = []
        self.tick = 0

    def reset(self, seed: int) -> None:
        """Create a fresh grid and population; the same seed repeats the positions."""
        self.grid = create_grid(self.config.width, self.config.height)
        self.agents = spawn_agents(self.grid, self.config.num_agents, seed)
        # Reserve the former eating stream so placement/regrowth seeds stay stable.
        food_seed, eating_seed, regrowth_seed = np.random.SeedSequence(seed).spawn(3)
        food_rng = np.random.default_rng(food_seed)
        self.eating_rng = np.random.default_rng(eating_seed)
        self.regrowth_rng = np.random.default_rng(regrowth_seed)
        self.food_consumed = 0
        self.food_picked_up = 0
        self.food_spoiled = 0
        self.food_energy_gained = 0
        self.food_energy_wasted = 0
        self.missed_meals = 0
        # Start with food on about 10% of the land; at most one unit per cell.
        self.food = (food_rng.random(self.grid.shape) < 0.1) & (self.grid == 0)
        self.tick = 0

    def step(self, actions: dict[int, Action]) -> dict[int, float]:
        """Apply simultaneous moves. Missing or invalid actions become STAY.

        Cells occupied at the start of the tick cannot be entered, even if
        their occupant moves away. Contenders for the same empty cell all stay.
        Dead agents do not occupy cells or block movement.
        EAT stays in place and consumes food if energy is below 100.
        With food_carrying enabled, PICKUP stores one underfoot food (no reward
        or energy gain); EAT uses stored food first, otherwise underfoot food.
        PICKUP and EAT each cost one tick. Carried food is lost on death.
        Optional decay counts subsequent ticks only; EAT at freshness 1 is valid
        before end-of-tick expiry, which removes the food without reward.
        Eating restores 25 energy (capped at 100), then each living agent loses
        one energy, including when staying. Uneaten food remains for later ticks.
        Return rewards by agent ID: death -10, signed hunger-based eating rewards,
        and a one-time +5
        for agents alive at the configured horizon. Ordinary survival earns 0.
        With horizon=None there is no end-survival bonus.
        Already-dead agents receive zero.
        """
        # Normally expiry is processed at the previous tick's end; also reject
        # stale inventory in explicitly constructed/imported states.
        if self.config.carried_food_decay_ticks is not None:
            for agent in self.agents:
                if agent.alive and agent.carried_food and agent.food_ticks_left <= 0:
                    agent.carried_food = 0
                    agent.food_ticks_left = 0
                    self.food_spoiled += 1
        occupied = {(agent.x, agent.y) for agent in self.agents if agent.alive}
        destinations = {}
        chosen_actions = {}
        food_available_before_action = {
            a.id: bool(self.food[a.y, a.x]) for a in self.agents if a.alive
        }
        height, width = self.grid.shape

        for agent in self.agents:
            if not agent.alive:
                continue
            try:
                action = Action(actions.get(agent.id, Action.STAY))
            except (ValueError, TypeError):
                action = Action.STAY
            if action == Action.PICKUP and not self.config.food_carrying:
                action = Action.STAY
            chosen_actions[agent.id] = action
            if action in (Action.EAT, Action.PICKUP):
                continue
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
            gained = 0
            picked_up = False
            energy_before_eating = agent.energy
            if (energy_before_eating < 30 and food_available_before_action[agent.id]
                    and chosen_actions[agent.id] != Action.EAT):
                self.missed_meals += 1  # Diagnostic only; never a reward penalty.
            if (self.config.food_carrying and chosen_actions[agent.id] == Action.PICKUP
                    and agent.carried_food == 0 and self.food[agent.y, agent.x]):
                self.food[agent.y, agent.x] = False
                agent.carried_food = 1
                agent.food_ticks_left = self.config.carried_food_decay_ticks or 0
                picked_up = True
                self.food_picked_up += 1
            from_inventory = self.config.food_carrying and agent.carried_food == 1
            should_eat = (self.eating_rng.random() < 0.5 if self.automatic_eating
                          and self.food[agent.y, agent.x] else False)
            if ((from_inventory or self.food[agent.y, agent.x]) and (should_eat if self.automatic_eating
                    else chosen_actions[agent.id] == Action.EAT and agent.energy < 100)):
                if from_inventory:
                    agent.carried_food = 0
                    agent.food_ticks_left = 0
                else:
                    self.food[agent.y, agent.x] = False
                self.food_consumed += 1
                gained = min(25, 100 - agent.energy)
                agent.energy += gained
                self.food_energy_gained += gained
                self.food_energy_wasted += 25 - gained
            agent.energy = max(0, agent.energy - 1)
            if agent.energy == 0:
                agent.alive = False
                agent.carried_food = 0  # Carried food is lost on death; no drops yet.
                agent.food_ticks_left = 0
            elif (self.config.carried_food_decay_ticks is not None
                    and agent.carried_food and not picked_up):
                # Ten subsequent action opportunities; do not age on pickup tick.
                agent.food_ticks_left -= 1
                if agent.food_ticks_left == 0:
                    agent.carried_food = 0
                    self.food_spoiled += 1
            episode_end = (self.config.horizon is not None
                           and self.tick + 1 == self.config.horizon)
            rewards[agent.id] = calculate_reward(
                True, agent.alive, gained, episode_end,
                energy_before_eating=energy_before_eating,
            )
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


# TODO: Return useful transition results without rendering or policy decisions.
