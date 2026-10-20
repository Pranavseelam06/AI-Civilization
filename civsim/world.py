"""World state and seeded initialization."""

from collections import Counter

from civsim.actions import Action, MOVEMENT_DELTAS
from civsim.config import SimulationConfig
from civsim.terrain import create_grid, spawn_agents


class World:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.grid = create_grid(config.width, config.height)
        self.agents = []
        self.tick = 0

    def reset(self, seed: int) -> None:
        """Create a fresh grid and population; the same seed repeats the positions."""
        self.grid = create_grid(self.config.width, self.config.height)
        self.agents = spawn_agents(self.grid, self.config.num_agents, seed)
        self.tick = 0

    def step(self, actions: dict[int, Action]) -> None:
        """Apply simultaneous moves. Missing or invalid actions become STAY.

        Cells occupied at the start of the tick cannot be entered, even if
        their occupant moves away. Contenders for the same empty cell all stay.
        """
        occupied = {(agent.x, agent.y) for agent in self.agents}
        destinations = {}
        height, width = self.grid.shape

        for agent in self.agents:
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
            destination = destinations.get(agent.id)
            if destination is not None and counts[destination] == 1:
                agent.x, agent.y = destination
        self.tick += 1


# TODO: Add food, energy, and an owned random generator when needed.
# TODO: Update resources/energy and handle death.
# TODO: Return useful transition results without rendering or policy decisions.
