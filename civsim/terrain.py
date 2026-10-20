"""Create the simulation grid."""

import numpy as np

from civsim.agents import AgentState


def create_grid(width: int, height: int) -> np.ndarray:
    """Return empty land: 0 means empty, and cells are accessed as grid[y, x]."""
    return np.zeros((height, width), dtype=np.int8)


def spawn_agents(
    grid: np.ndarray, num_agents: int, seed: int
) -> list[AgentState]:
    """Choose distinct empty cells, reproducibly for a given grid and seed."""
    if grid.ndim != 2:
        raise ValueError("Grid must be two-dimensional.")
    empty_cells = np.argwhere(grid == 0)
    if num_agents < 0 or num_agents > len(empty_cells):
        raise ValueError("Agent count must be between zero and the empty-cell count.")

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(empty_cells), size=num_agents, replace=False)
    agents = []
    for agent_id, cell_index in enumerate(selected):
        y, x = empty_cells[cell_index]
        agents.append(AgentState(id=agent_id, x=int(x), y=int(y)))
    return agents


# TODO: Add obstacles and food when needed.
