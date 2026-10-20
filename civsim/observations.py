"""Helpers for observing the local world."""

import numpy as np
from typing import TYPE_CHECKING
from civsim.actions import Action, MOVEMENT_DELTAS

from civsim.agents import AgentState

if TYPE_CHECKING:
    from civsim.world import World


OBSERVATION_SIZE = 101  # Four 5 x 5 maps, followed by one energy value.


def movement_mask(observation: np.ndarray) -> np.ndarray:
    """Legal destinations from local observation only; STAY is always allowed.

    Matches pre-step occupancy rules. Contested empty destinations cannot be
    predicted from one observation and are still resolved by World.step().
    """
    patch = observation[:100].reshape(4, 5, 5)
    mask = np.ones(len(Action), dtype=bool)
    for action, (dx, dy) in MOVEMENT_DELTAS.items():
        if action != Action.STAY:
            mask[action] = not patch[1:4, 2+dy, 2+dx].any()
    return mask


def observe_agent(world: "World", agent: AgentState) -> np.ndarray:
    """Return a fresh float32 vector containing only local information.

    Entries 0:100 are four flattened 5 x 5 maps, in this order: food,
    obstacles (nonzero terrain), outside-world boundaries, other living agents.
    Each map is row-major, with the observer at [2, 2]. North is up; the
    view does not rotate. Outside-world cells have boundary=1 and all other
    channels=0. The observer is excluded from the other-agent channel.
    Entry 100 is own energy / 100, clipped to [0, 1]. No IDs, absolute
    coordinates, global counters, or distant state are included.
    """
    patch = np.zeros((4, 5, 5), dtype=np.float32)
    height, width = world.grid.shape
    neighbors = {
        (other.x, other.y)
        for other in world.agents
        if other.alive and other.id != agent.id
        and abs(other.x - agent.x) <= 2 and abs(other.y - agent.y) <= 2
    }
    for row, dy in enumerate(range(-2, 3)):
        for column, dx in enumerate(range(-2, 3)):
            x, y = agent.x + dx, agent.y + dy
            if not (0 <= x < width and 0 <= y < height):
                patch[2, row, column] = 1
                continue
            patch[0, row, column] = world.food[y, x]
            patch[1, row, column] = world.grid[y, x] != 0
            patch[3, row, column] = (x, y) in neighbors
    energy = np.array([np.clip(agent.energy / 100, 0, 1)], dtype=np.float32)
    return np.concatenate((patch.ravel(), energy))


def nearest_food(
    food: np.ndarray, agent: AgentState
) -> tuple[int, int, int] | None:
    """Return (dx, dy, Manhattan distance) for nearest food in a 5 x 5 view.

    Return None when none is visible. Ties favor the first cell in row order
    (top to bottom, then left to right). Distance ignores obstacles and agents.
    """
    height, width = food.shape
    left, right = max(0, agent.x - 2), min(width, agent.x + 3)
    top, bottom = max(0, agent.y - 2), min(height, agent.y + 3)
    nearest = None
    for local_y, local_x in np.argwhere(food[top:bottom, left:right]):
        dx = int(local_x) + left - agent.x
        dy = int(local_y) + top - agent.y
        distance = abs(dx) + abs(dy)
        if nearest is None or distance < nearest[2]:
            nearest = (dx, dy, distance)
    return nearest

# TODO: Add own inventory when inventory is implemented.
