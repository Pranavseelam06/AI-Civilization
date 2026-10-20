"""Helpers for observing the local world."""

import numpy as np

from civsim.agents import AgentState


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

# TODO: Encode an egocentric 5 x 5 patch and own energy/inventory.
# TODO: Document channels, padding, shape, and normalization.
# TODO: Do not leak hidden global state to policies.
