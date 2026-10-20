"""Simple baseline policies; these rules are written by us, not learned."""

import numpy as np

from civsim.actions import Action


def random_policy(rng: np.random.Generator) -> Action:
    """Choose one of the five movement actions using the supplied generator."""
    return Action(int(rng.integers(len(Action))))


def heuristic_policy(
    energy: int,
    visible_food: tuple[int, int, int] | None,
    rng: np.random.Generator,
) -> Action:
    """Below 75 energy, approach visible food; otherwise explore randomly.

    visible_food is (dx, dy, distance), supplied by the local observation helper.
    Stay on food to allow the world's temporary eating coin flip to run.
    This simple policy does not plan around obstacles or other agents.
    """
    if energy < 75 and visible_food is not None:
        dx, dy, _ = visible_food
        if dx > 0:
            return Action.EAST
        if dx < 0:
            return Action.WEST
        if dy > 0:
            return Action.SOUTH
        if dy < 0:
            return Action.NORTH
        return Action.STAY
    return random_policy(rng)
