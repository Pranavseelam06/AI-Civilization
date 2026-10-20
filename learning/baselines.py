"""Simple baseline policies; these rules are written by us, not learned."""

import numpy as np

from civsim.actions import Action, MOVEMENT_DELTAS


def random_policy(rng: np.random.Generator) -> Action:
    """Choose uniformly among movement, STAY, and EAT."""
    return Action(int(rng.integers(len(Action))))


def heuristic_policy(
    energy: int,
    visible_food: tuple[int, int, int] | None,
    rng: np.random.Generator,
) -> Action:
    """At 75 energy or below, approach visible food and explicitly EAT.

    visible_food is (dx, dy, distance), supplied by the local observation helper.
    Eating at 75 or below avoids wasting any of the 25-energy food unit.
    This simple policy does not plan around obstacles or other agents.
    """
    if energy <= 75 and visible_food is not None:
        dx, dy, _ = visible_food
        if dx > 0:
            return Action.EAST
        if dx < 0:
            return Action.WEST
        if dy > 0:
            return Action.SOUTH
        if dy < 0:
            return Action.NORTH
        return Action.EAT
    return Action(int(rng.integers(len(MOVEMENT_DELTAS))))
