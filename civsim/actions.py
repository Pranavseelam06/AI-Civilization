"""Movement actions and their (dx, dy) coordinate changes."""

from enum import IntEnum


class Action(IntEnum):
    STAY = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


MOVEMENT_DELTAS = {
    Action.STAY: (0, 0),
    Action.NORTH: (0, -1),
    Action.SOUTH: (0, 1),
    Action.EAST: (1, 0),
    Action.WEST: (-1, 0),
}


# TODO: Later add harvest, eat, and share.
# Movement validation and conflict resolution live in World.step().
