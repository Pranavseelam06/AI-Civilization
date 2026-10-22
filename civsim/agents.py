"""Basic agent identity and grid position."""

from dataclasses import dataclass


@dataclass
class AgentState:
    id: int
    x: int
    y: int
    z: int = 0
    energy: int = 100
    alive: bool = True
    carried_food: int = 0  # One slot: 0 or 1.
    food_ticks_left: int = 0  # Used only when carried-food decay is enabled.

# TODO: Keep learned policies separate from mutable simulation state.
