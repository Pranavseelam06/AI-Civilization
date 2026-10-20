"""Basic agent identity and grid position."""

from dataclasses import dataclass


@dataclass
class AgentState:
    id: int
    x: int
    y: int


# TODO: Add energy, inventory, and alive status when needed.
# TODO: Keep learned policies separate from mutable simulation state.
