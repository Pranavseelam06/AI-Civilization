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

# TODO: Add inventory when needed.
# TODO: Keep learned policies separate from mutable simulation state.
