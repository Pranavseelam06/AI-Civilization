"""Starting simulation settings."""

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    width: int = 32
    height: int = 32
    num_agents: int = 12
    horizon: int | None = 500  # None means no tick limit.


# TODO: Validate configuration values and add other settings when needed.
# TODO: Load configs/forage.yaml without creating world state.
