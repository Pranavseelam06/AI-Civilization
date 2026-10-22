"""Starting simulation settings."""

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    width: int = 32
    height: int = 32
    num_agents: int = 12
    horizon: int | None = 500  # None means no tick limit.
    food_carrying: bool = False  # Opt in; preserves the original experiments.
    carried_food_decay_ticks: int | None = None  # None preserves nonperishable food.

    def __post_init__(self):
        ttl = self.carried_food_decay_ticks
        if ttl is not None and (type(ttl) is not int or ttl <= 0 or not self.food_carrying):
            raise ValueError('Carried-food decay requires carrying and a positive integer lifetime')


# TODO: Validate configuration values and add other settings when needed.
# TODO: Load configs/forage.yaml without creating world state.
