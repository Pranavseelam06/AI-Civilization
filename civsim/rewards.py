"""Per-agent reward from the actual outcome of a simulation tick."""


REWARD_SPEC = {'version': 5, 'survive_episode': 5, 'death': -10,
               'food': '1 - 2 * (energy_before_eating / 100) when food is consumed',
               'survive_tick': 0}


def calculate_reward(was_alive: bool, is_alive: bool, energy_restored: int,
                     episode_end: bool = False, *,
                     energy_before_eating: int | None = None) -> float:
    """Reward eating based on hunger before consumption; no consumption earns zero.

    Consumption always restores positive energy under the current world rules.
    Death stays -10; end-of-episode survival stays +5.
    """
    if not was_alive:
        return 0
    survival = 5 if is_alive and episode_end else 0
    food_reward = 0.0
    if energy_restored > 0:
        if energy_before_eating is None:
            raise ValueError('Eating reward requires energy_before_eating')
        food_reward = 1 - 2 * (energy_before_eating / 100)
    return (-10 if not is_alive else survival) + food_reward
