"""Per-agent reward from the actual outcome of a simulation tick."""


REWARD_SPEC = {'version': 2, 'survive_episode': 5, 'death': -10, 'food': 1,
               'survive_tick': 0}


def calculate_reward(was_alive: bool, is_alive: bool, ate_food: bool,
                     episode_end: bool = False) -> int:
    """Food +1, death -10, survival +5 only at the finite episode end."""
    if not was_alive:
        return 0
    survival = 5 if is_alive and episode_end else 0
    return (-10 if not is_alive else survival) + int(ate_food)
