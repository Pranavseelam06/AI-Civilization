"""Per-agent reward from the actual outcome of a simulation tick."""


def calculate_reward(was_alive: bool, is_alive: bool, ate_food: bool) -> int:
    """Survival +5, death -10, food +1; agents already dead receive zero."""
    if not was_alive:
        return 0
    return (5 if is_alive else -10) + int(ate_food)
