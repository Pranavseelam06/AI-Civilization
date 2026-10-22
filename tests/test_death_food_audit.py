"""Food visibility and optimistic energy budgets must match actual tick rules."""

import numpy as np

from civsim.actions import Action
from civsim.config import SimulationConfig
from civsim.world import World
from experiments.death_food_audit import snapshot


def test_food_outside_view_is_not_counted_as_visible_and_budget_includes_eat():
    world = World(SimulationConfig(width=10, height=10, num_agents=1))
    world.reset(1)
    agent = world.agents[0]
    agent.x = agent.y = 4
    agent.energy = 3
    world.food[:] = False
    world.food[4, 7] = True  # Three moves away, outside 5x5 view.
    before = world.food.copy()
    row = snapshot(world, agent, Action.EAST, 'moved', np.argwhere(world.food))
    assert row['visible_food'] == 0
    assert row['nearest_global_distance'] == 3
    assert not row['any_food_within_energy_budget']  # Dies on third move before eating.
    agent.energy = 4
    row = snapshot(world, agent, Action.EAST, 'moved', np.argwhere(world.food))
    assert row['any_food_within_energy_budget']
    assert not row['visible_within_energy_budget']
    assert np.array_equal(before, world.food)
    assert world.tick == 0


def test_underfoot_food_is_still_edible_at_one_energy_and_empty_map_is_distinct():
    world = World(SimulationConfig(num_agents=1))
    world.reset(1)
    agent = world.agents[0]
    agent.energy = 1
    world.food[:] = False
    empty = snapshot(world, agent, Action.STAY, 'stay', np.argwhere(world.food))
    assert empty['nearest_global_distance'] == -1
    assert not empty['any_food_within_energy_budget']
    world.food[agent.y, agent.x] = True
    row = snapshot(world, agent, Action.EAT, 'eat', np.argwhere(world.food))
    assert row['visible_food'] == row['food_underfoot'] == 1
    assert row['nearest_visible_distance'] == 0
    assert row['visible_within_energy_budget']
