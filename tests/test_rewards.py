import numpy as np
import pytest

from civsim.agents import AgentState
from civsim.rewards import calculate_reward


@pytest.mark.parametrize('was_alive,is_alive,ate,expected', [
    (True,True,False,5), (True,True,True,6),
    (True,False,False,-10), (False,False,False,0),
])
def test_reward_values(was_alive,is_alive,ate,expected):
    assert calculate_reward(was_alive,is_alive,ate) == expected


def test_step_rewards_only_successful_events(world):
    world.agents = [AgentState(0,1,1), AgentState(1,3,3,energy=1),
                    AgentState(2,5,5,energy=0,alive=False), AgentState(3,7,7,energy=50)]
    world.food[:] = False
    world.food[7,7] = True
    world.eating_rng = np.random.default_rng(2)  # First draw succeeds.
    assert world.step({}) == {0:5, 1:-10, 2:0, 3:6}
    assert world.food_consumed == 1
    assert world.step({}) == {0:5, 1:0, 2:0, 3:5}


def test_failed_eating_has_no_food_bonus(world):
    world.agents = [AgentState(0,1,1,energy=50)]
    world.food[1,1] = True
    world.eating_rng = np.random.default_rng(0)  # First draw fails.
    assert world.step({}) == {0:5}
    assert world.food[1,1]


def test_eating_before_energy_drain_prevents_death(world):
    world.agents = [AgentState(0,1,1,energy=1)]
    world.food[1,1] = True
    world.eating_rng = np.random.default_rng(2)
    assert world.step({}) == {0:6}
    assert world.agents[0].alive and world.agents[0].energy == 25
