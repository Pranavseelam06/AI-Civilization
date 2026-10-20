import numpy as np
import pytest

from civsim.agents import AgentState
from civsim.actions import Action
from civsim.rewards import calculate_reward


@pytest.mark.parametrize('was_alive,is_alive,restored,expected', [
    (True,True,0,0), (True,True,25,1), (True,True,5,-0.6),
    (True,False,0,-10), (False,False,0,0),
])
def test_reward_values(was_alive,is_alive,restored,expected):
    assert calculate_reward(was_alive,is_alive,restored) == pytest.approx(expected)


def test_step_rewards_only_successful_events(world):
    world.agents = [AgentState(0,1,1), AgentState(1,3,3,energy=1),
                    AgentState(2,5,5,energy=0,alive=False), AgentState(3,7,7,energy=50)]
    world.food[:] = False
    world.food[7,7] = True
    assert world.step({3:Action.EAT}) == {0:0, 1:-10, 2:0, 3:1}
    assert world.food_consumed == 1
    assert world.step({}) == {0:0, 1:0, 2:0, 3:0}


def test_failed_eating_has_no_food_bonus(world):
    world.agents = [AgentState(0,1,1,energy=50)]
    world.food[1,1] = True
    assert world.step({}) == {0:0}
    assert world.food[1,1]


def test_eating_before_energy_drain_prevents_death(world):
    world.agents = [AgentState(0,1,1,energy=1)]
    world.food[1,1] = True
    assert world.step({0:Action.EAT}) == {0:1}
    assert world.agents[0].alive and world.agents[0].energy == 25


def test_survival_bonus_only_at_episode_end(world):
    world.config.horizon = 3
    world.food[:] = False
    assert set(world.step({}).values()) == {0}
    assert set(world.step({}).values()) == {0}
    assert set(world.step({}).values()) == {5}
    assert set(world.step({}).values()) == {0}  # Cannot award the bonus twice.


def test_food_and_survival_bonus_can_combine(world):
    world.config.horizon = 1
    world.agents = [AgentState(0,1,1,energy=50),AgentState(1,3,3,energy=1)]
    world.food[:] = False
    world.food[1,1] = True
    assert world.step({0:Action.EAT}) == {0:6,1:-10}
    assert world.step({}) == {0:0,1:0}


def test_unlimited_world_has_no_survival_bonus(world):
    world.config.horizon = None
    world.food[:] = False
    assert set(world.step({}).values()) == {0}


@pytest.mark.parametrize('energy,expected_gain,expected_reward',[
    (50,25,1.0),(75,25,1.0),(87,13,0.04),(88,12,-0.04),
    (95,5,-0.6),(99,1,-0.92),(100,0,0.0),
])
def test_food_reward_uses_actual_energy_before_tick_cost(world,energy,expected_gain,expected_reward):
    world.agents=[AgentState(0,1,1,energy=energy)]
    world.food[:]=False
    world.food[1,1]=True
    rewards=world.step({0:Action.EAT})
    assert rewards[0] == pytest.approx(expected_reward)
    assert world.agents[0].energy == energy+expected_gain-1
    assert world.food_energy_gained == expected_gain
    assert world.food_consumed == int(expected_gain>0)
    assert world.food_energy_wasted == (25-expected_gain if expected_gain else 0)


def test_partial_food_reward_combines_with_end_survival_bonus(world):
    world.config.horizon=1
    world.agents=[AgentState(0,1,1,energy=95)]
    world.food[1,1]=True
    assert world.step({0:Action.EAT})[0] == pytest.approx(4.4)
