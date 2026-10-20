import numpy as np
import pytest

from civsim.agents import AgentState
from civsim.actions import Action
from civsim.rewards import calculate_reward


@pytest.mark.parametrize('was_alive,is_alive,restored,before,expected', [
    (True,True,0,50,0), (True,True,25,20,0.6), (True,True,5,95,-0.9),
    (True,False,0,1,-10), (False,False,0,0,0),
])
def test_reward_values(was_alive,is_alive,restored,before,expected):
    assert calculate_reward(was_alive,is_alive,restored,energy_before_eating=before) == pytest.approx(expected)


def test_step_rewards_only_successful_events(world):
    world.agents = [AgentState(0,1,1), AgentState(1,3,3,energy=1),
                    AgentState(2,5,5,energy=0,alive=False), AgentState(3,7,7,energy=50)]
    world.food[:] = False
    world.food[7,7] = True
    assert world.step({3:Action.EAT}) == {0:0, 1:-10, 2:0, 3:0}
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
    assert world.step({0:Action.EAT})[0] == pytest.approx(0.98)
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
    assert world.step({0:Action.EAT}) == {0:5,1:-10}
    assert world.step({}) == {0:0,1:0}


def test_unlimited_world_has_no_survival_bonus(world):
    world.config.horizon = None
    world.food[:] = False
    assert set(world.step({}).values()) == {0}


@pytest.mark.parametrize('energy,expected_gain,expected_reward',[
    (20,25,0.6),(50,25,0.0),(75,25,-0.5),(87,13,-0.74),(88,12,-0.76),
    (95,5,-0.9),(99,1,-0.98),(100,0,0.0),
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
    assert world.step({0:Action.EAT})[0] == pytest.approx(4.1)


def test_consumption_requires_pre_eating_energy_for_reward():
    with pytest.raises(ValueError,match='energy_before_eating'):
        calculate_reward(True,True,25)


def test_signed_reward_decreases_with_energy():
    rewards=[calculate_reward(True,True,min(25,100-energy),energy_before_eating=energy)
             for energy in range(1,100)]
    assert rewards[0] > 0 and rewards[-1] < 0
    assert all(a>b for a,b in zip(rewards,rewards[1:]))


@pytest.mark.parametrize('energy,penalty',[(30,0),(31,0),(29,0),(20,0),(1,0)])
def test_no_reward_penalty_for_skipping_available_meal(world,energy,penalty):
    world.agents=[AgentState(0,1,1,energy=energy)]
    world.food[:]=False
    world.food[1,1]=True
    reward=world.step({})[0]
    assert reward == pytest.approx(penalty + (-10 if energy==1 else 0))
    assert world.missed_meals == int(energy<30)


def test_no_penalty_for_food_that_requires_movement_to_reach(world):
    world.agents=[AgentState(0,0,1,energy=20)]
    world.food[:]=False
    world.food[1,1]=True
    assert world.step({0:Action.EAST})[0] == 0
    assert world.missed_meals == 0
    assert world.step({0:Action.EAT})[0] == pytest.approx(0.62)


def test_skipped_meals_remain_diagnostic_only(world):
    world.agents=[AgentState(0,1,1,energy=20)]
    world.food[:]=False
    world.food[1,1]=True
    assert world.step({0:Action.EAST})[0] == 0
    assert world.missed_meals == 1 and world.food[1,1]
    world.reset(123)
    assert world.missed_meals == 0


def test_no_missed_meal_penalty_if_no_food_is_available(world):
    world.agents=[AgentState(0,1,1,energy=20)]
    world.food[:]=False
    assert world.step({})[0] == 0
    assert world.step({0:Action.EAT})[0] == 0
    assert world.missed_meals == 0
