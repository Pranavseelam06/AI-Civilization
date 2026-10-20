import numpy as np
import pytest
from civsim.agents import AgentState
from civsim.actions import Action

class FixedDraw:
    def __init__(self,value): self.value = value
    def random(self,shape=None):
        return self.value if shape is None else np.full(shape,self.value)

def test_eating_requires_explicit_action(world):
    world.agents = [AgentState(0,1,1,energy=50)]
    world.food[:] = False
    world.food[1,1] = True
    world.step({})
    assert world.agents[0].energy == 49 and world.food[1,1]
    world.step({0:Action.EAT})
    assert world.agents[0].energy == 73 and not world.food[1,1]
    assert world.food_consumed == 1
    assert world.food_energy_gained == 25 and world.food_energy_wasted == 0


def test_eating_caps_before_tick_cost_and_tracks_waste(world):
    world.agents = [AgentState(0,1,1,energy=95)]
    world.food[1,1] = True
    world.step({0:Action.EAT})
    assert world.agents[0].energy == 99
    assert world.food_energy_gained == 5 and world.food_energy_wasted == 20
    world.reset(123)
    assert world.food_energy_gained == world.food_energy_wasted == 0


def test_eating_at_full_energy_does_not_consume_food(world):
    world.agents = [AgentState(0,1,1)]
    world.food[1,1] = True
    world.step({0:Action.EAT})
    assert world.agents[0].energy == 99 and world.food[1,1]
    assert world.food_consumed == 0


def test_empty_eat_is_noop_except_tick_cost(world):
    world.agents = [AgentState(0,1,1,energy=50)]
    world.food[:] = False
    assert world.step({0:Action.EAT}) == {0:0}
    assert world.agents[0].energy == 49


def test_moving_onto_food_does_not_automatically_eat(world):
    world.agents = [AgentState(0,0,1,energy=50)]
    world.food[:] = False
    world.food[1,1] = True
    world.step({0:Action.EAST})
    assert world.agents[0].x == 1 and world.agents[0].energy == 49
    assert world.food[1,1] and world.food_consumed == 0


def test_regrowth_timing_and_eligible_cells(world):
    world.food[:] = False
    world.agents = [AgentState(0,1,1),AgentState(1,2,2,energy=0,alive=False)]
    world.grid[0,0] = 1
    world.regrowth_rng = FixedDraw(0)
    for _ in range(19): world.step({})
    assert not world.food.any()
    world.step({})
    assert not world.food[0,0]
    assert not world.food[1,1]
    assert world.food[2,2]  # Dead agents no longer occupy cells.
    assert world.food[3,3]
    assert world.food.dtype == bool



def test_dead_agent_cannot_eat(world):
    world.agents=[AgentState(0,1,1,energy=0,alive=False)]
    world.food[1,1]=True
    assert world.step({0:Action.EAT}) == {0:0}
    assert world.food[1,1] and world.food_consumed == 0
