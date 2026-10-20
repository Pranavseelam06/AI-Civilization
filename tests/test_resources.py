import numpy as np
import pytest
from civsim.agents import AgentState

class FixedDraw:
    def __init__(self,value): self.value = value
    def random(self,shape=None):
        return self.value if shape is None else np.full(shape,self.value)

@pytest.mark.parametrize('draw,energy,eaten', [(0.1,74,True),(0.9,49,False)])
def test_eating_coin_flip(world, draw, energy, eaten):
    world.agents = [AgentState(0,1,1,energy=50)]
    world.food[:] = False
    world.food[1,1] = True
    world.eating_rng = FixedDraw(draw)
    world.step({})
    assert world.agents[0].energy == energy
    assert bool(world.food[1,1]) == (not eaten)
    assert world.food_consumed == int(eaten)

def test_eating_caps_before_tick_cost(world):
    world.agents = [AgentState(0,1,1,energy=99)]
    world.food[1,1] = True
    world.eating_rng = FixedDraw(0)
    world.step({})
    assert world.agents[0].energy == 99

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

def test_regrowth_does_not_advance_eating_rng(world):
    expected = np.random.default_rng()
    expected.bit_generator.state = world.eating_rng.bit_generator.state
    world.food[:] = False
    for _ in range(20): world.step({})
    assert world.eating_rng.random() == expected.random()
