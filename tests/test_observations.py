import numpy as np
from civsim.agents import AgentState
from civsim.observations import nearest_food

def test_nearest_food_is_local_and_uses_manhattan_distance():
    food = np.zeros((8,8),dtype=bool)
    agent = AgentState(0,3,3)
    assert nearest_food(food,agent) is None
    food[3,6] = True
    assert nearest_food(food,agent) is None
    food[4,5] = True
    assert nearest_food(food,agent) == (2,1,3)
    food[3,3] = True
    assert nearest_food(food,agent) == (0,0,0)

def test_edges_and_tie_breaking():
    food = np.zeros((8,8),dtype=bool)
    food[0,1] = food[1,0] = True
    assert nearest_food(food,AgentState(0,0,0)) == (1,0,1)


from civsim.observations import OBSERVATION_SIZE, observe_agent
from civsim.world import World
from civsim.config import SimulationConfig


def test_observation_channels_and_energy(world):
    agent = AgentState(0, 4, 4, energy=50)
    world.agents = [agent, AgentState(1,5,4), AgentState(2,4,5,energy=0,alive=False)]
    world.food[:] = False
    world.food[3,4] = True
    world.grid[4,3] = 1
    observation = observe_agent(world,agent)
    assert observation.shape == (OBSERVATION_SIZE,) == (101,)
    assert observation.dtype == np.float32
    assert np.all((observation >= 0) & (observation <= 1))
    patch = observation[:100].reshape(4,5,5)
    assert patch[0,1,2] == 1  # Food north.
    assert patch[1,2,1] == 1  # Obstacle west.
    assert not patch[2].any()  # No boundaries in the interior.
    assert patch[3,2,3] == 1  # Living neighbor east.
    assert patch[3].sum() == 1  # Neither self nor dead agent is included.
    assert observation[-1] == 0.5


def test_observation_corner_padding_does_not_wrap():
    world = World(SimulationConfig(width=7,height=3,num_agents=1))
    world.reset(123)
    agent = AgentState(0,0,0)
    world.agents = [agent]
    world.food[:] = False
    world.food[2,6] = True  # Must not wrap into the negative-index padding.
    patch = observe_agent(world,agent)[:100].reshape(4,5,5)
    assert patch[2].sum() == 16  # Only the bottom-right 3 x 3 area is in bounds.
    assert not patch[[0,1,3]].any()
    assert patch[2,2,2] == 0
    agent.x,agent.y = 6,2
    patch = observe_agent(world,agent)[:100].reshape(4,5,5)
    assert patch[2].sum() == 16
    assert patch[0,2,2] == 1


def test_distant_state_and_global_counters_do_not_leak(world):
    agent = AgentState(0,5,5)
    world.agents = [agent,AgentState(1,20,20)]
    before = observe_agent(world,agent)
    world.food[20,20] = not world.food[20,20]
    world.grid[20,20] = 1
    world.agents[1].x = 25
    world.tick = 99
    world.food_consumed = 1000
    np.testing.assert_array_equal(observe_agent(world,agent),before)


def test_observation_is_an_independent_snapshot(world):
    agent = world.agents[0]
    before = observe_agent(world,agent)
    saved = before.copy()
    world.food[:] = ~world.food
    agent.energy = 10
    np.testing.assert_array_equal(before,saved)
    assert observe_agent(world,agent)[-1] == np.float32(0.1)


def test_observation_uses_relative_positions_only():
    world = World(SimulationConfig())
    world.reset(123)
    world.food[:] = False
    agent = AgentState(0,5,5,energy=75)
    world.agents = [agent,AgentState(1,6,5)]
    world.food[4,5] = True
    first = observe_agent(world,agent)
    world.food[:] = False
    world.food[9,10] = True
    agent.x,agent.y = 10,10
    world.agents[1].x,world.agents[1].y = 11,10
    np.testing.assert_array_equal(observe_agent(world,agent),first)
