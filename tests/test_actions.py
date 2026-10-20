import pytest
from civsim.actions import Action as A
from civsim.agents import AgentState

@pytest.mark.parametrize('action,expected', [(A.NORTH,(2,1)), (A.SOUTH,(2,3)), (A.WEST,(1,2)), (A.EAST,(3,2)), (A.STAY,(2,2))])
def test_movement(world, action, expected):
    world.agents = [AgentState(0,2,2)]
    world.food[:] = False
    world.step({0:action})
    assert (world.agents[0].x,world.agents[0].y) == expected
    assert world.tick == 1

@pytest.mark.parametrize('positions,actions,expected', [
    ([(0,0)], {0:A.NORTH}, [(0,0)]),
    ([(0,0),(2,0)], {0:A.EAST,1:A.WEST}, [(0,0),(2,0)]),
    ([(0,0),(1,0)], {0:A.EAST,1:A.WEST}, [(0,0),(1,0)]),
    ([(0,0),(1,0)], {0:A.EAST,1:A.EAST}, [(0,0),(2,0)]),
    ([(0,0)], {0:999}, [(0,0)]),
    ([(0,0)], {}, [(0,0)]),
])
def test_conflicts_and_invalid_moves(world, positions, actions, expected):
    world.agents = [AgentState(i,x,y) for i,(x,y) in enumerate(positions)]
    world.step(actions)
    assert [(a.x,a.y) for a in world.agents] == expected

def test_obstacle_blocks_movement(world):
    world.agents = [AgentState(0,0,0)]
    world.grid[0,1] = 1
    world.step({0:A.EAST})
    assert world.agents[0].x == 0

def test_processing_order_does_not_change_collision_result(world):
    world.agents = [AgentState(0,0,0),AgentState(1,2,0)]
    world.agents.reverse()
    world.step({0:A.EAST,1:A.WEST})
    assert {a.id:(a.x,a.y) for a in world.agents} == {0:(0,0),1:(2,0)}
