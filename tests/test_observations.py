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
