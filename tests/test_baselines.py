import numpy as np
import pytest
from learning.baselines import heuristic_policy, random_policy
from civsim.actions import Action as A

@pytest.mark.parametrize('food,action', [((1,1,2),A.EAST),((-1,0,1),A.WEST),((0,1,1),A.SOUTH),((0,-1,1),A.NORTH),((0,0,0),A.EAT)])
def test_hungry_heuristic(food,action):
    assert heuristic_policy(50,food,np.random.default_rng(1)) == action

def test_exploration_and_seeded_randomness():
    one, two = np.random.default_rng(42),np.random.default_rng(42)
    assert [random_policy(one) for _ in range(50)] == [random_policy(two) for _ in range(50)]
    for energy,food in [(76,(0,0,0)),(10,None)]:
        assert heuristic_policy(energy,food,np.random.default_rng(42)) == A(int(np.random.default_rng(42).integers(5)))
