import numpy as np
import pytest

torch=pytest.importorskip('torch')

from civsim.agents import AgentState
from civsim.actions import Action
from experiments.compare_exploration import exploration_actions


def test_only_foodless_stay_is_replaced_with_legal_movement(world):
    world.agents=[AgentState(0,0,0)]
    world.food[:]=False
    world.grid[1,0]=1  # North/west outside, south blocked: east is the only move.
    actions={0:Action.STAY}
    result,stats=exploration_actions(world,actions,np.random.default_rng(1))
    assert result=={0:Action.EAST} and actions=={0:Action.STAY}
    assert stats['replacements']==1
    world.food[0,1]=True
    result,stats=exploration_actions(world,actions,np.random.default_rng(1))
    assert result==actions and stats['replacements']==0


def test_boxed_agent_stays_and_existing_movement_is_unchanged(world):
    world.agents=[AgentState(0,0,0)]
    world.food[:]=False
    world.grid[1,0]=world.grid[0,1]=1
    result,stats=exploration_actions(world,{0:Action.STAY},np.random.default_rng(1))
    assert result=={0:Action.STAY} and stats['boxed_stays']==1
    result,stats=exploration_actions(world,{0:Action.EAST},np.random.default_rng(1))
    assert result=={0:Action.EAST} and stats['replacements']==0


def test_disabled_intervention_does_not_draw_randomness(world):
    world.agents=[AgentState(0,3,3)]
    world.food[:]=False
    one,two=np.random.default_rng(1),np.random.default_rng(1)
    result,stats=exploration_actions(world,{0:Action.STAY},one,enabled=False)
    assert result=={0:Action.STAY} and stats['foodless_stays']==1
    assert one.random()==two.random()
