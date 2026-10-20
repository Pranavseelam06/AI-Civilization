import pytest
from experiments.run_baselines import run_episode
from civsim.config import SimulationConfig
from civsim.observations import nearest_food
from learning.baselines import heuristic_policy
import numpy as np

@pytest.mark.parametrize('seed',range(5))
def test_500_tick_invariants(world,seed):
    world.reset(seed)
    rng=np.random.default_rng(seed+10000)
    for _ in range(500):
        world.step({a.id:heuristic_policy(a.energy,nearest_food(world.food,a),rng) for a in world.agents if a.alive})
        living=[a for a in world.agents if a.alive]
        assert len({(a.x,a.y) for a in living}) == len(living)
        assert all(0 <= a.x < 32 and 0 <= a.y < 32 for a in world.agents)
        assert all(0 <= a.energy <= 100 and a.alive == (a.energy > 0) for a in world.agents)
    assert world.tick == 500

@pytest.mark.parametrize('policy',['random','heuristic'])
def test_evaluation_is_repeatable_and_horizon_capped(policy):
    config=SimulationConfig(horizon=1)
    result=run_episode(policy,123,config)
    assert result == run_episode(policy,123,config)
    assert result['episode_ticks'] == result['mean_survival_ticks'] == 1
    assert result['survivors'] == 12
