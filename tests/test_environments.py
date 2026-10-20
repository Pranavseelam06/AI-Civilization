import numpy as np
import pytest

from civsim.config import SimulationConfig
from civsim.environments import SimulationEnv


def test_reset_and_step_observation_contract():
    env = SimulationEnv()
    with pytest.raises(RuntimeError):
        env.step({})
    initial = env.reset(123)
    assert len(initial) == 12 and set(initial) == set(env.agents)
    assert all(obs.shape == (101,) and obs.dtype == np.float32 for obs in initial.values())
    env.world.food[:] = False
    result = env.step({})
    assert not result.done and result.info['tick'] == 1
    assert set(result.observations) == set(result.rewards) == set(initial)
    assert set(result.terminated) == set(result.truncated) == set(initial)
    assert set(result.rewards.values()) == {0}
    assert not any(result.terminated.values()) and not any(result.truncated.values())


def test_death_is_reported_once_then_agent_is_excluded():
    env = SimulationEnv()
    env.reset(123)
    env.world.food[:] = False
    env.world.agents[0].energy = 1
    result = env.step({})
    assert result.terminated[0] and not result.truncated[0]
    assert result.rewards[0] == -10 and result.observations[0][-1] == 0
    assert 0 not in env.agents and not result.done
    with pytest.raises(ValueError):
        env.step({0:0})
    following = env.step({})
    for mapping in (following.observations,following.rewards,following.terminated,following.truncated):
        assert 0 not in mapping


def test_horizon_preserves_final_observations_and_requires_reset():
    env = SimulationEnv(SimulationConfig(horizon=1))
    initial = env.reset(123)
    result = env.step({})
    assert result.done and env.agents == []
    assert all(result.truncated.values()) and not any(result.terminated.values())
    assert len(result.observations) == 12
    assert all(agent.alive for agent in env.world.agents)
    with pytest.raises(RuntimeError):
        env.step({})
    reset = env.reset(123)
    for key in initial:
        np.testing.assert_array_equal(initial[key],reset[key])


def test_death_on_horizon_is_not_truncation():
    env = SimulationEnv(SimulationConfig(horizon=1))
    env.reset(123)
    env.world.food[:] = False
    env.world.agents[0].energy = 1
    result = env.step({})
    assert result.terminated[0] and not result.truncated[0]
    assert result.truncated[1] and not result.terminated[1]


def test_unlimited_horizon_still_ends_when_everyone_dies():
    env = SimulationEnv(SimulationConfig(horizon=None))
    env.reset(123)
    env.world.food[:] = False
    for agent in env.world.agents:
        agent.energy = 1
    result = env.step({})
    assert result.done and all(result.terminated.values())
    assert not any(result.truncated.values()) and env.agents == []


def test_unknown_actions_do_not_advance_world():
    env = SimulationEnv()
    env.reset(123)
    with pytest.raises(ValueError):
        env.step({999:0})
    assert env.world.tick == 0
