"""The 3D bridge must preserve the existing Python simulation behavior."""

import pytest
from visualization.sim3d import SimulationSession
from experiments.run_baselines import run_episode
from civsim.config import SimulationConfig


@pytest.mark.parametrize('policy', ['random', 'heuristic'])
def test_3d_session_matches_baseline(policy):
    session = SimulationSession()
    session.reset(123, policy)
    for _ in range(500):
        session.step()
    state = session.state()
    baseline = run_episode(policy, 123, SimulationConfig())
    assert state['finished']
    assert sum(a['alive'] for a in state['agents']) == baseline['survivors']
    assert state['food_consumed'] == baseline['food_consumed']
    assert state['tick'] == baseline['episode_ticks']
    assert session.step() == state  # Ended episodes cannot advance.


def test_session_reset_restores_snapshot():
    session = SimulationSession()
    start = session.state()
    for _ in range(30):
        session.step()
    assert start['tick'] == 0
    assert session.reset(123, 'heuristic') == start


@pytest.mark.parametrize('seed,policy', [(-1,'random'),(1.5,'random'),(True,'random'),(0,'missing')])
def test_invalid_session_reset_is_rejected(seed, policy):
    session = SimulationSession()
    before = session.state()
    with pytest.raises(ValueError):
        session.reset(seed,policy)
    assert session.state() == before
