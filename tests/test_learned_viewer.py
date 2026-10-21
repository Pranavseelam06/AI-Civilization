import numpy as np
import pytest

torch = pytest.importorskip('torch')

from learning.networks import ActorCritic
from visualization import sim3d
from civsim.world import World
from civsim.config import SimulationConfig
from civsim.actions import Action


@pytest.fixture
def checkpoints(tmp_path,monkeypatch):
    monkeypatch.setattr(sim3d,'CHECKPOINTS',tmp_path)
    model=ActorCritic()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.actor.bias.fill_(-100)
        model.actor.bias[Action.EAST]=100
    torch.save({'model_state_dict':model.state_dict()},tmp_path/'east.pt')
    return tmp_path


def test_trained_viewer_uses_loaded_actions_without_updating_weights(checkpoints):
    session=sim3d.SimulationSession()
    state=session.reset(123,'ppo','east.pt')
    assert state['checkpoint']=='east.pt'
    before={k:v.clone() for k,v in session.learned.model.state_dict().items()}
    expected=World(SimulationConfig());expected.reset(123)
    for _ in range(5):
        session.step()
        expected.step({a.id:Action.EAST for a in expected.agents if a.alive})
    assert session.world.agents==expected.agents
    assert np.array_equal(session.world.food,expected.food)
    for key,value in session.learned.model.state_dict().items():
        torch.testing.assert_close(value,before[key])
    assert all(p.grad is None for p in session.learned.model.parameters())
    assert not session.learned.model.training


def test_ppo_reset_is_repeatable_and_randomness_is_session_local(checkpoints):
    torch.save({'model_state_dict':ActorCritic().state_dict()},checkpoints/'sample.pt')
    a,b=sim3d.SimulationSession(),sim3d.SimulationSession()
    a.reset(42,'ppo','sample.pt');b.reset(42,'ppo','sample.pt')
    for _ in range(10):
        assert a.step()==b.step()
    a.reset(42,'ppo','sample.pt')
    b.reset(42,'ppo','sample.pt')
    assert a.step()==b.step()


def test_invalid_checkpoint_preserves_current_session(checkpoints):
    session=sim3d.SimulationSession()
    before=session.state()
    for name in ('../east.pt','missing.pt'):
        with pytest.raises(ValueError):
            session.reset(123,'ppo',name)
        assert session.state()==before
    (checkpoints/'bad.pt').write_bytes(b'invalid model')
    with pytest.raises(ValueError):
        session.reset(123,'ppo','bad.pt')
    assert session.state()==before


def test_switching_back_to_heuristic_discards_learned_policy(checkpoints):
    session=sim3d.SimulationSession()
    session.reset(123,'ppo','east.pt')
    state=session.reset(123,'heuristic')
    assert session.learned is None and state['checkpoint'] is None


def test_old_five_action_checkpoints_are_preserved_but_not_selected(checkpoints):
    from learning.inference import LearnedPolicy
    legacy=ActorCritic(action_count=5)
    path=checkpoints/'old.pt'
    torch.save({'model_state_dict':legacy.state_dict()},path)
    assert path.exists() and 'old.pt' not in sim3d.checkpoint_names()
    with pytest.raises(ValueError,match='old five-action'):
        LearnedPolicy(path,SimulationConfig(),123)


def test_experiment_catalog_hides_extra_runs_and_uses_best_and_archive(checkpoints):
    import shutil
    (checkpoints / 'archive').mkdir()
    for name in ('experiment_5.pt', 'experiment_12.pt', 'experiment_12_best.pt',
                 'experiment_13_best.pt', 'experiment_13_seed1001.pt'):
        shutil.copyfile(checkpoints / 'east.pt', checkpoints / name)
    shutil.copyfile(checkpoints / 'east.pt', checkpoints / 'archive' / 'experiment_9.pt')
    entries = sim3d.experiment_catalog()
    assert len(entries) == 13
    assert entries[11]['name'] == 'experiment_12_best.pt'
    assert entries[12]['name'] == 'experiment_13_best.pt'
    assert entries[8]['name'] == 'archive/experiment_9.pt'
    assert entries[8]['available']
    assert not entries[0]['available']
    session = sim3d.SimulationSession()
    assert session.reset(123, 'ppo', entries[8]['name'])['checkpoint'] == entries[8]['name']
