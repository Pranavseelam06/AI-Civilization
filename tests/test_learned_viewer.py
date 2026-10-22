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


def test_old_five_action_checkpoints_use_legacy_viewer_mode(checkpoints):
    from learning.inference import LearnedPolicy
    legacy=ActorCritic(action_count=5)
    path=checkpoints/'old.pt'
    torch.save({'model_state_dict':legacy.state_dict()},path)
    assert path.exists() and 'old.pt' in sim3d.checkpoint_names()
    session = sim3d.SimulationSession()
    assert session.reset(123, 'ppo', 'old.pt')['eating_mode'] == 'automatic'
    first = [session.step() for _ in range(5)]
    session.reset(123, 'ppo', 'old.pt')
    assert first == [session.step() for _ in range(5)]
    assert session.learned.model.actor.out_features == 5
    session.reset(123, 'heuristic')
    assert not session.world.automatic_eating
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
    assert len(entries) == 14
    assert entries[11]['name'] == 'experiment_12_best.pt'
    assert entries[12]['name'] == 'experiment_13_best.pt'
    assert entries[8]['name'] == 'archive/experiment_9.pt'
    assert entries[8]['available']
    assert not entries[0]['available']
    session = sim3d.SimulationSession()
    assert session.reset(123, 'ppo', entries[8]['name'])['checkpoint'] == entries[8]['name']


def test_legacy_automatic_eating_can_consume_at_full_energy():
    world = World(SimulationConfig(width=1, height=1, num_agents=1), automatic_eating=True)
    world.reset(123)
    world.food[:] = True
    world.eating_rng = np.random.default_rng(2)  # First draw < 0.5.
    world.step({0: Action.STAY})
    assert world.food_consumed == 1
    assert world.food_energy_wasted == 25
    assert world.agents[0].energy == 99


def test_inspection_probabilities_are_read_only_and_actions_are_from_previous_tick(checkpoints):
    torch.save({'model_state_dict': ActorCritic().state_dict()}, checkpoints / 'sample.pt')
    inspected, control = sim3d.SimulationSession(), sim3d.SimulationSession()
    initial = inspected.reset(42, 'ppo', 'sample.pt')
    control.reset(42, 'ppo', 'sample.pt')
    assert all(a['last_action'] is None for a in initial['agents'])
    assert all(sum(a['next_action_probabilities'].values()) == pytest.approx(1)
               for a in initial['agents'])
    for _ in range(10):
        # Repeated UI reads must not consume sampling RNG or advance the world.
        before = inspected.state()
        assert inspected.state() == before
        assert inspected.step() == control.step()
    assert all(a['last_action'] in Action.__members__ for a in inspected.state()['agents'])
    assert all(a['last_outcome'] for a in inspected.state()['agents'])
    reset = inspected.reset(42, 'ppo', 'sample.pt')
    assert all(a['last_action'] is None for a in reset['agents'])


def test_inspection_baselines_and_dead_agents_have_no_model_probabilities(checkpoints):
    session = sim3d.SimulationSession()
    assert all(not a['next_action_probabilities'] for a in session.state()['agents'])
    session.reset(123, 'ppo', 'east.pt')
    session.world.agents[0].alive = False
    session.world.agents[0].energy = 0
    assert session.state()['agents'][0]['next_action_probabilities'] == {}
    session.world.tick = session.world.config.horizon
    final = session.state()
    assert all(not a['next_action_probabilities'] for a in final['agents'])
    assert session.step() == final


def test_legacy_inspection_has_five_probabilities(checkpoints):
    torch.save({'model_state_dict': ActorCritic(action_count=5).state_dict()}, checkpoints / 'old.pt')
    session = sim3d.SimulationSession()
    state = session.reset(123, 'ppo', 'old.pt')
    assert state['eating_mode'] == 'automatic'
    for agent in state['agents']:
        assert len(agent['next_action_probabilities']) == 5
        assert 'EAT' not in agent['next_action_probabilities']
