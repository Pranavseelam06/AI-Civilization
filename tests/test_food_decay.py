"""Carried-food lifetime boundaries and versioned observation/checkpoint rules."""

import numpy as np
import pytest

from civsim.actions import Action as A
from civsim.agents import AgentState
from civsim.config import SimulationConfig
from civsim.observations import observe_agent, movement_mask
from civsim.world import World


@pytest.fixture
def decay_world():
    w = World(SimulationConfig(width=5, height=5, num_agents=1,
        food_carrying=True, carried_food_decay_ticks=10))
    w.reset(123)
    w.food[:] = False
    w.agents = [AgentState(0, 2, 2, energy=60)]
    w.food[2, 2] = True
    return w


def test_ten_subsequent_ticks_then_expiry_no_energy_or_reward(decay_world):
    w = decay_world; a = w.agents[0]
    assert w.step({0: A.PICKUP}) == {0: 0}
    assert a.food_ticks_left == 10 and w.food_spoiled == 0
    for expected in range(9, 0, -1):
        assert w.step({0: A.STAY}) == {0: 0}
        assert a.food_ticks_left == expected and a.carried_food == 1
    assert movement_mask(observe_agent(w, a))[A.EAT]
    assert w.step({0: A.STAY}) == {0: 0}
    assert a.food_ticks_left == a.carried_food == 0 and w.food_spoiled == 1
    assert w.food_consumed == 0 and a.energy == 49
    assert not movement_mask(observe_agent(w, a))[A.EAT]
    w.step({0: A.EAT})
    assert a.energy == 48 and w.food_consumed == 0 and w.food_spoiled == 1


def test_eating_is_valid_on_last_fresh_tick(decay_world):
    w = decay_world; a = w.agents[0]
    w.step({0: A.PICKUP})
    for _ in range(9):
        w.step({0: A.STAY})
    assert a.food_ticks_left == 1
    w.step({0: A.EAT})
    assert a.energy == 74 and a.carried_food == a.food_ticks_left == 0
    assert w.food_consumed == 1 and w.food_spoiled == 0


def test_ground_food_does_not_age_and_full_slot_cannot_refresh_timer(decay_world):
    w = decay_world; a = w.agents[0]
    for _ in range(11):
        w.step({0: A.STAY})
    assert w.food[2, 2] and w.food_spoiled == 0
    w.step({0: A.PICKUP})
    w.food[2, 2] = True
    w.step({0: A.PICKUP})
    assert a.food_ticks_left == 9 and w.food[2, 2] and w.food_picked_up == 1


def test_expired_slot_can_be_refilled_and_reset_clears_freshness(decay_world):
    w = decay_world; a = w.agents[0]
    w.step({0: A.PICKUP})
    for _ in range(10):
        w.step({0: A.STAY})
    w.food[2, 2] = True
    assert movement_mask(observe_agent(w, a))[A.PICKUP]
    w.step({0: A.PICKUP})
    assert a.food_ticks_left == 10
    w.reset(123)
    assert all(a.food_ticks_left == a.carried_food == 0 for a in w.agents)
    assert w.food_spoiled == 0


def test_freshness_observation_and_stale_inventory_cannot_feed(decay_world):
    w = decay_world; a = w.agents[0]
    w.step({0: A.PICKUP})
    observation = observe_agent(w, a)
    assert observation.shape == (103,) and observation[102] == 1
    w.step({0: A.STAY})
    assert observe_agent(w, a)[102] == np.float32(.9)
    a.food_ticks_left = 0  # Explicitly constructed stale state.
    assert not movement_mask(observe_agent(w, a))[A.EAT]
    before = a.energy
    w.step({0: A.EAT})
    assert a.energy == before-1 and w.food_consumed == 0 and w.food_spoiled == 1


def test_death_loses_inventory_without_counting_spoilage(decay_world):
    w = decay_world; a = w.agents[0]
    a.energy = 1
    w.step({0: A.PICKUP})
    assert not a.alive and a.carried_food == a.food_ticks_left == 0
    assert w.food_spoiled == 0


def test_nonperishable_food_and_old_observations_unchanged():
    w = World(SimulationConfig(food_carrying=True, num_agents=1)); w.reset(123)
    a = w.agents[0]; w.food[a.y, a.x] = True
    w.step({a.id: A.PICKUP})
    for _ in range(20):
        w.step({})
    assert a.carried_food == 1 and w.food_spoiled == 0
    assert observe_agent(w, a).shape == (102,)
    for ttl in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            SimulationConfig(food_carrying=True, carried_food_decay_ticks=ttl)
    with pytest.raises(ValueError):
        SimulationConfig(carried_food_decay_ticks=10)


def test_decay_checkpoint_and_viewer_select_matching_rules(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from learning.networks import ActorCritic
    from learning.inference import LearnedPolicy, compatible_checkpoint
    from visualization import sim3d
    model = ActorCritic(observation_size=103, action_count=7)
    path = tmp_path/'experiment_15.pt'
    torch.save({'model_state_dict':model.state_dict(),'metadata':{'action_masking':True,
        'environment':{'carried_food_decay_ticks':10}}},path)
    assert compatible_checkpoint(path)
    with pytest.raises(ValueError, match='decay'):
        LearnedPolicy(path, SimulationConfig(food_carrying=True), 123)
    monkeypatch.setattr(sim3d, 'CHECKPOINTS', tmp_path)
    s = sim3d.SimulationSession()
    state = s.reset(123,'ppo',path.name)
    assert state['config']['carried_food_decay_ticks'] == 10
    s.step()
    assert s.learned.observation_size == 103
    assert any(e['name']==path.name and e['label']=='Experiment 15' for e in sim3d.experiment_catalog())
    state = s.reset(123,'heuristic',food_carrying=True)
    assert state['config']['carried_food_decay_ticks'] is None


def test_spoilage_evaluation_is_not_reported_as_death_loss(monkeypatch):
    pytest.importorskip('torch')
    from learning import evaluate
    class FoodAtSpawn(World):
        def reset(self, seed):
            super().reset(seed)
            for a in self.agents:
                self.food[a.y, a.x] = True
    class PickThenWait:
        action_masking = True
        def __init__(self, *args):
            pass
        def actions(self, world):
            return {a.id: A.PICKUP if world.tick == 0 else A.STAY for a in world.agents if a.alive}
        def probabilities(self, world):
            return {a.id:[1/7]*7 for a in world.agents if a.alive}
    monkeypatch.setattr(evaluate, 'World', FoodAtSpawn)
    monkeypatch.setattr(evaluate, 'LearnedPolicy', PickThenWait)
    row, _ = evaluate.evaluate_episode('ppo',123,SimulationConfig(horizon=11,
        food_carrying=True,carried_food_decay_ticks=10),'fake.pt')
    assert row['food_picked_up'] == row['food_spoiled'] == 12
    assert row['stored_food_consumed'] == row['carried_food_remaining'] == row['carried_food_lost'] == 0
    assert row['food_energy_gained'] == row['food_energy_wasted'] == 0


def test_replay_preserves_freshness_and_spoilage(decay_world, tmp_path):
    from civsim.replay import ReplayRecorder, load_replay
    w = decay_world
    record = ReplayRecorder(tmp_path/'decay.jsonl', w, 123, 456)
    w.step({0:A.PICKUP}); record.record(w)
    for _ in range(10):
        w.step({0:A.STAY})
    record.record(w); record.finish(w)
    frames, summary = load_replay(tmp_path/'decay.jsonl')
    assert frames[1].agents[0].food_ticks_left == 10
    assert frames[-1].agents[0].food_ticks_left == 0
    assert frames[-1].food_spoiled == summary['food_spoiled'] == 1
    assert frames[-1].config.carried_food_decay_ticks == 10
