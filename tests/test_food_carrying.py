"""One-slot storage, consumption, observations, and checkpoint compatibility."""

import numpy as np
import pytest

from civsim.actions import Action as A
from civsim.agents import AgentState
from civsim.config import SimulationConfig
from civsim.observations import observe_agent, movement_mask
from civsim.world import World
from experiments.diagnostics import movement_outcomes


@pytest.fixture
def carrying_world():
    world = World(SimulationConfig(width=5, height=5, num_agents=1, food_carrying=True))
    world.reset(123)
    world.food[:] = False
    world.agents = [AgentState(0, 2, 2, energy=60)]
    return world


def test_pickup_then_move_and_eat_without_ground_food(carrying_world):
    w = carrying_world
    a = w.agents[0]
    w.food[2, 2] = True
    assert movement_outcomes(w, {0: A.PICKUP})[0] == 'pickup'
    assert w.step({0: A.PICKUP}) == {0: 0}  # No food reward until consumption.
    assert (a.x, a.y, a.energy, a.carried_food) == (2, 2, 59, 1)
    assert not w.food[2, 2] and w.food_picked_up == 1
    assert w.food_consumed == w.food_energy_gained == 0
    w.step({0: A.EAST})
    assert (a.x, a.carried_food) == (3, 1)
    assert movement_outcomes(w, {0: A.EAT})[0] == 'eat'
    reward = w.step({0: A.EAT})[0]
    assert reward == pytest.approx(1 - 2 * .58)
    assert (a.energy, a.carried_food) == (82, 0)
    assert w.food_consumed == 1 and w.food_energy_gained == 25


def test_full_slot_cannot_take_second_food_and_inventory_is_eaten_first(carrying_world):
    w = carrying_world; a = w.agents[0]
    a.carried_food = 1
    w.food[2, 2] = True
    w.step({0: A.PICKUP})
    assert w.food[2, 2] and a.carried_food == 1 and w.food_picked_up == 0
    assert a.energy == 59
    w.step({0: A.EAT})
    assert w.food[2, 2] and a.carried_food == 0  # Ground food remains.
    w.step({0: A.EAT})
    assert not w.food[2, 2] and w.food_consumed == 2


def test_no_food_pickup_and_full_energy_eat_are_noops_except_tick_cost(carrying_world):
    w = carrying_world; a = w.agents[0]
    w.step({0: A.PICKUP})
    assert (a.carried_food, a.energy) == (0, 59)
    a.carried_food = 1; a.energy = 100
    assert not movement_mask(observe_agent(w, a))[A.EAT]
    w.step({0: A.EAT})
    assert (a.carried_food, a.energy, w.food_consumed) == (1, 99, 0)


def test_pickup_does_not_save_agent_at_one_energy_but_eating_does(carrying_world):
    w = carrying_world; a = w.agents[0]
    a.energy = 1; w.food[2, 2] = True
    assert w.step({0: A.PICKUP}) == {0: -10}
    assert not a.alive and a.carried_food == 0 and w.food_consumed == 0
    assert not w.food[2, 2]  # Lost on death; no dropping mechanic.
    before = w.food_picked_up
    w.step({0: A.PICKUP})
    assert w.food_picked_up == before
    w.reset(123)
    assert all(a.carried_food == 0 for a in w.agents) and w.food_picked_up == 0
    a = w.agents[0]; a.energy = 1; a.carried_food = 1
    w.step({0: A.EAT})
    assert a.alive and a.energy == 25 and a.carried_food == 0


def test_observation_inventory_and_mask_are_local(carrying_world):
    w = carrying_world; a = w.agents[0]
    observation = observe_agent(w, a)
    assert observation.shape == (102,) and observation[100] == np.float32(.6)
    assert observation[101] == 0 and len(movement_mask(observation)) == 7
    assert not movement_mask(observation)[A.PICKUP]
    w.food[2, 2] = True
    assert movement_mask(observe_agent(w, a))[A.PICKUP]
    a.carried_food = 1; w.food[2, 2] = False
    mask = movement_mask(observe_agent(w, a))
    assert mask[A.EAT] and not mask[A.PICKUP]
    assert observe_agent(w, a)[101] == 1
    for action in A:
        outcome = movement_outcomes(w, {a.id: action})[a.id]
        assert bool(mask[action]) == (outcome in ('stay', 'moved', 'eat', 'pickup'))


def test_carrying_disabled_keeps_original_observation_and_pickup_does_nothing():
    w = World(SimulationConfig(num_agents=1)); w.reset(123)
    a = w.agents[0]; w.food[a.y, a.x] = True
    w.step({a.id: A.PICKUP})
    assert a.carried_food == 0 and w.food[a.y, a.x] and a.energy == 99
    assert observe_agent(w, a).shape == (101,)
    with pytest.raises(ValueError):
        World(SimulationConfig(food_carrying=True), automatic_eating=True)


def test_carrying_rollout_ppo_and_checkpoint_viewer(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from civsim.environments import SimulationEnv
    from learning.networks import ActorCritic
    from learning.train import RolloutCollector, save_rollout
    from learning.ppo import ppo_update, PPOConfig
    from learning.inference import LearnedPolicy, compatible_checkpoint
    from visualization import sim3d
    config = SimulationConfig(food_carrying=True)
    model = ActorCritic(observation_size=102, action_count=7)
    env = SimulationEnv(config)
    rollout = RolloutCollector(env, model).collect(4)
    assert all(r.observation.shape == (102,) and len(r.action_mask) == 7 for r in rollout.transitions)
    metrics = ppo_update(model, torch.optim.Adam(model.parameters()), rollout, PPOConfig(epochs=1))
    assert np.isfinite(metrics['value_loss'])
    path = tmp_path / 'carrying.pt'
    torch.save({'model_state_dict': model.state_dict(), 'metadata': {'action_masking': True}}, path)
    assert compatible_checkpoint(path)
    with pytest.raises(ValueError, match='mode'):
        LearnedPolicy(path, SimulationConfig(), 123)
    monkeypatch.setattr(sim3d, 'CHECKPOINTS', tmp_path)
    session = sim3d.SimulationSession()
    state = session.reset(123, 'ppo', path.name)
    assert state['config']['food_carrying']
    assert 'PICKUP' in state['agents'][0]['next_action_probabilities']
    assert any(e['name'] == path.name for e in sim3d.experiment_catalog())
    session.step()
    save_rollout(tmp_path / 'rollout.npz', rollout, config, 123)
    with np.load(tmp_path / 'rollout.npz') as saved:
        import json
        metadata = json.loads(str(saved['metadata']))
        assert metadata['actions'][-1] == 'PICKUP' and metadata['config']['food_carrying']
    torch.save({'model_state_dict': ActorCritic().state_dict()}, tmp_path / 'old.pt')
    state = session.reset(123, 'ppo', 'old.pt', food_carrying=True)
    assert not state['config']['food_carrying']  # Existing experiment picks its original task.


def test_inventory_replay_round_trip(carrying_world, tmp_path):
    from civsim.replay import ReplayRecorder, load_replay
    w = carrying_world
    w.food[2, 2] = True
    recorder = ReplayRecorder(tmp_path / 'replay.jsonl', w, 123, 456)
    w.step({0: A.PICKUP}); recorder.record(w); recorder.finish(w)
    frames, _ = load_replay(tmp_path / 'replay.jsonl')
    assert frames[-1].agents[0].carried_food == 1
    assert frames[-1].food_picked_up == 1 and frames[-1].config.food_carrying


def test_evaluation_distinguishes_stored_meals_from_pickup_and_loss(monkeypatch):
    pytest.importorskip('torch')
    from learning import evaluate
    class FoodAtSpawn(World):
        def reset(self, seed):
            super().reset(seed)
            for agent in self.agents:
                self.food[agent.y, agent.x] = True
    class CollectThenEat:
        action_masking = True
        def __init__(self, *args):
            pass
        def actions(self, world):
            return {a.id: A.PICKUP if world.tick == 0 else A.EAT for a in world.agents if a.alive}
        def probabilities(self, world):
            return {a.id: [1/7]*7 for a in world.agents if a.alive}
    monkeypatch.setattr(evaluate, 'World', FoodAtSpawn)
    monkeypatch.setattr(evaluate, 'LearnedPolicy', CollectThenEat)
    row, _ = evaluate.evaluate_episode('ppo', 123, SimulationConfig(horizon=2, food_carrying=True), 'fake.pt')
    assert row['food_picked_up'] == row['stored_food_consumed'] == row['food_consumed'] == 12
    assert row['carrying_agent_ticks'] == 12
    assert row['carried_food_remaining'] == row['carried_food_lost'] == 0
    row, _ = evaluate.evaluate_episode('ppo', 123, SimulationConfig(horizon=1, food_carrying=True), 'fake.pt')
    assert row['food_picked_up'] == row['carried_food_remaining'] == 12
    assert row['stored_food_consumed'] == row['carried_food_lost'] == 0
