import numpy as np
import pytest

torch = pytest.importorskip('torch')

from civsim.config import SimulationConfig
from civsim.environments import SimulationEnv
from civsim.observations import observe_agent
from learning.networks import ActorCritic
from learning.train import RolloutCollector, save_rollout


def test_rollout_shapes_and_weights_unchanged(tmp_path):
    torch.manual_seed(123)
    model = ActorCritic()
    before = {name:value.clone() for name,value in model.state_dict().items()}
    env = SimulationEnv()
    collector = RolloutCollector(env,model)
    rollout = collector.collect(3)
    assert rollout.world_ticks == 3 and len(rollout.transitions) == 36
    assert env.world.tick == 3
    for row in rollout.transitions:
        assert row.observation.shape == row.next_observation.shape == (101,)
        assert 0 <= row.action < 5
        assert np.isfinite([row.value,row.next_value,row.log_probability]).all()
        assert not row.terminated and not row.truncated
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,before[name])
    assert all(parameter.grad is None for parameter in model.parameters())
    path = tmp_path/'rollout.npz'
    save_rollout(path,rollout,env.world.config,123)
    with np.load(path,allow_pickle=False) as saved:
        assert saved['observation'].shape == (36,101)
        assert saved['action'].shape == (36,)
    rollout.transitions[0].observation[:] = -1
    assert np.all(rollout.transitions[12].observation >= 0)


def test_resets_keep_final_observation_without_bootstrapping_finite_task_end():
    model = ActorCritic()
    with torch.no_grad():
        model.critic.weight.zero_()
        model.critic.bias.fill_(7)
    env = SimulationEnv(SimulationConfig(horizon=1))
    collector = RolloutCollector(env,model)
    rows = collector.collect(2).transitions
    assert len(rows) == 24
    assert {r.seed for r in rows} == {123,124}
    assert all(r.truncated and not r.terminated and r.next_value == 0 for r in rows)
    assert all(r.tick == 1 for r in rows)
    assert rows[0].next_observation[-1] < 1  # Final state, not reset energy.
    assert rows[12].observation[-1] == 1


def test_deaths_have_no_bootstrap_and_disappear():
    env = SimulationEnv()
    collector = RolloutCollector(env,ActorCritic())
    env.world.food[:] = False
    env.world.agents[0].energy = 1
    collector.observations = {a.id:observe_agent(env.world,a) for a in env.world.agents}
    rows = collector.collect(2).transitions
    first = next(r for r in rows if r.agent_id == 0)
    assert first.terminated and first.next_value == 0 and first.reward == -10
    assert first.next_observation[-1] == 0
    assert len([r for r in rows if r.agent_id == 0]) == 1
    assert len(rows) == 23


def test_collect_calls_continue_same_world_without_false_truncation():
    collector = RolloutCollector(SimulationEnv(),ActorCritic())
    first = collector.collect(1).transitions
    second = collector.collect(1).transitions
    assert all(r.episode == 0 and r.tick == 2 for r in second)
    for previous,current in zip(first,second):
        np.testing.assert_array_equal(previous.next_observation,current.observation)
        assert not previous.truncated


def test_rollouts_repeat_with_same_torch_and_world_seeds():
    def run():
        torch.manual_seed(99)
        return RolloutCollector(SimulationEnv(),ActorCritic(),123).collect(3).transitions
    one,two = run(),run()
    assert [r.action for r in one] == [r.action for r in two]
    assert [r.reward for r in one] == [r.reward for r in two]
    for a,b in zip(one,two):
        np.testing.assert_array_equal(a.next_observation,b.next_observation)
