from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from civsim.environments import SimulationEnv
from learning.networks import ActorCritic
from learning.ppo import PPOConfig, advantages_and_returns, clipped_policy_loss, ppo_update
from learning.train import Rollout, RolloutCollector, Transition


def transition(**changes):
    base = Transition(0,123,1,0,np.zeros(101,dtype=np.float32),0,0,0,0,
                      np.zeros(101,dtype=np.float32),0,False,False)
    return replace(base,**changes)


def test_advantages_do_not_cross_agents():
    rows = [transition(reward=1,value=2,next_value=3),
            transition(agent_id=1,reward=10),
            transition(tick=2,reward=4,value=3,terminated=True,next_value=999)]
    advantages,returns = advantages_and_returns(Rollout(rows,2),gamma=1,gae_lambda=1)
    torch.testing.assert_close(advantages,torch.tensor([3.,10.,1.]))
    torch.testing.assert_close(returns,torch.tensor([5.,10.,4.]))


def test_time_limit_bootstraps_without_crossing_reset():
    rows = [transition(reward=2,value=1,next_value=7,truncated=True),
            transition(episode=1,seed=124,reward=100,terminated=True)]
    advantages,returns = advantages_and_returns(Rollout(rows,2),1,1)
    assert advantages.tolist() == [8,100]
    assert returns.tolist() == [9,100]


def test_rollout_cutoff_bootstrap_and_discount():
    row = transition(reward=2,value=1,next_value=6)
    advantages,returns = advantages_and_returns(Rollout([row],1),0.5,0.9)
    assert advantages.item() == 4 and returns.item() == 5


def test_gap_does_not_join_trajectories():
    rows=[transition(reward=1),transition(tick=3,reward=100,terminated=True)]
    advantages,_=advantages_and_returns(Rollout(rows,2),1,1)
    assert advantages.tolist() == [1,100]


@pytest.mark.parametrize('ratio,advantage,expected', [
    (1.5,1,-1.2),(0.5,1,-0.5),(1.5,-1,1.5),(0.5,-1,0.8),
])
def test_clipping_for_both_advantage_signs(ratio,advantage,expected):
    loss=clipped_policy_loss(torch.tensor([ratio]).log(),torch.tensor([0.]),
                             torch.tensor([float(advantage)]),0.2)
    assert loss.item() == pytest.approx(expected)


def test_ppo_changes_weights_and_preserves_rollout_data():
    torch.manual_seed(12)
    model=ActorCritic()
    rollout=RolloutCollector(SimulationEnv(),model).collect(4)
    before={k:v.clone() for k,v in model.state_dict().items()}
    old_probs=[r.log_probability for r in rollout.transitions]
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    metrics=ppo_update(model,optimizer,rollout,PPOConfig(epochs=2,minibatch_size=16))
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert old_probs == [r.log_probability for r in rollout.transitions]
    assert all(torch.isfinite(p).all() for p in model.parameters())
