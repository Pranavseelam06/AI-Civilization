import numpy as np
import pytest

from civsim.actions import Action as A
from civsim.agents import AgentState
from civsim.observations import observe_agent, movement_mask
from experiments.diagnostics import movement_outcomes


def test_mask_uses_local_boundary_obstacle_and_living_occupancy(world):
    agent=AgentState(0,0,0)
    world.agents=[agent,AgentState(1,1,0)]
    world.grid[1,0]=1
    mask=movement_mask(observe_agent(world,agent))
    np.testing.assert_array_equal(mask,[True,False,False,False,False,False])
    world.agents[1].alive=False
    mask=movement_mask(observe_agent(world,agent))
    assert mask[A.EAST]  # Bodies no longer occupy cells.


def test_mask_matches_world_destination_rules(world):
    for agent in world.agents:
        mask=movement_mask(observe_agent(world,agent))
        assert mask[A.STAY]
        for action in list(A)[:len(mask)]:
            outcome=movement_outcomes(world,{agent.id:action})[agent.id]
            assert bool(mask[action]) == (outcome in ('stay','moved','eat'))


def test_mask_does_not_predict_simultaneous_contention(world):
    world.agents=[AgentState(0,0,0),AgentState(1,2,0)]
    assert movement_mask(observe_agent(world,world.agents[0]))[A.EAST]
    assert movement_mask(observe_agent(world,world.agents[1]))[A.WEST]
    assert set(movement_outcomes(world,{0:A.EAST,1:A.WEST}).values()) == {'contested'}


def test_masked_distribution_never_samples_illegal_moves_and_has_finite_gradients():
    torch=pytest.importorskip('torch')
    from learning.networks import mask_action_logits
    logits=torch.tensor([[1.,100.,100.,100.,100.]],requires_grad=True)
    distribution=torch.distributions.Categorical(logits=mask_action_logits(logits,[[True,False,False,False,False]]))
    assert distribution.probs.tolist()==[[1,0,0,0,0]]
    assert distribution.sample((100,)).sum()==0
    loss=-distribution.log_prob(torch.tensor([0])).mean()-distribution.entropy().mean()
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_masked_ppo_reuses_sampling_mask_for_identical_log_probs():
    torch=pytest.importorskip('torch')
    from civsim.environments import SimulationEnv
    from learning.networks import ActorCritic
    from learning.train import RolloutCollector
    from learning.ppo import ppo_update,PPOConfig
    torch.manual_seed(123)
    model=ActorCritic()
    rollout=RolloutCollector(SimulationEnv(),model).collect(8)
    assert all(r.action_mask[r.action] for r in rollout.transitions)
    optimizer=torch.optim.Adam(model.parameters(),lr=0)
    stats=ppo_update(model,optimizer,rollout,PPOConfig(epochs=1,minibatch_size=256))
    assert stats['approx_kl']==pytest.approx(0,abs=1e-7)
    assert stats['clip_fraction']==0


def test_inference_masks_only_checkpoints_trained_with_masks(world,tmp_path):
    torch=pytest.importorskip('torch')
    from learning.networks import ActorCritic
    from learning.inference import LearnedPolicy
    world.agents=[AgentState(0,0,0)]
    model=ActorCritic()
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.actor.bias[A.NORTH]=100
    legacy=tmp_path/'legacy.pt';masked=tmp_path/'masked.pt'
    torch.save({'model_state_dict':model.state_dict()},legacy)
    torch.save({'model_state_dict':model.state_dict(),'metadata':{'action_masking':True}},masked)
    old,new=LearnedPolicy(legacy,world.config,123),LearnedPolicy(masked,world.config,123)
    assert old.probabilities(world)[0][A.NORTH]>0.99
    assert new.probabilities(world)[0][A.NORTH]==0
    assert new.probabilities(world)[0][A.WEST]==0
    for _ in range(20):
        assert new.actions(world)[0] in (A.STAY,A.SOUTH,A.EAST)


def test_eat_mask_requires_food_and_nonfull_energy(world):
    agent=AgentState(0,3,3,energy=95)
    world.agents=[agent]
    world.food[:]=False
    assert not movement_mask(observe_agent(world,agent))[A.EAT]
    world.food[3,3]=True
    assert movement_mask(observe_agent(world,agent))[A.EAT]
    agent.energy=100
    assert not movement_mask(observe_agent(world,agent))[A.EAT]
