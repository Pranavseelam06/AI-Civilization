import numpy as np
import pytest

torch=pytest.importorskip('torch')

from civsim.actions import Action
from civsim.observations import observe_agent
from experiments.probe_food import probe_policy,probe_world
from learning.networks import ActorCritic
from learning.inference import LearnedPolicy
from civsim.config import SimulationConfig


def save_policy(tmp_path,responsive):
    model=ActorCritic()
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        if responsive:
            for hidden,(action,food_index) in enumerate([(Action.NORTH,7),(Action.SOUTH,17),(Action.EAST,13),(Action.WEST,11)]):
                model.features[0].weight[hidden,food_index]=1
                model.features[2].weight[hidden,hidden]=1
                model.actor.weight[action,hidden]=8
        else:
            model.actor.bias[Action.NORTH]=2
    path=tmp_path/'probe.pt'
    torch.save({'model_state_dict':model.state_dict(),'metadata':{'action_masking':True}},path)
    return LearnedPolicy(path,SimulationConfig(),123)


def test_probe_inputs_only_change_food_location():
    world=probe_world(50)
    baseline=observe_agent(world,world.agents[0])
    world.food[15,16]=True
    difference=observe_agent(world,world.agents[0])-baseline
    assert np.flatnonzero(difference).tolist()==[7]
    assert baseline[-1]==0.5
    assert not baseline[:100].any()


def test_probe_detects_food_blind_directional_bias(tmp_path):
    policy=save_policy(tmp_path,False)
    rows=probe_policy(policy,'test')
    assert len(rows)==15
    assert all(row['distribution_change']==0 for row in rows)
    assert all(row['top_action']=='north' for row in rows)


def test_probe_detects_directional_food_response_without_updating_policy(tmp_path):
    policy=save_policy(tmp_path,True)
    before={k:v.clone() for k,v in policy.model.state_dict().items()}
    rng=policy.generator.get_state().clone()
    rows=probe_policy(policy,'test')
    for row in rows:
        if row['food']!='none':
            assert row['top_action_toward_food']
            assert row['food_direction_delta']>0.5
    assert torch.equal(rng,policy.generator.get_state())
    for key,value in policy.model.state_dict().items():
        torch.testing.assert_close(value,before[key])
