import pytest
import numpy as np
from civsim.actions import Action as A
from civsim.agents import AgentState
from experiments.diagnostics import movement_outcomes


@pytest.mark.parametrize('agents,actions,blocked,expected',[
    ([AgentState(0,0,0)],{0:A.WEST},[],{0:'boundary'}),
    ([AgentState(0,0,0)],{0:A.EAST},[(1,0)],{0:'obstacle'}),
    ([AgentState(0,0,0),AgentState(1,1,0)],{0:A.EAST},[],{0:'occupied',1:'stay'}),
    ([AgentState(0,0,0),AgentState(1,2,0)],{0:A.EAST,1:A.WEST},[],{0:'contested',1:'contested'}),
    ([AgentState(0,0,0)],{0:A.EAST},[],{0:'moved'}),
    ([AgentState(0,0,0)],{0:999},[],{0:'invalid'}),
])
def test_diagnostics_match_world_movement_without_mutation(world,agents,actions,blocked,expected):
    world.agents=agents
    for x,y in blocked:world.grid[y,x]=1
    positions={a.id:(a.x,a.y) for a in agents}
    assert movement_outcomes(world,actions)==expected
    assert {a.id:(a.x,a.y) for a in agents}==positions
    world.step(actions)
    for agent in agents:
        moved=(agent.x,agent.y)!=positions[agent.id]
        assert moved==(expected[agent.id]=='moved')


def test_diagnostic_evaluation_matches_existing_baseline():
    pytest.importorskip('torch')
    from learning.evaluate import evaluate_episode
    from experiments.run_baselines import run_episode
    from civsim.config import SimulationConfig
    config=SimulationConfig()
    for policy in ('random','heuristic'):
        row,_=evaluate_episode(policy,10000,config)
        expected=run_episode(policy,10000,config)
        for key in ('survivors','mean_survival_ticks','food_consumed','episode_ticks'):
            assert row[key]==expected[key]
        assert row['agent_ticks']==sum(row[k] for k in row if k.startswith('actions_'))


def test_corner_trace_and_probability_inspection_do_not_change_sampling(tmp_path):
    torch=pytest.importorskip('torch')
    from learning.networks import ActorCritic
    from learning.inference import LearnedPolicy
    from learning.evaluate import evaluate_episode
    from civsim.config import SimulationConfig
    from civsim.world import World
    config=SimulationConfig(width=4,height=4,num_agents=1,horizon=10)
    model=ActorCritic()
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        model.actor.bias.fill_(-100)
        model.actor.bias[A.EAST]=100
    path=tmp_path/'east.pt';torch.save({'model_state_dict':model.state_dict()},path)
    row,trace=evaluate_episode('ppo',123,config,path)
    assert trace and all(r['p_east']==1 for r in trace)
    assert row['actions_boundary']>0
    assert row['corner_tick_fraction']==1
    world=World(config);world.reset(123)
    one,two=LearnedPolicy(path,config,123),LearnedPolicy(path,config,123)
    one.probabilities(world)
    assert one.actions(world)==two.actions(world)


def test_evaluation_reports_selection_mode_and_argmax_is_repeatable(tmp_path):
    torch = pytest.importorskip('torch')
    from learning.evaluate import evaluate_episode
    from learning.networks import ActorCritic
    from civsim.config import SimulationConfig
    model = ActorCritic()
    path = tmp_path / 'policy.pt'
    torch.save({'model_state_dict': model.state_dict(), 'metadata': {'action_masking': True}}, path)
    config = SimulationConfig(width=4, height=4, num_agents=1, horizon=10)
    first, trace = evaluate_episode('ppo', 123, config, path, action_selection='argmax')
    assert first['action_selection'] == 'argmax'
    assert all(row['action_selection'] == 'argmax' for row in trace)
    assert (first, trace) == evaluate_episode('ppo', 123, config, path, action_selection='argmax')
    baseline, _ = evaluate_episode('random', 123, config, action_selection='argmax')
    assert baseline['action_selection'] == 'baseline'
    with pytest.raises(ValueError, match='action_selection'):
        evaluate_episode('ppo', 123, config, path, action_selection='invalid')
