"""Test replacing foodless STAY choices with random legal movement; no training."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from civsim.actions import Action
from civsim.config import SimulationConfig
from civsim.observations import observe_agent, movement_mask
from civsim.rewards import REWARD_SPEC
from civsim.world import World
from experiments.diagnostics import movement_outcomes
from learning.evaluate import write_csv
from learning.inference import LearnedPolicy


def exploration_actions(world, actions, rng, enabled=True):
    """Only replace explicit STAY with no visible food and at least one legal move.

    Existing movement choices and food-visible STAY are untouched. STAY is kept
    if boxed in. Extra randomness uses its own generator, not the policy's RNG.
    """
    result = dict(actions)
    counts = dict(foodless_agent_ticks=0, foodless_stays=0, replacements=0, boxed_stays=0)
    for agent in world.agents:
        if not agent.alive:
            continue
        observation = observe_agent(world,agent)
        if observation[:25].any():
            continue
        counts['foodless_agent_ticks'] += 1
        if actions.get(agent.id,Action.STAY) != Action.STAY:
            continue
        counts['foodless_stays'] += 1
        legal = np.flatnonzero(movement_mask(observation))
        legal = legal[legal != Action.STAY]
        if len(legal) == 0:
            counts['boxed_stays'] += 1
        elif enabled:
            result[agent.id] = int(rng.choice(legal))
            counts['replacements'] += 1
    return result,counts


def run_episode(checkpoint, seed, explore, config=None):
    config = config or SimulationConfig()
    if config.horizon is None or config.horizon <= 0:
        raise ValueError('Evaluation requires a finite positive horizon')
    world = World(config);world.reset(seed)
    policy = LearnedPolicy(checkpoint,config,seed)
    rng = np.random.default_rng(seed + 20000)
    totals = dict(foodless_agent_ticks=0,foodless_stays=0,replacements=0,boxed_stays=0)
    survival_ticks = reward = rejections = 0
    while world.tick < config.horizon and any(a.alive for a in world.agents):
        survival_ticks += sum(a.alive for a in world.agents)
        actions,stats = exploration_actions(world,policy.actions(world),rng,enabled=explore)
        for key in totals:totals[key] += stats[key]
        rejections += sum(reason not in ('moved','stay','eat','eat_unavailable') for reason in movement_outcomes(world,actions).values())
        reward += sum(world.step(actions).values())
    return {'variant':'explore_when_foodless' if explore else 'normal',
            'world_seed':seed,'policy_seed':seed+10000,'exploration_seed':seed+20000,
            'survivors':sum(a.alive for a in world.agents),
            'mean_survival_ticks':survival_ticks/config.num_agents,
            'food_consumed':world.food_consumed,'total_reward':reward,
            'episode_ticks':world.tick,'rejected_moves':rejections,**totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('checkpoints/experiment_5.pt'))
    parser.add_argument('--episodes',type=int,default=30)
    parser.add_argument('--start-seed',type=int,default=40000)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if not args.checkpoint.is_file():parser.error('Checkpoint not found')
    if args.episodes <= 0 or args.start_seed < 0:parser.error('Invalid episode count or start seed')
    torch.set_num_threads(1)
    output=args.output_dir or Path(f'runs/exploration_{datetime.now():%Y%m%d_%H%M%S_%f}')
    if output.exists() and any(output.iterdir()):parser.error('Choose an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    config=SimulationConfig()
    rows=[]
    for seed in range(args.start_seed,args.start_seed+args.episodes):
        rows.extend(run_episode(args.checkpoint,seed,explore,config) for explore in (False,True))
    write_csv(output/'episodes.csv',rows)
    summary={}
    for variant in ('normal','explore_when_foodless'):
        subset=[r for r in rows if r['variant']==variant]
        summary[variant]={key:mean(r[key] for r in subset) for key in
                          ('survivors','mean_survival_ticks','food_consumed','total_reward','foodless_stays','replacements')}
        s=summary[variant]
        print(f"{variant}: survivors {s['survivors']:.2f}/12, survival {s['mean_survival_ticks']:.1f} ticks, "
              f"food {s['food_consumed']:.1f}, foodless STAY choices {s['foodless_stays']:.1f}, "
              f"replacements {s['replacements']:.1f}")
    differences=[rows[i+1]['mean_survival_ticks']-rows[i]['mean_survival_ticks'] for i in range(0,len(rows),2)]
    summary['paired_survival_difference']={
        'mean':mean(differences),'sd':stdev(differences) if len(differences)>1 else 0,
        'wins':sum(d>0 for d in differences),'ties':sum(d==0 for d in differences),
        'losses':sum(d<0 for d in differences)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'metadata.json').write_text(json.dumps({'checkpoint':str(args.checkpoint.resolve()),
        'world_seeds':list(range(args.start_seed,args.start_seed+args.episodes)),
        'config':asdict(config),'reward_spec':REWARD_SPEC,
        'intervention':'Replace sampled STAY only if no food is visible and a legal movement exists',
        'note':'Same seeds, separate exploration RNG. Later trajectories and food coin flips can diverge. One trained checkpoint; no retraining or weight changes.'},indent=2))
    delta=summary['paired_survival_difference']
    print(f"Paired survival difference: {delta['mean']:+.1f} ticks; "
          f"wins/ties/losses: {delta['wins']}/{delta['ties']}/{delta['losses']}")
    print(f'Saved {output}')


if __name__=='__main__':
    main()
