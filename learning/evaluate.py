"""Compare checkpoints and baselines on matched evaluation seeds without learning."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, stdev
import time

import numpy as np
import torch

from civsim.actions import Action
from civsim.config import SimulationConfig
from civsim.world import World
from civsim.rewards import REWARD_SPEC
from civsim.observations import nearest_food
from experiments.diagnostics import movement_outcomes, near_corner
from learning.baselines import random_policy, heuristic_policy
from learning.inference import LearnedPolicy, compatible_checkpoint


REJECTED = ('boundary','obstacle','occupied','contested','invalid')


def evaluate_episode(policy, seed, config, checkpoint=None):
    world = World(config)
    if config.horizon is None or config.horizon <= 0:
        raise ValueError('Evaluation needs a positive finite horizon')
    world.reset(seed)
    learned = LearnedPolicy(checkpoint,config,seed) if policy == 'ppo' else None
    if policy not in ('ppo','random','heuristic'):
        raise ValueError('Unknown policy')
    rng = np.random.default_rng(seed+10000)
    counts = Counter()
    survival_ticks = total_reward = corner_ticks = 0
    corner_samples = []
    while world.tick < config.horizon and any(a.alive for a in world.agents):
        living = [a for a in world.agents if a.alive]
        survival_ticks += len(living)
        corners = [a for a in living if near_corner(world,a)]
        corner_ticks += len(corners)
        if learned:
            actions = learned.actions(world)
        else:
            actions = {a.id:(random_policy(rng) if policy=='random' else
                            heuristic_policy(a.energy,nearest_food(world.food,a),rng)) for a in living}
        outcomes = movement_outcomes(world,actions)
        counts.update(outcomes.values())
        # Limit real-corner samples per episode, without advancing action RNG.
        if learned and corners and len(corner_samples) < 20:
            probabilities = learned.probabilities(world)
            for agent in corners:
                if len(corner_samples) >= 20:
                    break
                corner_samples.append({'checkpoint':str(checkpoint), 'world_seed':seed,
                    'tick':world.tick, 'agent_id':agent.id,'x':agent.x,'y':agent.y,
                    'energy':agent.energy,'action':Action(actions[agent.id]).name,
                    'outcome':outcomes[agent.id],
                    **{f'p_{a.name.lower()}':probabilities[agent.id][a.value] for a in Action}})
        total_reward += sum(world.step(actions).values())
    attempts = sum(counts.values())
    move_attempts = attempts - counts['stay'] - counts['eat'] - counts['eat_unavailable']
    rejected = sum(counts[k] for k in REJECTED)
    row = {'policy':policy, 'checkpoint':str(checkpoint) if checkpoint else '',
           'action_masking':bool(learned and learned.action_masking),
           'world_seed':seed, 'policy_seed':seed+10000,
           'survivors':sum(a.alive for a in world.agents),
           'mean_survival_ticks':survival_ticks/config.num_agents,
           'food_consumed':world.food_consumed,'total_reward':total_reward,
           'food_energy_gained':world.food_energy_gained,'food_energy_wasted':world.food_energy_wasted,
           'episode_ticks':world.tick,'agent_ticks':attempts,
           'move_attempts':move_attempts,'rejected_moves':rejected,
           'rejected_move_fraction':rejected/move_attempts if move_attempts else 0,
           'corner_tick_fraction':corner_ticks/attempts if attempts else 0,
           **{f'actions_{k}':counts[k] for k in ('stay','moved','eat','eat_unavailable',*REJECTED)}}
    return row,corner_samples


def write_csv(path,rows):
    if not rows:
        return
    with path.open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',nargs='+',default=['latest'],help='Paths, latest, or all')
    parser.add_argument('--episodes',type=int,default=10)
    parser.add_argument('--start-seed',type=int,help='Default: safely beyond recorded training bounds')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.episodes <= 0 or (args.start_seed is not None and args.start_seed < 0):
        parser.error('episodes must be positive; start-seed must be nonnegative')
    torch.set_num_threads(1)
    folder=Path(__file__).resolve().parents[1]/'checkpoints'
    available=sorted((p for p in folder.glob('*.pt') if compatible_checkpoint(p)),key=lambda p:p.stat().st_mtime,reverse=True)
    paths=[]
    for requested in args.checkpoints:
        if requested in ('latest','all'):
            if not available:
                parser.error('No compatible six-action checkpoints found; train a new model first')
            paths.extend(available[:1] if requested=='latest' else available)
        else:
            path=Path(requested)
            if not path.is_file(): parser.error(f'Checkpoint not found: {path}')
            paths.append(path)
    paths=list(dict.fromkeys(p.resolve() for p in paths))
    training=[]
    safe_seed=10000
    for path in paths:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        metadata=saved.get('metadata',{})
        first=metadata.get('seed')
        ticks=saved.get('world_ticks')
        # At most one reset per world tick: a conservative upper bound.
        if isinstance(first,int) and isinstance(ticks,int):
            safe_seed=max(safe_seed,first+ticks+1)
        training.append({'checkpoint':str(path),'world_ticks':ticks,
                         'agent_transitions':saved.get('agent_transitions'),'metadata':metadata})
    start=args.start_seed if args.start_seed is not None else safe_seed
    seeds=list(range(start,start+args.episodes))
    output=args.output_dir or Path(f'runs/evaluation_{datetime.now():%Y%m%d_%H%M%S_%f}')
    if output.exists() and any(output.iterdir()): parser.error('Choose an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    config=SimulationConfig()
    rows,samples=[],[]
    started=time.perf_counter()
    for policy,checkpoint in [('random',None),('heuristic',None),*[('ppo',p) for p in paths]]:
        results=[]
        for seed in seeds:
            row,trace=evaluate_episode(policy,seed,config,checkpoint)
            rows.append(row);results.append(row);samples.extend(trace)
        survivals=[r['mean_survival_ticks'] for r in results]
        spread=stdev(survivals) if len(survivals)>1 else 0
        name=checkpoint.name if checkpoint else policy
        print(f'{name}: survivors {mean(r["survivors"] for r in results):.2f}/{config.num_agents}, '
              f'survival {mean(survivals):.1f} ticks (SD {spread:.1f}), '
              f'food {mean(r["food_consumed"] for r in results):.1f}, '
              f'rejected moves {mean(r["rejected_move_fraction"] for r in results):.1%}, '
              f'near corners {mean(r["corner_tick_fraction"] for r in results):.1%}',flush=True)
    write_csv(output/'episodes.csv',rows)
    write_csv(output/'corner_probabilities.csv',samples)
    metadata={'evaluation_seeds':seeds,'environment':asdict(config),'reward_spec':REWARD_SPEC,'checkpoints':training,
              'elapsed_seconds':time.perf_counter()-started,
              'seed_note':'Auto seeds exceed conservative recorded training bounds; explicit seeds require checking overlap. Missing checkpoint metadata prevents verifying held-out status.',
              'corner_definition':'Within two cells of both an X edge and a Y edge',
              'rejected_fraction_denominator':'Non-STAY requests, including invalid requests',
              'sampling':'Stochastic policy; one policy RNG seed per world seed; no weight updates'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'Evaluation seeds: {start}–{seeds[-1]}. Saved {output}')
    if not samples: print('No near-corner PPO samples observed; no corner CSV written.')


if __name__=='__main__':
    main()
