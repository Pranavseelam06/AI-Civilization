"""Controlled food-location probes, not a measure of full-world performance."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from civsim.actions import Action, MOVEMENT_DELTAS
from civsim.agents import AgentState
from civsim.config import SimulationConfig
from civsim.world import World
from learning.inference import LearnedPolicy, compatible_checkpoint


CONDITIONS = {'none': None, 'north': Action.NORTH, 'south': Action.SOUTH,
              'east': Action.EAST, 'west': Action.WEST}


def probe_world(energy):
    """One isolated agent in the interior, no food, obstacles, or boundaries visible."""
    world = World(SimulationConfig())
    world.reset(123)
    world.grid[:] = 0
    world.food[:] = False
    world.agents = [AgentState(0, 16, 16, energy=energy)]
    return world


def probe_policy(policy, checkpoint, energies=(25,50,100)):
    rows = []
    for energy in energies:
        world = probe_world(energy)
        agent = world.agents[0]
        baseline = policy.probabilities(world)[agent.id]
        for condition, target in CONDITIONS.items():
            world.food[:] = False
            if target is not None:
                dx,dy = MOVEMENT_DELTAS[target]
                world.food[agent.y+dy,agent.x+dx] = True
            probabilities = policy.probabilities(world)[agent.id]
            top = Action(max(range(len(Action)),key=lambda i:probabilities[i]))
            rows.append({'checkpoint':str(checkpoint),'energy':energy,'food':condition,
                         'action_masking':policy.action_masking,'top_action':top.name.lower(),
                         'top_action_toward_food':target is not None and top == target,
                         'food_direction_probability':probabilities[target] if target is not None else '',
                         'food_direction_delta':probabilities[target]-baseline[target] if target is not None else '',
                         'distribution_change':sum(abs(a-b) for a,b in zip(probabilities,baseline))/2,
                         **{f'p_{a.name.lower()}':probabilities[a] for a in Action}})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',nargs='+',default=['all'],help='all, latest, or checkpoint paths')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    folder=Path(__file__).resolve().parents[1]/'checkpoints'
    available=sorted((p for p in folder.glob('*.pt') if compatible_checkpoint(p)),key=lambda p:p.stat().st_mtime,reverse=True)
    paths=[]
    for name in args.checkpoints:
        if name in ('all','latest'):
            paths.extend(available if name=='all' else available[:1])
        else:
            path=Path(name)
            if not path.is_file():parser.error(f'Checkpoint not found: {path}')
            paths.append(path)
    paths=list(dict.fromkeys(p.resolve() for p in paths))
    if not paths:parser.error('No checkpoints found')
    output=args.output or Path(f'runs/food_probe_{datetime.now():%Y%m%d_%H%M%S_%f}.csv')
    if output.exists():parser.error('Output already exists; choose a new path')
    rows=[]
    for path in paths:
        policy=LearnedPolicy(path,SimulationConfig(),123)
        results=probe_policy(policy,path)
        rows.extend(results)
        print(f'\n{path.name} — 50 energy, one adjacent food item:')
        print('Food    Stay   North  South  East   West   Eat    Change toward food')
        for row in results:
            if row['energy']!=50:continue
            cells=' '.join(f"{row['p_'+a.name.lower()]:6.1%}" for a in Action)
            change=f"{100*row['food_direction_delta']:+.2f} pp" if row['food']!='none' else 'baseline'
            print(f"{row['food']:6} {cells}   {change}")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as destination:
        writer=csv.DictWriter(destination,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(f'\nSaved {output} (energies 25, 50, 100). No actions sampled or weights changed.')
    print('Change = target-direction probability minus its no-food baseline; these are controlled probes only.')


if __name__=='__main__':
    main()
