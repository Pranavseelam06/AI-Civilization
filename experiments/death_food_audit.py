"""Read-only food and action traces from the final 30 ticks before deaths."""

from collections import deque
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from civsim.actions import Action
from civsim.config import SimulationConfig
from civsim.world import World
from experiments.diagnostics import movement_outcomes
from experiments.long_survival import write_csv
from learning.inference import LearnedPolicy


def snapshot(world, agent, action, outcome, food_positions):
    """Observe pre-action state; distance is optimistic, not a path guarantee."""
    distances = np.abs(food_positions - [agent.y, agent.x])
    visible = distances.max(axis=1) <= 2 if len(distances) else np.zeros(0, dtype=bool)
    manhattan = distances.sum(axis=1)
    nearest = int(manhattan.min()) if len(manhattan) else -1
    visible_distance = int(manhattan[visible].min()) if visible.any() else -1
    return {'tick': world.tick, 'energy': agent.energy, 'x': agent.x, 'y': agent.y,
            'action': Action(action).name, 'outcome': outcome,
            'visible_food': int(visible.sum()), 'food_underfoot': int(world.food[agent.y, agent.x]),
            'nearest_visible_distance': visible_distance, 'global_food': len(food_positions),
            'nearest_global_distance': nearest,
            'visible_within_energy_budget': int(visible_distance >= 0 and visible_distance < agent.energy),
            'any_food_within_energy_budget': int(nearest >= 0 and nearest < agent.energy)}


def run_episode(checkpoint, seed, horizon=2000, window=30):
    world = World(SimulationConfig(horizon=horizon))
    world.reset(seed)
    policy = LearnedPolicy(checkpoint, world.config, seed)
    histories = {a.id: deque(maxlen=window) for a in world.agents}
    traces, deaths = [], []
    for _ in range(horizon):
        living = [a for a in world.agents if a.alive]
        if not living:
            break
        actions = policy.actions(world)
        outcomes = movement_outcomes(world, actions)
        food = np.argwhere(world.food)
        for agent in living:
            histories[agent.id].append(snapshot(world, agent, actions[agent.id], outcomes[agent.id], food))
        world.step(actions)
        for agent in living:
            if not agent.alive:
                deaths.append({'world_seed': seed, 'agent_id': agent.id, 'death_tick': world.tick})
                traces.extend({'world_seed': seed, 'agent_id': agent.id, 'death_tick': world.tick, **row}
                              for row in histories[agent.id])
    return deaths, traces


def summarize(traces, deaths):
    size = len(traces)
    return {'deaths': len(deaths), 'pre_death_agent_ticks': size,
            'window': 30,
            'no_visible_food_fraction': sum(r['visible_food'] == 0 for r in traces) / size if size else 0,
            'visible_food_within_energy_budget_fraction': sum(r['visible_within_energy_budget'] for r in traces) / size if size else 0,
            'no_global_food_within_energy_budget_fraction': sum(not r['any_food_within_energy_budget'] for r in traces) / size if size else 0,
            'foodless_stay_ticks': sum(r['action'] == 'STAY' and r['visible_food'] == 0 for r in traces),
            'missed_underfoot_meals': sum(r['food_underfoot'] and r['action'] != 'EAT' for r in traces),
            'eat_actions': sum(r['action'] == 'EAT' for r in traces),
            'blocked_moves': sum(r['outcome'] in ('occupied', 'contested', 'boundary', 'obstacle') for r in traces),
            'deaths_with_no_visible_food_in_window': sum(not any(r['visible_food'] for r in traces
                if r['world_seed'] == d['world_seed'] and r['agent_id'] == d['agent_id']) for d in deaths)}


def main():
    torch.set_num_threads(1)
    checkpoint = Path('checkpoints/experiment_12_best.pt')
    output = Path('runs/experiment_12_death_food_audit')
    output.mkdir(exist_ok=False)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    protocol = {'checkpoint': str(checkpoint), 'sha256': digest, 'seeds': list(range(500000, 500030)),
                'horizon': 2000, 'window': 30,
                'measurement': 'Before actions; visible = same 5x5 square as policy; no RNG draws by diagnostics',
                'distance': 'Manhattan lower bound ignores blocking and competitors; distance < energy allows movement then EAT',
                'scope': 'Death windows only; observational audit, not causal attribution'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    deaths, traces = [], []
    for seed in protocol['seeds']:
        events, rows = run_episode(checkpoint, seed)
        deaths.extend(events); traces.extend(rows)
        print(f'Seed {seed}: {len(events)} deaths traced', flush=True)
    # Exact replay must match the preceding audit, not merely similar totals.
    with Path('runs/experiment_12_long_survival/deaths.csv').open() as file:
        previous = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(file)]
    if deaths != previous:
        raise RuntimeError('Death events differ from the original audit')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Checkpoint changed during audit')
    write_csv(output / 'death_windows.csv', traces, list(traces[0]))
    summary = summarize(traces, deaths)
    summary.update({'exact_death_replay_match': True, 'checkpoint_unchanged': True})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
