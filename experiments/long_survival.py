"""Audit a frozen policy past its training horizon, recording deaths and food waste."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch

from civsim.config import SimulationConfig
from civsim.world import World
from learning.inference import LearnedPolicy


def run_episode(checkpoint, seed, horizon=2000):
    world = World(SimulationConfig(horizon=horizon))
    world.reset(seed)
    policy = LearnedPolicy(checkpoint, world.config, seed)
    deaths = []
    milestones = []
    agent_ticks = 0
    for _ in range(horizon):
        living = {a.id for a in world.agents if a.alive}
        if not living:
            break
        agent_ticks += len(living)
        world.step(policy.actions(world))
        deaths.extend({'world_seed': seed, 'agent_id': a.id, 'death_tick': world.tick}
                      for a in world.agents if a.id in living and not a.alive)
        if world.tick % 500 == 0 or world.tick == horizon:
            milestones.append({'world_seed': seed, 'tick': world.tick,
                               'survivors': sum(a.alive for a in world.agents),
                               'food_consumed': world.food_consumed,
                               'food_energy_wasted': world.food_energy_wasted})
    # Extinct populations remain zero at later milestones.
    for tick in sorted(set(range(500, horizon + 1, 500)) | {horizon}):
        if tick > world.tick:
            milestones.append({'world_seed': seed, 'tick': tick, 'survivors': 0,
                               'food_consumed': world.food_consumed,
                               'food_energy_wasted': world.food_energy_wasted})
    row = {'world_seed': seed, 'survivors': sum(a.alive for a in world.agents),
           'mean_capped_lifespan': agent_ticks / world.config.num_agents,
           'episode_ticks': world.tick, 'food_consumed': world.food_consumed,
           'food_energy_gained': world.food_energy_gained,
           'food_energy_wasted': world.food_energy_wasted}
    return row, deaths, milestones


def write_csv(path, rows, fields):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/experiment_12_best.pt'))
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--start-seed', type=int, default=500000)
    parser.add_argument('--horizon', type=int, default=2000)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/experiment_12_long_survival'))
    args = parser.parse_args()
    if args.episodes < 1 or args.horizon < 1 or args.start_seed < 0:
        parser.error('episodes/horizon must be positive and start-seed nonnegative')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose an empty output directory')
    torch.set_num_threads(1)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {'checkpoint': str(args.checkpoint), 'sha256': checkpoint_hash,
                'horizon': args.horizon, 'episodes': args.episodes, 'start_seed': args.start_seed,
                'sampling': 'Frozen stochastic policy, policy seed = world seed + 10000; no training',
                'lifespan': 'Capped at horizon; survivors have unknown eventual death times'}
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    episodes, deaths, milestones = [], [], []
    for seed in range(args.start_seed, args.start_seed + args.episodes):
        row, events, stages = run_episode(args.checkpoint, seed, args.horizon)
        episodes.append(row); deaths.extend(events); milestones.extend(stages)
        print(f'Seed {seed}: {row["survivors"]}/12 survivors, '
              f'{row["mean_capped_lifespan"]:.1f} capped lifespan, '
              f'{row["food_energy_wasted"]} wasted energy', flush=True)
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    write_csv(args.output_dir / 'episodes.csv', episodes, list(episodes[0]))
    write_csv(args.output_dir / 'deaths.csv', deaths, ['world_seed', 'agent_id', 'death_tick'])
    write_csv(args.output_dir / 'milestones.csv', milestones, list(milestones[0]))
    capacity = 25 * sum(r['food_consumed'] for r in episodes)
    summary = {'mean_survivors_by_tick': {
        str(tick): mean(r['survivors'] for r in milestones if r['tick'] == tick)
        for tick in sorted({r['tick'] for r in milestones})},
        'mean_capped_lifespan': mean(r['mean_capped_lifespan'] for r in episodes),
        'all_survive_episodes': sum(r['survivors'] == 12 for r in episodes),
        'observed_deaths': len(deaths),
        'deaths_after_500': sum(r['death_tick'] > 500 for r in deaths),
        'food_consumed': sum(r['food_consumed'] for r in episodes),
        'food_energy_wasted': sum(r['food_energy_wasted'] for r in episodes),
        'pooled_waste_fraction': sum(r['food_energy_wasted'] for r in episodes) / capacity if capacity else 0,
        'checkpoint_unchanged': True}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
