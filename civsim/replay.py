"""Versioned JSONL snapshots for playback, not resumable training checkpoints."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from civsim.agents import AgentState
from civsim.config import SimulationConfig


class ReplayRecorder:
    def __init__(self, path, world, seed: int, policy_seed: int):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('w')
        self.write({'type': 'header', 'version': 1, 'config': asdict(world.config),
                    'eating_mode': 'explicit_action',
                    'seed': seed, 'policy_seed': policy_seed, 'policy': 'heuristic',
                    'terrain': world.grid.tolist()})
        self.record(world)

    def write(self, record):
        self.file.write(json.dumps(record) + '\n')
        self.file.flush()

    def record(self, world):
        self.write({'type': 'frame', 'tick': world.tick,
                    'agents': [asdict(a) for a in world.agents],
                    'food': np.argwhere(world.food).tolist(),
                    'food_consumed': world.food_consumed,
                    'food_picked_up': world.food_picked_up,
                    'food_spoiled': world.food_spoiled,
                    'food_energy_gained': world.food_energy_gained,
                    'food_energy_wasted': world.food_energy_wasted})

    def finish(self, world):
        self.write({'type': 'summary', 'tick': world.tick,
                    'survivors': sum(a.alive for a in world.agents),
                    'food_consumed': world.food_consumed,
                    'food_picked_up': world.food_picked_up,
                    'food_spoiled': world.food_spoiled,
                    'food_energy_gained': world.food_energy_gained,
                    'food_energy_wasted': world.food_energy_wasted})
        self.file.close()


def load_replay(path):
    """Load independent display states. No World instance or step calls required."""
    try:
        with Path(path).open() as source:
            records = [json.loads(line) for line in source if line.strip()]
        header = records[0]
        if header['type'] != 'header' or header['version'] != 1:
            raise ValueError('Unsupported replay header/version')
        config = SimulationConfig(**header['config'])
        terrain = np.array(header['terrain'])
        if terrain.shape != (config.height, config.width):
            raise ValueError('Terrain shape does not match configuration')
        frames = []
        summary = None
        for record in records[1:]:
            if record['type'] == 'summary':
                summary = record
                continue
            if record['type'] != 'frame':
                raise ValueError('Unknown replay record type')
            food = np.zeros(terrain.shape, dtype=bool)
            for y, x in record['food']:
                if not (0 <= x < config.width and 0 <= y < config.height):
                    raise ValueError('Food outside grid')
                food[y, x] = True
            agents = [AgentState(**a) for a in record['agents']]
            if any(not (0 <= a.x < config.width and 0 <= a.y < config.height)
                   for a in agents):
                raise ValueError('Agent outside grid')
            frames.append(SimpleNamespace(config=config, grid=terrain.copy(), food=food,
                          agents=agents, tick=record['tick'],
                          food_consumed=record['food_consumed'],
                          food_picked_up=record.get('food_picked_up',0),
                          food_spoiled=record.get('food_spoiled',0),
                          food_energy_gained=record.get('food_energy_gained',0),
                          food_energy_wasted=record.get('food_energy_wasted',0)))
        if not frames:
            raise ValueError('Replay contains no frames')
        return frames, summary
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(f'Cannot load replay {path}: {exc}') from exc
