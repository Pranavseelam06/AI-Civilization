"""Train a shared PPO policy locally, or collect experience with --collect-only.

Run: python -m learning.train --updates 20 --ticks 128
"""

import argparse
import csv
from datetime import datetime
import time
import subprocess
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from civsim.environments import SimulationEnv
from learning.networks import ActorCritic
from learning.ppo import PPOConfig, ppo_update


@dataclass
class Transition:
    episode: int
    seed: int
    tick: int
    agent_id: int
    observation: np.ndarray
    action: int
    log_probability: float
    value: float
    reward: int
    next_observation: np.ndarray
    next_value: float
    terminated: bool
    truncated: bool


@dataclass
class Rollout:
    transitions: list[Transition]
    world_ticks: int


class RolloutCollector:
    """Keep one world's state between collect calls and use increasing reset seeds.

    Each row represents one living agent's action, not an entire world tick.
    Episode and agent IDs identify a trajectory. Deaths have next_value=0;
    horizon truncations retain the final-state value for later bootstrapping.
    Never join a trajectory across resets, even when the agent ID is reused.
    A collection boundary alone is not a termination or truncation.
    """

    def __init__(self, env: SimulationEnv, model: ActorCritic, seed: int = 123):
        self.env = env
        self.model = model
        self.seed = seed
        self.episode = 0
        self.observations = env.reset(seed)

    @torch.no_grad()
    def collect(self, ticks: int = 128) -> Rollout:
        if type(ticks) is not int or ticks <= 0:
            raise ValueError('ticks must be a positive integer')
        device = next(self.model.parameters()).device
        rows = []
        for _ in range(ticks):
            if self.env.done:
                self.episode += 1
                self.seed += 1
                self.observations = self.env.reset(self.seed)
            ids = list(self.env.agents)
            inputs = torch.as_tensor(
                np.stack([self.observations[i] for i in ids]),
                dtype=torch.float32, device=device,
            )
            actions, log_probs, values = self.model.act(inputs)
            action_list = actions.cpu().tolist()
            result = self.env.step(dict(zip(ids, action_list)))
            final_inputs = torch.as_tensor(
                np.stack([result.observations[i] for i in ids]),
                dtype=torch.float32, device=device,
            )
            _, next_values = self.model(final_inputs)
            for index, agent_id in enumerate(ids):
                terminal = result.terminated[agent_id]
                rows.append(Transition(
                    episode=self.episode, seed=self.seed, tick=result.info['tick'],
                    agent_id=agent_id,
                    observation=self.observations[agent_id].copy(),
                    action=action_list[index],
                    log_probability=float(log_probs[index].item()),
                    value=float(values[index].item()), reward=result.rewards[agent_id],
                    next_observation=result.observations[agent_id].copy(),
                    next_value=0.0 if terminal else float(next_values[index].item()),
                    terminated=terminal, truncated=result.truncated[agent_id],
                ))
            self.observations = {
                i: result.observations[i] for i in self.env.agents
            }
        return Rollout(rows, ticks)


def save_rollout(path: Path, rollout: Rollout, config, torch_seed: int) -> None:
    """Save plain numeric arrays and JSON metadata, readable without pickle."""
    rows = rollout.transitions
    arrays = {
        name: np.asarray([getattr(row, name) for row in rows])
        for name in Transition.__dataclass_fields__
    }
    metadata = {'config': asdict(config), 'torch_seed': torch_seed,
                'torch_version': str(torch.__version__), 'world_ticks': rollout.world_ticks,
                'policy': 'ActorCritic rollout (weights not embedded)', 'format_version': 1}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as output:
        np.savez_compressed(output, **arrays, metadata=json.dumps(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ticks', type=int, default=128, help='World ticks per update')
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/rollout.npz'),
                        help='Output path in collect-only mode')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--threads', type=int, default=1, help='CPU threads; small networks need few')
    args = parser.parse_args()
    if args.ticks <= 0 or args.updates <= 0 or args.seed < 0 or args.threads <= 0:
        parser.error('ticks, updates, threads must be positive; seed must be nonnegative')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    env = SimulationEnv()
    model = ActorCritic()
    collector = RolloutCollector(env, model, args.seed)
    if args.collect_only:
        rollout = collector.collect(args.ticks)
        save_rollout(args.output, rollout, env.world.config, args.seed)
        print(f'Collected {rollout.world_ticks} world ticks, {len(rollout.transitions)} agent transitions.')
        print(f'Saved {args.output}. Network weights were not updated.')
        return

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir = args.run_dir or Path(f'runs/ppo_{stamp}')
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error('Choose an empty run directory to preserve earlier results')
    checkpoint = args.checkpoint or Path(f'checkpoints/ppo_{stamp}.pt')
    if checkpoint.exists():
        parser.error('Checkpoint already exists; choose a new path')
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ppo_config = PPOConfig()
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)
    commit = subprocess.run(['git','rev-parse','HEAD'], capture_output=True, text=True).stdout.strip()
    metadata = {'seed': args.seed, 'environment': asdict(env.world.config),
                'ppo': asdict(ppo_config), 'updates': args.updates, 'ticks_per_update': args.ticks,
                'torch_version': str(torch.__version__), 'git_commit': commit,
                'device': 'cpu', 'threads': args.threads,
                'architecture': {'observation_size': 101, 'hidden_units': [128,128], 'action_count': 5}}
    (run_dir/'config.json').write_text(json.dumps(metadata,indent=2))
    started = time.perf_counter()
    total_ticks = total_transitions = 0
    with (run_dir/'metrics.csv').open('w',newline='') as output:
        writer = None
        for update in range(1,args.updates+1):
            # Never train on old saved rollouts: collect fresh data after each update.
            rollout = collector.collect(args.ticks)
            metrics = ppo_update(model,optimizer,rollout,ppo_config)
            total_ticks += rollout.world_ticks
            total_transitions += len(rollout.transitions)
            metrics = {'update':update, 'world_ticks':total_ticks,
                       'agent_transitions':total_transitions,
                       'mean_step_reward':float(np.mean([r.reward for r in rollout.transitions])),
                       'deaths':sum(r.terminated for r in rollout.transitions),
                       'elapsed_seconds':time.perf_counter()-started, **metrics}
            if writer is None:
                writer = csv.DictWriter(output,fieldnames=list(metrics))
                writer.writeheader()
            writer.writerow(metrics)
            output.flush()
            print(f"Update {update}/{args.updates} | ticks {total_ticks} | "
                  f"reward/transition {metrics['mean_step_reward']:.2f} | "
                  f"value loss {metrics['value_loss']:.2f}",flush=True)
    # Saves learned weights and optimizer, but not an exact resumable world state.
    torch.save({'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),
                'metadata':metadata, 'world_ticks':total_ticks,
                'agent_transitions':total_transitions},checkpoint)
    print(f'Saved checkpoint: {checkpoint}')
    print(f'Metrics and configuration: {run_dir}')
    print('Training completed. Held-out evaluation is needed to establish improvement.')


if __name__ == '__main__':
    main()
