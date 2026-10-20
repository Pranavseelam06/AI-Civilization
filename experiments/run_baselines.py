"""Compare random and heuristic policies on matched world seeds."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import numpy as np

from civsim.config import SimulationConfig
from civsim.observations import nearest_food
from civsim.world import World
from learning.baselines import heuristic_policy, random_policy


def run_episode(policy: str, seed: int, config: SimulationConfig) -> dict:
    """Measure survival time capped at the evaluation horizon, not full lifespan."""
    if policy not in ("random", "heuristic"):
        raise ValueError("Unknown policy.")
    if config.horizon is None or config.horizon <= 0 or config.num_agents <= 0:
        raise ValueError("Evaluation requires a positive horizon and population.")
    world = World(config)
    world.reset(seed)
    policy_seed = seed + 10000
    rng = np.random.default_rng(policy_seed)
    survival_ticks = 0

    while world.tick < config.horizon and any(a.alive for a in world.agents):
        # Count each tick that an agent starts alive, including its death tick.
        survival_ticks += sum(a.alive for a in world.agents)
        actions = {}
        for agent in world.agents:
            if not agent.alive:
                continue
            if policy == "random":
                actions[agent.id] = random_policy(rng)
            else:
                actions[agent.id] = heuristic_policy(
                    agent.energy, nearest_food(world.food, agent), rng
                )
        world.step(actions)

    return {
        "policy": policy,
        "world_seed": seed,
        "policy_seed": policy_seed,
        **asdict(config),
        "episode_ticks": world.tick,
        "survivors": sum(a.alive for a in world.agents),
        "mean_survival_ticks": survival_ticks / config.num_agents,
        "food_consumed": world.food_consumed,
        "food_energy_gained": world.food_energy_gained,
        "food_energy_wasted": world.food_energy_wasted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("runs/baselines.csv"))
    args = parser.parse_args()
    if args.episodes <= 0 or args.start_seed < 0:
        parser.error("episodes must be positive and start-seed must be nonnegative")

    config = SimulationConfig()
    rows = [
        run_episode(policy, seed, config)
        for seed in range(args.start_seed, args.start_seed + args.episodes)
        for policy in ("random", "heuristic")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Matched world seeds: {args.start_seed}–{args.start_seed + args.episodes - 1}")
    print(f"Survival time is capped at {config.horizon} ticks.")
    for policy in ("random", "heuristic"):
        results = [row for row in rows if row["policy"] == policy]
        survivors = [row["survivors"] for row in results]
        print(
            f"{policy}: mean survivors {mean(survivors):.2f}/{config.num_agents} "
            f"(range {min(survivors)}–{max(survivors)}), "
            f"mean survival {mean(r['mean_survival_ticks'] for r in results):.1f} ticks, "
            f"mean food consumed {mean(r['food_consumed'] for r in results):.1f}"
        )
    print(f"Saved episode results to {args.output}")


if __name__ == "__main__":
    main()
