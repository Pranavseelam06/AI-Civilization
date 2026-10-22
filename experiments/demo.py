"""Run a small movement-only simulation with random actions."""

import numpy as np

from civsim.actions import Action
from civsim.config import SimulationConfig
from civsim.world import World


def main() -> None:
    config = SimulationConfig()
    world = World(config)
    world.reset(seed=123)
    # Policy randomness is separate from world initialization.
    rng = np.random.default_rng(456)

    print("Starting positions:")
    for agent in world.agents:
        print(f"Agent {agent.id}: ({agent.x}, {agent.y})")

    # Keep this demo finite even if the world is configured without a limit.
    ticks = config.horizon if config.horizon is not None else 500
    for _ in range(ticks):
        actions = {
            agent.id: Action(int(rng.integers(6)))
            for agent in world.agents
        }
        world.step(actions)

    print("\nEnding positions:")
    for agent in world.agents:
        print(f"Agent {agent.id}: ({agent.x}, {agent.y})")
    print(f"\nFinal tick: {world.tick}")


if __name__ == "__main__":
    main()
