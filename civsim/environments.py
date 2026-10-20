"""Small custom parallel training interface; no external RL dependency yet."""

from dataclasses import dataclass

import numpy as np

from civsim.actions import Action
from civsim.config import SimulationConfig
from civsim.observations import observe_agent
from civsim.world import World


@dataclass
class StepResult:
    # Includes every agent alive before this step, even one that just died.
    # Final observations are retained for training's end-of-episode handling.
    observations: dict[int, np.ndarray]
    rewards: dict[int, float]
    terminated: dict[int, bool]
    truncated: dict[int, bool]
    done: bool
    info: dict[str, int]


class SimulationEnv:
    """Observe -> act -> receive StepResult. Call reset before the first step.

    agents lists IDs currently allowed to act. Missing actions become STAY,
    just as in World. Actions for dead/unknown agents are rejected. An agent
    that dies appears in its final StepResult, then disappears from subsequent
    results. At a horizon, surviving agents are truncated rather than dead.
    Both endings require reset before another step. horizon=None is unlimited.
    This is a custom interface, not a Gymnasium/PettingZoo implementation.
    """

    def __init__(self, config: SimulationConfig | None = None):
        config = config if config is not None else SimulationConfig()
        if config.num_agents <= 0:
            raise ValueError('Training requires at least one agent.')
        if config.horizon is not None and config.horizon <= 0:
            raise ValueError('Horizon must be positive or None.')
        self.world = World(config)
        self.agents: list[int] = []
        self.done = True

    def reset(self, seed: int) -> dict[int, np.ndarray]:
        self.world.reset(seed)
        self.agents = [agent.id for agent in self.world.agents if agent.alive]
        self.done = False
        return {agent.id: observe_agent(self.world, agent) for agent in self.world.agents}

    def step(self, actions: dict[int, Action]) -> StepResult:
        if self.done:
            raise RuntimeError('Call reset(seed) before stepping a new episode.')
        if set(actions) - set(self.agents):
            raise ValueError('Actions may only target current living agents.')
        acting = set(self.agents)
        rewards = self.world.step(actions)
        participants = [agent for agent in self.world.agents if agent.id in acting]
        reached_limit = (self.world.config.horizon is not None
                         and self.world.tick >= self.world.config.horizon)
        terminated = {agent.id: not agent.alive for agent in participants}
        truncated = {agent.id: reached_limit and agent.alive for agent in participants}
        self.done = reached_limit or not any(agent.alive for agent in self.world.agents)
        self.agents = ([] if self.done else
                       [agent.id for agent in self.world.agents if agent.alive])
        return StepResult(
            observations={agent.id: observe_agent(self.world, agent) for agent in participants},
            rewards={agent.id: rewards[agent.id] for agent in participants},
            terminated=terminated,
            truncated=truncated,
            done=self.done,
            info={'tick': self.world.tick,
                  'food_consumed': self.world.food_consumed,
                  'alive': sum(agent.alive for agent in self.world.agents)},
        )
