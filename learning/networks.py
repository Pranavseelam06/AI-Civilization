"""Shared actor-critic network for local reinforcement learning."""

import torch
from torch import nn
from torch.distributions import Categorical

from civsim.observations import OBSERVATION_SIZE


class ActorCritic(nn.Module):
    def __init__(self, observation_size=OBSERVATION_SIZE, action_count=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, action_count)
        self.critic = nn.Linear(128, 1)

    def forward(self, observations):
        """Map (batch, 101) observations to (batch, 5) logits and (batch,) values."""
        features = self.features(observations)
        return self.actor(features), self.critic(features).squeeze(-1)

    def act(self, observations):
        """Sample actions; retain log probabilities and values for training."""
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), values


# TODO: Implement PPO training separately; this network starts untrained.
