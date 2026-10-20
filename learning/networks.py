"""Shared actor-critic network for local reinforcement learning."""

import torch
from torch import nn
from torch.distributions import Categorical

from civsim.observations import OBSERVATION_SIZE
from civsim.actions import Action


def mask_action_logits(logits, action_mask=None):
    """Exclude illegal actions consistently during sampling and PPO updates."""
    if action_mask is None:
        return logits
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
    if mask.shape != logits.shape or not mask.any(dim=-1).all():
        raise ValueError('Action mask must match logits and allow at least one action')
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class ActorCritic(nn.Module):
    def __init__(self, observation_size=OBSERVATION_SIZE, action_count=len(Action)):
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
        """Map observations to six action logits and one value per agent."""
        features = self.features(observations)
        return self.actor(features), self.critic(features).squeeze(-1)

    def act(self, observations, action_mask=None):
        """Sample actions; retain log probabilities and values for training."""
        logits, values = self(observations)
        distribution = Categorical(logits=mask_action_logits(logits, action_mask))
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), values


# PPO updates live in learning/ppo.py; weights start untrained.
