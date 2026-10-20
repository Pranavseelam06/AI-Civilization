"""Parameter-shared PPO with local actor/critic inputs (not centralized MAPPO).

Reference: https://arxiv.org/abs/1707.06347
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical
from learning.networks import mask_action_logits


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    epochs: int = 4
    minibatch_size: int = 256
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5


def advantages_and_returns(rollout, gamma=0.99, gae_lambda=0.95):
    """GAE follows each agent separately; never join deaths, resets, or gaps.

    Death zeroes bootstrap. Time limits use the supplied next_value and stop
    the GAE chain; the collector supplies zero for this finite survival task.
    Ordinary rollout cutoffs bootstrap their next_value.
    """
    rows = rollout.transitions
    advantages = np.zeros(len(rows), dtype=np.float32)
    following = {}
    for index in range(len(rows)-1, -1, -1):
        row = rows[index]
        key = (row.episode, row.agent_id)
        delta = row.reward + gamma * (0 if row.terminated else row.next_value) - row.value
        next_tick, next_advantage = following.get(key, (-1, 0))
        continues = not (row.terminated or row.truncated) and next_tick == row.tick + 1
        advantage = delta + (gamma * gae_lambda * next_advantage if continues else 0)
        advantages[index] = advantage
        following[key] = (row.tick, advantage)
    values = np.asarray([row.value for row in rows], dtype=np.float32)
    return torch.from_numpy(advantages), torch.from_numpy(advantages + values)


def clipped_policy_loss(new_log_probs, old_log_probs, advantages, clip_ratio):
    ratio = (new_log_probs - old_log_probs).exp()
    return -torch.minimum(ratio * advantages,
                          ratio.clamp(1-clip_ratio, 1+clip_ratio) * advantages).mean()


def ppo_update(model, optimizer, rollout, config: PPOConfig):
    """Update on a fresh rollout, with fixed old probabilities and value targets."""
    rows = rollout.transitions
    if not rows:
        raise ValueError('Cannot update from an empty rollout')
    if config.epochs <= 0 or config.minibatch_size <= 0:
        raise ValueError('epochs and minibatch_size must be positive')
    device = next(model.parameters()).device
    observations = torch.as_tensor(np.stack([r.observation for r in rows]), device=device)
    actions = torch.tensor([r.action for r in rows], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([r.log_probability for r in rows], device=device)
    masks = torch.as_tensor(np.stack([
        r.action_mask if r.action_mask is not None else np.ones(model.actor.out_features, dtype=bool)
        for r in rows
    ]), dtype=torch.bool, device=device)
    if not masks.gather(1, actions[:, None]).all():
        raise ValueError('Rollout contains an action excluded by its recorded mask')
    advantages, returns = advantages_and_returns(rollout, config.gamma, config.gae_lambda)
    advantages, returns = advantages.to(device), returns.to(device)
    if len(rows) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    totals = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0, clip_fraction=0.0)
    samples = 0
    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(len(rows), device=device)
        for batch in order.split(config.minibatch_size):
            logits, values = model(observations[batch])
            distribution = Categorical(logits=mask_action_logits(logits, masks[batch]))
            log_probs = distribution.log_prob(actions[batch])
            policy_loss = clipped_policy_loss(log_probs, old_log_probs[batch], advantages[batch], config.clip_ratio)
            value_loss = 0.5 * (values - returns[batch]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                log_ratio = log_probs - old_log_probs[batch]
                ratio = log_ratio.exp()
                stats = {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(),
                         'entropy': entropy.item(), 'approx_kl': ((ratio-1)-log_ratio).mean().item(),
                         'clip_fraction': ((ratio-1).abs() > config.clip_ratio).float().mean().item()}
            for key in totals:
                totals[key] += stats[key] * len(batch)
            samples += len(batch)
    return {key: value/samples for key,value in totals.items()}
