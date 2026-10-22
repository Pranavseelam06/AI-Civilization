"""Load a local PPO checkpoint for inference without updating its weights."""

import pickle
import numpy as np
import torch

from civsim.observations import observe_agent, movement_mask
from learning.networks import ActorCritic, mask_action_logits
from civsim.actions import Action


def checkpoint_layout(saved):
    """Supported observation/action pairs; never resize or reinterpret weights."""
    weights = saved['model_state_dict']
    layout = (weights['features.0.weight'].shape[1], weights['actor.weight'].shape[0])
    if layout not in ((101, 5), (101, 6), (102, 7)):
        raise ValueError('Unsupported checkpoint observation/action layout')
    return layout


def checkpoint_food_carrying(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    return checkpoint_layout(saved) == (102, 7)


def compatible_checkpoint(path, *, allow_legacy=False):
    """Accept original explicit-EAT and carrying models; legacy five is opt-in."""
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        layout = checkpoint_layout(saved)
        return layout != (101, 5) or allow_legacy
    except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError):
        return False


class LearnedPolicy:
    def __init__(self, path, config, seed, *, allow_legacy=False):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(saved, dict) or 'model_state_dict' not in saved:
            raise ValueError('Checkpoint must contain model_state_dict')
        self.observation_size, self.action_count = checkpoint_layout(saved)
        self.food_carrying = self.observation_size == 102
        if self.food_carrying != config.food_carrying:
            raise ValueError('Checkpoint food-carrying mode does not match environment')
        self.automatic_eating = self.action_count == 5
        if self.automatic_eating and not allow_legacy:
            raise ValueError('This checkpoint uses the old five-action automatic-eating rules. '
                             'Train a new six-action checkpoint with explicit EAT.')
        environment = saved.get('metadata', {}).get('environment', {})
        # Legacy checkpoints keep their original unmasked behavior for comparison.
        self.action_masking = saved.get('metadata', {}).get('action_masking', False)
        # Current viewer supports the existing environment and observation format.
        for key in ('width','height','num_agents'):
            if key in environment and environment[key] != getattr(config,key):
                raise ValueError(f'Checkpoint {key} does not match viewer settings')
        # Model initialization must not alter another session's action randomness.
        with torch.random.fork_rng():
            self.model = ActorCritic(observation_size=self.observation_size, action_count=self.action_count)
        self.model.load_state_dict(saved['model_state_dict'])
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError('Checkpoint contains non-finite weights')
        self.model.eval()
        self.generator = torch.Generator(device='cpu').manual_seed(seed + 10000)

    @torch.no_grad()
    def probabilities(self, world):
        """Inspect action probabilities without drawing actions or advancing RNG."""
        if world.config.food_carrying != self.food_carrying:
            raise ValueError('Checkpoint food-carrying mode does not match world')
        agents = [a for a in world.agents if a.alive]
        if not agents:
            return {}
        inputs = torch.from_numpy(np.stack([observe_agent(world,a) for a in agents]))
        logits, _ = self.model(inputs)
        if self.action_masking:
            masks = np.stack([movement_mask(row)[:self.action_count] for row in inputs.numpy()])
            logits = mask_action_logits(logits, masks)
        return {a.id: row for a,row in zip(agents,logits.softmax(-1).tolist())}

    @torch.no_grad()
    def actions(self, world):
        if world.config.food_carrying != self.food_carrying:
            raise ValueError('Checkpoint food-carrying mode does not match world')
        agents = [a for a in world.agents if a.alive]
        if not agents:
            return {}
        observations = torch.from_numpy(np.stack([observe_agent(world,a) for a in agents]))
        logits, _ = self.model(observations)
        if self.action_masking:
            masks = np.stack([movement_mask(row)[:self.action_count] for row in observations.numpy()])
            logits = mask_action_logits(logits, masks)
        probabilities = logits.softmax(dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=self.generator).squeeze(-1)
        return {a.id:int(action) for a,action in zip(agents,sampled.tolist())}
