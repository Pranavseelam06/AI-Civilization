"""Load a local PPO checkpoint for inference without updating its weights."""

import numpy as np
import torch

from civsim.observations import observe_agent
from learning.networks import ActorCritic


class LearnedPolicy:
    def __init__(self, path, config, seed):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(saved, dict) or 'model_state_dict' not in saved:
            raise ValueError('Checkpoint must contain model_state_dict')
        environment = saved.get('metadata', {}).get('environment', {})
        # Current viewer supports the existing environment and observation format.
        for key in ('width','height','num_agents'):
            if key in environment and environment[key] != getattr(config,key):
                raise ValueError(f'Checkpoint {key} does not match viewer settings')
        # Model initialization must not alter another session's action randomness.
        with torch.random.fork_rng():
            self.model = ActorCritic()
        self.model.load_state_dict(saved['model_state_dict'])
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError('Checkpoint contains non-finite weights')
        self.model.eval()
        self.generator = torch.Generator(device='cpu').manual_seed(seed + 10000)

    @torch.no_grad()
    def actions(self, world):
        agents = [a for a in world.agents if a.alive]
        if not agents:
            return {}
        observations = torch.from_numpy(np.stack([observe_agent(world,a) for a in agents]))
        logits, _ = self.model(observations)
        probabilities = logits.softmax(dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=self.generator).squeeze(-1)
        return {a.id:int(action) for a,action in zip(agents,sampled.tolist())}
