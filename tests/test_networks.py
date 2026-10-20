import pytest
import numpy as np

torch = pytest.importorskip("torch", reason="Install the learning extra for network tests")

from learning.networks import ActorCritic
from civsim.environments import SimulationEnv


@pytest.mark.parametrize("batch_size", [1, 12])
def test_actor_critic_shapes_and_action_probabilities(batch_size):
    model = ActorCritic()
    observations = torch.zeros((batch_size, 101))
    logits, values = model(observations)
    assert logits.shape == (batch_size, 6)
    assert values.shape == (batch_size,)
    actions, log_probs, sampled_values = model.act(observations)
    assert actions.shape == log_probs.shape == sampled_values.shape == (batch_size,)
    assert torch.all((actions >= 0) & (actions < 6))
    expected = torch.log_softmax(logits, dim=-1).gather(1, actions[:, None]).squeeze(-1)
    torch.testing.assert_close(log_probs, expected)
    torch.testing.assert_close(sampled_values, values)


def test_both_heads_and_shared_layers_receive_finite_gradients():
    torch.manual_seed(123)
    model = ActorCritic()
    _, log_probs, values = model.act(torch.rand((12, 101)))
    loss = -log_probs.mean() + (values - 1).square().mean()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_network_actions_work_with_environment():
    env = SimulationEnv()
    observations = env.reset(123)
    ids = list(observations)
    inputs = torch.tensor(np.stack([observations[i] for i in ids]))
    with torch.no_grad():
        actions, _, _ = ActorCritic().act(inputs)
    result = env.step(dict(zip(ids, actions.tolist())))
    assert result.info['tick'] == 1
    assert set(result.rewards) == set(ids)
