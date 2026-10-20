from civsim.config import SimulationConfig

def test_starting_settings():
    config = SimulationConfig()
    assert (config.width, config.height, config.num_agents, config.horizon) == (32, 32, 12, 500)
    assert SimulationConfig(horizon=None).horizon is None
# TODO: Test validation when configuration validation is implemented.
