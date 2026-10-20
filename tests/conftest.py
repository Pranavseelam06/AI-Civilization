import pytest
from civsim.config import SimulationConfig
from civsim.world import World

@pytest.fixture
def world():
    world = World(SimulationConfig())
    world.reset(123)
    return world
