import numpy as np
from civsim.world import World
from civsim.config import SimulationConfig

def test_seeded_transitions(world):
    other = World(SimulationConfig())
    other.reset(123)
    rng = np.random.default_rng(456)
    for _ in range(150):
        assert world.agents == other.agents
        assert np.array_equal(world.food,other.food)
        assert world.food_consumed == other.food_consumed
        actions = {a.id:int(rng.integers(5)) for a in world.agents if a.alive}
        world.step(actions)
        other.step(actions)
    assert world.agents == other.agents
    assert np.array_equal(world.food,other.food)
