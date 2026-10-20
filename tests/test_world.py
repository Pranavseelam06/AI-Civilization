import numpy as np
import pytest
from civsim.terrain import create_grid, spawn_agents

def test_rectangular_grid_and_unique_spawns():
    grid = create_grid(7, 3)
    grid[0, 0] = 1
    agents = spawn_agents(grid, 20, 123)
    assert grid.shape == (3, 7)
    assert len({(a.x, a.y) for a in agents}) == 20
    assert all(grid[a.y, a.x] == 0 for a in agents)
    assert agents == spawn_agents(grid, 20, 123)

@pytest.mark.parametrize('count', [-1, 22])
def test_impossible_population(count):
    with pytest.raises(ValueError):
        spawn_agents(create_grid(7, 3), count, 123)

def test_reset_restores_state(world):
    original = [(a.x, a.y) for a in world.agents]
    world.agents[0].energy = 0
    world.agents[0].alive = False
    world.food_consumed = 9
    world.tick = 42
    world.reset(123)
    assert [(a.x, a.y) for a in world.agents] == original
    assert all(a.alive and a.energy == 100 and a.z == 0 for a in world.agents)
    assert world.food_consumed == world.tick == 0
