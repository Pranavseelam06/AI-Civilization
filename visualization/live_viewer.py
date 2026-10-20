"""Watch the world: python -m visualization.live_viewer."""

import numpy as np
import pygame
from datetime import datetime

from civsim.config import SimulationConfig
from civsim.observations import nearest_food
from civsim.world import World
from civsim.replay import ReplayRecorder
from learning.baselines import heuristic_policy


CELL_SIZE = 20
HEADER_HEIGHT = 0
TICKS_PER_SECOND = 10


def draw_world(screen: pygame.Surface, world: World) -> None:
    """Draw current state without modifying the world."""
    screen.fill((24, 30, 28))
    width = world.config.width * CELL_SIZE
    height = world.config.height * CELL_SIZE
    for x in range(world.config.width + 1):
        pygame.draw.line(screen, (48, 58, 52),
                         (x * CELL_SIZE, HEADER_HEIGHT),
                         (x * CELL_SIZE, HEADER_HEIGHT + height))
    for y in range(world.config.height + 1):
        pygame.draw.line(screen, (48, 58, 52),
                         (0, HEADER_HEIGHT + y * CELL_SIZE),
                         (width, HEADER_HEIGHT + y * CELL_SIZE))

    for y, x in np.argwhere(world.food):
        pygame.draw.circle(screen, (80, 220, 90),
                           (int(x) * CELL_SIZE + CELL_SIZE // 2,
                            HEADER_HEIGHT + int(y) * CELL_SIZE + CELL_SIZE // 2), 4)

    for agent in world.agents:
        if not agent.alive:
            continue
        color = pygame.Color(0)
        color.hsva = ((agent.id * 137) % 360, 65, 95, 100)
        rect = pygame.Rect(agent.x * CELL_SIZE + 2,
                           HEADER_HEIGHT + agent.y * CELL_SIZE + 2,
                           CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(screen, color, rect)
        # Keep uneaten food visible underneath an agent standing on its cell.
        if world.food[agent.y, agent.x]:
            pygame.draw.circle(screen, (80, 220, 90), rect.center, 4)

    limit = world.config.horizon
    counter = str(world.tick) if limit is None else f"{world.tick}/{limit}"
    alive_count = sum(agent.alive for agent in world.agents)
    finished = alive_count == 0 or (limit is not None and world.tick >= limit)
    label = f"Tick: {counter}  |  Alive: {alive_count}/{len(world.agents)}"
    if finished:
        label += "  |  Finished"
    pygame.display.set_caption(f"AI Civilization | {label}")


def main() -> None:
    world = World(SimulationConfig())
    world.reset(seed=123)
    rng = np.random.default_rng(456)
    path = f"runs/live_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl"
    recorder = ReplayRecorder(path, world, seed=123, policy_seed=456)
    print(f"Recording to {path}")

    pygame.init()
    try:
        screen = pygame.display.set_mode(
            (world.config.width * CELL_SIZE,
             world.config.height * CELL_SIZE + HEADER_HEIGHT)
        )
        pygame.display.set_caption("AI Civilization — live movement")
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            if not running:
                break

            limit = world.config.horizon
            if (limit is None or world.tick < limit) and any(a.alive for a in world.agents):
                actions = {
                    agent.id: heuristic_policy(
                        agent.energy, nearest_food(world.food, agent), rng
                    )
                    for agent in world.agents
                    if agent.alive
                }
                world.step(actions)
                recorder.record(world)
            draw_world(screen, world)
            pygame.display.flip()
            clock.tick(TICKS_PER_SECOND)
    finally:
        recorder.finish(world)
        pygame.quit()


if __name__ == "__main__":
    main()
