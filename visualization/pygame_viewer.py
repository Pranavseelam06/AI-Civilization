"""Play saved frames without rerunning the simulation."""

import argparse

import pygame

from civsim.replay import load_replay
from visualization.live_viewer import CELL_SIZE, HEADER_HEIGHT, draw_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', default='runs/demo.jsonl')
    args = parser.parse_args()
    try:
        frames, summary = load_replay(args.replay)
    except ValueError as exc:
        parser.error(str(exc))
    config = frames[0].config
    pygame.init()
    try:
        screen = pygame.display.set_mode((config.width * CELL_SIZE,
                                         config.height * CELL_SIZE + HEADER_HEIGHT))
        clock = pygame.time.Clock()
        index, speed = 0, 10
        paused, running = False, True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_UP:
                        speed = min(120, speed * 2)
                    elif event.key == pygame.K_DOWN:
                        speed = max(1, speed // 2)
                    elif event.key == pygame.K_r:
                        index = 0
            if not running:
                break
            draw_world(screen, frames[index])
            status = 'End' if index == len(frames) - 1 else ('Paused' if paused else 'Playing')
            pygame.display.set_caption(f'Replay | Tick {frames[index].tick} | {status} | {speed} ticks/s | Space: pause, Up/Down: speed, R: restart')
            pygame.display.flip()
            if not paused and index < len(frames) - 1:
                index += 1
            clock.tick(speed)
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
