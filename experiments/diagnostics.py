"""Read-only movement diagnostics matching the world's simultaneous rules."""

from collections import Counter

from civsim.actions import Action, MOVEMENT_DELTAS


def near_corner(world, agent):
    """Within two cells of both an X edge and a Y edge."""
    height, width = world.grid.shape
    return (min(agent.x, width-1-agent.x) <= 2
            and min(agent.y, height-1-agent.y) <= 2)


def movement_outcomes(world, actions):
    """Classify attempts before stepping; intentional STAY is not rejected."""
    occupied = {(a.x,a.y) for a in world.agents if a.alive}
    height, width = world.grid.shape
    outcomes, candidates = {}, {}
    for agent in world.agents:
        if not agent.alive:
            continue
        try:
            action = Action(actions.get(agent.id, Action.STAY))
        except (ValueError, TypeError):
            outcomes[agent.id] = 'invalid'
            continue
        if action == Action.STAY:
            outcomes[agent.id] = 'stay'
            continue
        dx,dy = MOVEMENT_DELTAS[action]
        x,y = agent.x+dx,agent.y+dy
        if not (0 <= x < width and 0 <= y < height):
            outcomes[agent.id] = 'boundary'
        elif world.grid[y,x] != 0:
            outcomes[agent.id] = 'obstacle'
        elif (x,y) in occupied:
            outcomes[agent.id] = 'occupied'
        else:
            candidates[agent.id] = (x,y)
    counts = Counter(candidates.values())
    for agent_id, destination in candidates.items():
        outcomes[agent_id] = 'contested' if counts[destination] > 1 else 'moved'
    return outcomes
