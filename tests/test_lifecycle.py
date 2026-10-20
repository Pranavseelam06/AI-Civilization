from civsim.agents import AgentState
from civsim.actions import Action

def test_death_and_dead_agent_inactivity(world):
    world.food[:] = False
    world.agents = [AgentState(0,0,0,energy=1)]
    world.step({})
    assert world.agents[0].energy == 0 and not world.agents[0].alive
    world.step({0:Action.EAST})
    assert world.agents[0].x == 0 and world.agents[0].energy == 0

def test_dead_agents_do_not_block(world):
    world.agents = [AgentState(0,0,0),AgentState(1,1,0,energy=0,alive=False)]
    world.step({0:Action.EAST})
    assert world.agents[0].x == 1
# TODO: Environment termination/truncation contract is not implemented yet.
