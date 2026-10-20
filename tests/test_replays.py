import json
import numpy as np
import pytest
from civsim.replay import ReplayRecorder,load_replay
from civsim.world import World
from civsim.config import SimulationConfig

def test_recording_round_trip_and_neutrality(world,tmp_path,monkeypatch):
    other=World(SimulationConfig()); other.reset(123)
    path=tmp_path/'replay.jsonl'
    recorder=ReplayRecorder(path,world,123,456)
    for _ in range(40):
        world.step({}); other.step({}); recorder.record(world)
    recorder.finish(world)
    assert world.agents == other.agents
    assert np.array_equal(world.food,other.food)
    def forbidden(*args,**kwargs): raise AssertionError('Replay must not simulate')
    monkeypatch.setattr(World,'step',forbidden)
    frames,summary=load_replay(path)
    assert len(frames)==41 and frames[0].tick==0 and frames[-1].tick==40
    assert frames[0].agents[0].energy==100
    assert frames[-1].agents==world.agents
    assert np.array_equal(frames[-1].food,world.food)
    assert summary['food_consumed']==world.food_consumed
    frames[-1].agents[0].energy=-1
    assert frames[0].agents[0].energy==100 and world.agents[0].energy>=0

@pytest.mark.parametrize('content',['','not-json','{"type":"header","version":999}'])
def test_invalid_replays_fail_clearly(tmp_path,content):
    path=tmp_path/'bad.jsonl'; path.write_text(content)
    with pytest.raises(ValueError,match='Cannot load replay'):
        load_replay(path)
