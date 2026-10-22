"""Death timing and censored lifetimes in the long-horizon audit."""

from experiments import long_survival
from civsim.world import World


class FoodlessWorld(World):
    def reset(self, seed):
        super().reset(seed)
        self.food[:] = False


class StayPolicy:
    def __init__(self, *args):
        pass

    def actions(self, world):
        return {}  # STAY; occupied cells cannot regrow food.


def test_extinction_records_each_death_once_and_fills_later_milestones(monkeypatch):
    monkeypatch.setattr(long_survival, 'World', FoodlessWorld)
    monkeypatch.setattr(long_survival, 'LearnedPolicy', StayPolicy)
    row, deaths, stages = long_survival.run_episode(None, 42, 2000)
    assert row['survivors'] == 0
    assert row['episode_ticks'] == row['mean_capped_lifespan'] == 100
    assert len(deaths) == len({r['agent_id'] for r in deaths}) == 12
    assert all(r['death_tick'] == 100 for r in deaths)
    assert [r['tick'] for r in stages] == [500, 1000, 1500, 2000]
    assert all(r['survivors'] == 0 for r in stages)


def test_survivors_are_censored_at_horizon_without_death_events(monkeypatch):
    monkeypatch.setattr(long_survival, 'World', FoodlessWorld)
    monkeypatch.setattr(long_survival, 'LearnedPolicy', StayPolicy)
    row, deaths, stages = long_survival.run_episode(None, 42, 20)
    assert row['survivors'] == 12
    assert row['mean_capped_lifespan'] == 20
    assert deaths == []
    assert len(stages) == 1 and stages[0]['tick'] == 20
