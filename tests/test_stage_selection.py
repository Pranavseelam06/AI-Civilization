import pytest

from experiments.select_stages import selection_rank,summarize_stages


def validation_row(root,update,seed,survivors=5,waste=0):
    return {'policy':'ppo','checkpoint':str(root/f'update_{update:04d}.pt'),
            'world_seed':str(seed),'survivors':str(survivors),
            'food_consumed':'10','food_energy_wasted':str(waste),
            'mean_survival_ticks':'400'}


def test_selection_prioritizes_survival_then_waste_then_earlier_update(tmp_path):
    root=tmp_path/'run'
    rows=[validation_row(root,update,seed,survivors,waste)
          for update,survivors,waste in [(100,4,0),(200,5,10),(300,5,0),(400,5,0)]
          for seed in (1000,1001)]
    summary=summarize_stages(rows,[root],expected_episodes=2)
    winner=min(summary,key=selection_rank)
    assert winner['updates']==300
    assert winner['food_energy_waste_fraction']==0


def test_selection_rejects_unequal_seed_sets(tmp_path):
    root=tmp_path/'run'
    rows=[validation_row(root,100,1000),validation_row(root,200,1001)]
    with pytest.raises(ValueError,match='same validation seed'):
        summarize_stages(rows,[root],expected_episodes=1)


def test_selection_rejects_duplicate_seeds_and_missing_runs(tmp_path):
    root=tmp_path/'run'
    row=validation_row(root,100,1000)
    with pytest.raises(ValueError,match='distinct validation seeds'):
        summarize_stages([row,row],[root],expected_episodes=2)
    with pytest.raises(ValueError,match='no validation results'):
        summarize_stages([row],[root,tmp_path/'missing'],expected_episodes=1)
