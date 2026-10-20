import pytest

torch=pytest.importorskip('torch')

from learning.networks import ActorCritic
from learning.train import save_checkpoint


def test_stage_checkpoint_records_actual_update_and_preserves_rng(tmp_path):
    model=ActorCritic()
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    metadata={'updates':1000,'seed':123}
    before=torch.random.get_rng_state().clone()
    path=tmp_path/'stages'/'update_0100.pt'
    save_checkpoint(path,model,optimizer,metadata,12800,120000,100)
    assert torch.equal(before,torch.random.get_rng_state())
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['metadata']['updates']==100
    assert saved['metadata']['planned_updates']==1000
    assert metadata['updates']==1000
    assert saved['world_ticks']==12800
    copy=ActorCritic();copy.load_state_dict(saved['model_state_dict'])
    for name,value in model.state_dict().items():
        torch.testing.assert_close(copy.state_dict()[name],value)
