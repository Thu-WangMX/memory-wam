import torch
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from fastwam.memory.helios_history import memory_variant_profile,validate_memory_variant,select_history
from fastwam.models.wan22.helios_video import MTMVideoDiT

REPO='/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison'

def test_profile_and_training_config():
    p=memory_variant_profile('long75_less_spatial')
    validate_memory_variant('long75_less_spatial',p['history_sizes'],p['patch_long'],p['patch_mid'],2)
    assert p['history_sizes']==[72,2,1] and p['patch_long']==[4,4,4]
    with pytest.raises(ValueError):
        validate_memory_variant('long75_less_spatial',[72,2,1],[4,8,8],[2,4,4],2)
    with initialize_config_dir(config_dir=REPO+'/configs',version_base='1.3'):
        base=compose(config_name='train',overrides=['task=robomme_helios_push75_flow'])
        cfg=compose(config_name='train',overrides=['task=robomme_helios_push75_flow','model.helios_config.memory_variant=long75_less_spatial','model.helios_config.patch_long=[4,4,4]'])
    b=OmegaConf.to_container(base.model.helios_config,resolve=True)
    c=OmegaConf.to_container(cfg.model.helios_config,resolve=True)
    assert {k for k in b if b[k]!=c[k]}=={'memory_variant','patch_long'}
    assert list(cfg.data.train.history_sizes)==[72,2,1]
    assert cfg.model.helios_config.past_action.past_query_slots==32
    assert cfg.model.helios_config.prediction_horizon==32

def test_patch_initialization_tokens_gradients_and_reload():
    torch.set_num_threads(2)
    kw=dict(hidden_dim=32,ffn_dim=64,text_dim=16,freq_dim=16,eps=1e-6,num_heads=2,attn_head_dim=24,num_layers=1,in_dim=4,out_dim=4,patch_size=(1,2,2),has_image_input=False,seperated_timestep=True,multi_term_memory=True,mtm_history_sizes=(72,2,1),mtm_anchor_size=2,mtm_pred_size=2,mtm_patch_kernel_long=(4,4,4))
    m=MTMVideoDiT(**kw)
    x=torch.randn(2,4,72,16,32)
    y=m.patch_long(x)
    assert y.shape==(2,32,18,4,8)
    y.square().mean().backward()
    assert m.patch_long.weight.grad.abs().sum()>0
    # Initialization preserves the base projection on a constant input.
    a=m.patch_long(torch.ones(1,4,4,4,4)).flatten()
    b=m.patch_embedding(torch.ones(1,4,1,2,2)).flatten()
    torch.testing.assert_close(a,b)
    restored=MTMVideoDiT(**kw);restored.load_state_dict(m.state_dict(),strict=True)
    torch.testing.assert_close(restored.patch_long(x),y)
    h=torch.randn(12,4,1,16,32)
    hist,valid=select_history(h,(72,2,1),'long75_less_spatial')
    assert valid.sum()==12 and hist.shape[0]==75
