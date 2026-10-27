"""Use the actual new spatial grid through all three experts and backward."""
import types
import pytest
import torch

from fastwam.memory.robomme_dense_clock import dense_prefix_frames
from test_robomme_query_density import make_model, batch


@pytest.mark.parametrize('current,boundary', [(0,0), (67,35), (1635,35)])
def test_new_spatial_grid_compact_training_and_gradient(current, boundary):
    m = make_model().train()
    m.helios_config.update(history_padding='compact', view_layout='two_view_horizontal',
                           anchor_mode='demo_execution_first')
    x = {k:v[:1] for k,v in batch(m,current=current,boundary=boundary).items()}
    x.update(anchor_latents=torch.randn(1,2,4,1,16,32),
             history_latents=torch.randn(1,75,4,1,16,32),
             input_latents=torch.randn(1,4,3,16,32),
             past_action_target=torch.randn(1,34,8,64))
    observed = min(75,len(dense_prefix_frames(current,boundary)))
    x['history_valid'] = torch.arange(75)[None] >= 75-observed
    x['anchor_valid'] = torch.tensor([[boundary > 0, True]])
    m.build_inputs = types.MethodType(lambda self,sample,tiled=False:sample,m)
    loss, metrics = m.training_loss(x)
    assert torch.isfinite(loss)
    loss.backward()
    for p in (m.action_expert.head.weight, m.past_action_expert.noisy_latent_embedding.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    if observed == 75:
        assert m.video_expert.patch_long.weight.grad.abs().sum() > 0
    assert metrics['loss_future_action'] > 0
