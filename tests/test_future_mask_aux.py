"""Real small MoT: information isolation, gradients, initialization, checkpoints."""
import copy
import types
import pytest
import torch
from test_helios_contract import model, make_inputs, joint


def enable(m):
    m._initialize_future_mask_aux(dict(enabled=True, loss_weight=1., latent_row_start=0))
    return m


def mask_sample():
    x = make_inputs(16)
    x['future_mask_latent'] = torch.randn(16, 4, 1, 8, 8)
    x['future_mask_image_is_pad'] = x['image_is_pad'].clone()
    return x


def test_projection_is_channel_concat_and_history_unchanged(model):
    m = enable(model)
    x = make_inputs(2)
    args = (x['anchor_latents'], x['history_latents'], x['input_latents'][:, :, 1:],
            torch.full((2,), 500.), x['context'], x['context_mask'])
    mask = torch.randn_like(args[2])
    a = m._video_pre(*args)
    b = m._video_pre(*args, future_mask=mask)
    torch.testing.assert_close(a['tokens'], b['tokens'], rtol=0, atol=0)
    torch.nn.init.normal_(m.video_expert.future_mask_patch.weight, std=.1)
    b = m._video_pre(*args, future_mask=mask)
    h = b['meta']['mtm']['L_hist']
    torch.testing.assert_close(a['tokens'][:, :h], b['tokens'][:, :h], rtol=0, atol=0)
    v = m.video_expert
    expected = torch.nn.functional.conv3d(torch.cat((args[2], mask), 1),
        torch.cat((v.patch_embedding.weight, v.future_mask_patch.weight), 1),
        v.patch_embedding.bias, stride=v.patch_size).flatten(2).transpose(1, 2)
    torch.testing.assert_close(b['tokens'][:, h:], expected)


def test_future_mask_cannot_change_action_or_online_inference(model):
    m = enable(model).eval()
    torch.nn.init.normal_(m.video_expert.future_mask_patch.weight, std=.1)
    x = make_inputs(2)
    original = m._video_pre
    with torch.no_grad():
        a, _ = joint(m, x)
        m._video_pre = types.MethodType(lambda self,*args,**kw:
            original(*args,**kw,future_mask=torch.randn_like(args[2])*100), m)
        b, _ = joint(m, x, torch.randn_like(x['input_latents'][:, :, 1:])*100)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    m._video_pre = original
    prefix = torch.randn(1,4,2,8,8)
    m._encode_input_image_latents_tensor = types.MethodType(lambda self,image,tiled=False:prefix,m)
    args = dict(prompt=None,input_image=torch.zeros(1),action_horizon=16,proprio=torch.zeros(14),
                context=x['context'][:1],context_mask=x['context_mask'][:1],
                full_kv_frame_index=1,num_inference_steps=2,seed=42)
    a = m.infer_action(**args)['action']
    with torch.no_grad():
        m.video_expert.future_mask_patch.weight.fill_(1000)
        for p in m.video_expert.future_mask_head.parameters(): p.fill_(1000)
    b = m.infer_action(**args)['action']
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_mask_loss_alone_trains_shared_visual_parameters(model):
    m = enable(model)
    m.build_inputs = types.MethodType(lambda self,sample,tiled=False:sample,m)
    m.loss_lambda_video = m.loss_lambda_action = 0.
    x = mask_sample()
    x['image_is_pad'][-1] = True
    x['future_mask_image_is_pad'][-1] = True
    loss, metrics = m.training_loss(x)
    assert metrics['loss_mask'] > 0 and torch.isfinite(loss)
    loss.backward()
    for p in (m.video_expert.future_mask_patch.weight, m.video_expert.future_mask_head.head.weight,
              m.video_expert.patch_long.weight, m.video_expert.patch_embedding.weight,
              m.video_expert.blocks[0].self_attn.q.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert m.action_expert.action_encoder.weight.grad.abs().sum() == 0


def test_mask_padding_guard_and_strict_checkpoint(model,tmp_path):
    baseline = copy.deepcopy(model)
    m = enable(model)
    m.build_inputs = types.MethodType(lambda self,sample,tiled=False:sample,m)
    x = mask_sample()
    x['future_mask_image_is_pad'][0,4] = True
    with pytest.raises(ValueError,match='padding'): m.training_loss(x)
    p = tmp_path / 'mask.pt'
    m.save_checkpoint(p,step=17)
    assert m.load_checkpoint(p) == 17
    with pytest.raises(ValueError,match='contract'): baseline.load_checkpoint(p)
    payload = torch.load(p,weights_only=False)
    del payload['mot']['mixtures.video.future_mask_patch.weight']
    torch.save(payload,p)
    with pytest.raises(RuntimeError): m.load_checkpoint(p)


def test_zero_mask_weight_preserves_initial_baseline_loss(model):
    base = copy.deepcopy(model)
    enable(model)
    model.helios_config['future_mask_aux']['loss_weight'] = 0.
    x = mask_sample()
    for m in (base, model):
        m.build_inputs = types.MethodType(lambda self,sample,tiled=False:sample,m)
    torch.manual_seed(123)
    baseline, old = base.training_loss(x)
    torch.manual_seed(123)
    augmented, new = model.training_loss(x)
    torch.testing.assert_close(baseline, augmented, atol=0, rtol=0)
    assert old['loss_video'] == new['loss_video'] and old['loss_action'] == new['loss_action']


def test_deployment_reconstructs_auxiliary_contract(model,tmp_path):
    # Execute the actual deployment constructor's config-loading prefix without
    # importing simulator dependencies or allocating a production model.
    import ast
    from pathlib import Path
    from omegaconf import OmegaConf
    source = Path(__file__).resolve().parents[1] / 'experiments/robotwin/fastwam_policy/deploy_policy.py'
    tree = ast.parse(source.read_text())
    constructor = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
        and n.name == '__init__' and 'model_cfg_copy = ' in ast.get_source_segment(source.read_text(),n))
    prefix = []
    for statement in constructor.body:
        if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='model' for t in statement.targets):
            break
        prefix.append(statement)
    wrapper = ast.parse('def restore(model_cfg, checkpoint_path):\n    pass').body[0]
    wrapper.body = prefix + [ast.Return(value=ast.Name(id='model_cfg_copy',ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]))
    scope = dict(OmegaConf=OmegaConf,torch=torch)
    exec(compile(module,str(source),'exec'),scope)
    enable(model)
    checkpoint = tmp_path/'deployment.pt'
    model.save_checkpoint(checkpoint)
    config = OmegaConf.create(dict(memory_backend='helios',load_text_encoder=False,
        helios_config=dict(condition_augmentation=True,backbone_init_checkpoint='/training-only.pt')))
    restored = scope['restore'](config,checkpoint)
    assert OmegaConf.to_container(restored.helios_config.future_mask_aux) == model.helios_config['future_mask_aux']
    assert restored.helios_config.backbone_init_checkpoint is None
    assert restored.load_text_encoder
    model.helios_config['future_mask_aux']['current_mask_input'] = True
    model.save_checkpoint(checkpoint)
    with pytest.raises(ValueError,match='Unsupported'): scope['restore'](config,checkpoint)
