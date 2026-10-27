"""Real pretrained model + RoboDojo: full backward, causal inference, checkpoint.

On a 24 GiB card each completed parameter gradient is checked on CPU then
released. This exercises the complete backward graph without accumulating a
second full model in GPU RAM. It deliberately performs no optimizer update;
the optional step-0 checkpoint is a wiring smoke, never a trained result.
"""
import argparse
import json
from pathlib import Path
import time

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from fastwam.datasets.robodojo import episode_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latents', required=True)
    parser.add_argument('--episode', type=int, default=300)
    parser.add_argument('--task', default='cover_blocks')
    parser.add_argument('--output', required=True)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--skip-backward', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(0.88)
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=[
            'task=robodojo_helios', f'data.train.tasks=[{args.task}]', 'data.train.split=all',
            f'data.train.episode_indices=[{args.episode}]', f'data.train.full_kv_cache_path={args.latents}',
            f'output_dir={output}',
        ])
    OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), output / 'config.yaml')
    dataset = instantiate(cfg.data.train)
    index = min(20, len(dataset)-1)
    sample = dataset[index]
    batch = torch.utils.data.default_collate([sample])
    print(f'Loading pretrained Anchor-Helios; episode={args.episode} frame={sample["frame_index"]}', flush=True)
    model = instantiate(cfg.model)
    report = {'episode': args.episode, 'frame': sample['frame_index'], 'optimizer_step': False,
              'gradient_storage': 'checked_on_cpu_then_released', 'trained_policy': False}
    if not args.skip_backward:
        model.eval().requires_grad_(False)
        model.dit.train().requires_grad_(True)
        model.proprio_encoder.train().requires_grad_(True)
        gradients = {}
        hooks = []

        def gradient_check(name):
            def check(parameter):
                grad = parameter.grad.detach().cpu().float()
                if not torch.isfinite(grad).all():
                    raise ValueError(f'Nonfinite gradient: {name}')
                gradients[name] = {'numel': grad.numel(), 'norm': float(grad.norm())}
                parameter.grad = None
            return check

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                hooks.append(parameter.register_post_accumulate_grad_hook(gradient_check(name)))
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        loss, metrics = model.training_loss(batch)
        # Convert only at this diagnostic output boundary.
        metrics = {key: float(value.detach()) if torch.is_tensor(value) else float(value)
                   for key, value in metrics.items()}
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        print(f'Forward passed: loss={float(loss.detach())}; backward starting', flush=True)
        loss.backward()
        torch.cuda.synchronize()
        for hook in hooks:
            hook.remove()
        for suffix in ('patch_long.weight', 'patch_mid.weight'):
            matches = [value for name, value in gradients.items() if name.endswith(suffix)]
            if len(matches) != 1 or matches[0]['norm'] <= 0:
                raise ValueError(f'Missing/nonpositive gradient for {suffix}: {matches}')
        report.update(loss=float(loss.detach()), metrics=metrics, gradient_tensors=len(gradients),
                      gradient_elements=sum(x['numel'] for x in gradients.values()),
                      memory_gradients={k: v for k, v in gradients.items() if 'patch_long' in k or 'patch_mid' in k},
                      backward_seconds=time.monotonic()-started,
                      peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        del loss, batch
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        (output / 'backward_smoke.json').write_text(json.dumps(report, indent=2) + '\n')
        print('Complete backward passed; action inference starting', flush=True)
    model.eval().requires_grad_(False)
    report['inference'] = []
    for decision in (0, 1):
        sample = dataset[decision]
        prefix = episode_video(dataset.root, dataset.info, dataset.episodes[0],
                               list(range(0, decision*16+1, 4))).unsqueeze(0).to(model.torch_dtype)
        with torch.no_grad():
            pred = model.infer_action(prompt=None, input_image=prefix, action_horizon=16,
                                     proprio=sample['proprio'][0], context=sample['context'],
                                     context_mask=sample['context_mask'], full_kv_frame_index=decision,
                                     num_inference_steps=2, seed=42)
        action = pred['action']
        if action.shape != (16, 14) or not torch.isfinite(action).all():
            raise ValueError('Invalid action inference')
        report['inference'].append({'decision': decision, 'action_shape': list(action.shape),
                                    'history_tokens': pred['full_kv_video_tokens'], 'denoising_steps': 2})
    if args.save_checkpoint:
        checkpoint = output / 'step_000000_smoke_only.pt'
        print(f'Saving step-0 smoke checkpoint: {checkpoint}', flush=True)
        model.save_checkpoint(checkpoint, step=0)
        model.load_checkpoint(checkpoint)
        deployment = {'env_cfg_type': 'arx_x5', 'action_type': 'joint',
                      'config_path': str((output / 'config.yaml').resolve()),
                      'checkpoint_path': str(checkpoint.resolve()),
                      'prepared_root': str(dataset.prepared_root), 'device': 'cuda:0',
                      'num_inference_steps': 2, 'seed': 42}
        (output / 'deployment.json').write_text(json.dumps(deployment, indent=2) + '\n')
        report['checkpoint_roundtrip'] = 'strict_load_passed'
        report['checkpoint'] = str(checkpoint)
    (output / 'model_smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
