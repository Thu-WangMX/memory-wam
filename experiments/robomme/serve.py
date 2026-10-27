"""Serve the trained policy using the official RoboMME websocket interface."""
import argparse
import json
from pathlib import Path
import sys
from omegaconf import OmegaConf
from experiments.robomme.policy import AnchorHeliosPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--port', type=int, default=18176)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', required=True)
    p.add_argument('--memory-gib', type=float, help='CUDA allocator cap; set before loading the model')
    p.add_argument('--robomme-src', default='/mnt/vepfs02/output/kevin_wang/memorywam/robomme_native/src')
    a = p.parse_args()
    if a.memory_gib is not None:
        import torch
        total = torch.cuda.get_device_properties(0).total_memory
        limit = int(a.memory_gib * 2**30)
        if not 0 < limit <= total:
            raise ValueError('Invalid policy CUDA allocator budget')
        torch.cuda.set_per_process_memory_fraction(limit / total, 0)
        print(f'ROBOMME_POLICY_ALLOCATOR_LIMIT_GIB={a.memory_gib}', flush=True)
    sys.path.insert(0, a.robomme_src)
    from challenge_interface.server import PolicyServer
    policy = AnchorHeliosPolicy.from_checkpoint(a.config, a.checkpoint, num_inference_steps=a.steps, seed=a.seed)
    cfg = OmegaConf.load(a.config)
    contract = dict(policy.model._helios_inference_contract(),
                    semantic_tasks=list(cfg.robomme_provenance.semantic_tasks),
                    checkpoint=str(Path(a.checkpoint).resolve()),
                    data_fingerprint=policy.model.helios_config['data_fingerprint'],
                    denoising_steps=a.steps, seed=a.seed,
                    policy_allocator_limit_gib=a.memory_gib)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'inference_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    print('ROBOMME_HELIOS_READY', flush=True)
    PolicyServer(policy, host='127.0.0.1', port=a.port, metadata=contract).serve_forever()


if __name__ == '__main__':
    main()
