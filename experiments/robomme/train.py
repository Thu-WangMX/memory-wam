"""Bind the data, VAE, normalization and phase clock before training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from fastwam.datasets.robodojo import sha256
from fastwam.memory.robomme_clock import (CONTRACT, LEGACY_ANCHOR_MODE,
    PHASE_ANCHOR_MODE, inference_contract, validate_anchor_mode)
from fastwam.memory.helios_history import validate_memory_variant
from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


def validate_recipe(cfg):
    memory = cfg.model.helios_config
    anchor_mode = validate_anchor_mode(memory.get('anchor_mode', LEGACY_ANCHOR_MODE))
    if cfg.data.train.get('anchor_mode', LEGACY_ANCHOR_MODE) != anchor_mode:
        raise ValueError('Dataset/model anchor modes disagree')
    view_layout = memory.get('view_layout', 'legacy')
    if cfg.data.train.get('view_layout', 'legacy') != view_layout:
        raise ValueError('Dataset/model camera layouts disagree')
    variant = memory.get('memory_variant', 'multiscale')
    validate_memory_variant(variant, memory.history_sizes, memory.patch_long,
                            memory.patch_mid, memory.anchor_frames)
    if (variant == 'full_history_dense'
            or list(cfg.data.train.history_sizes) != list(memory.history_sizes)
            or cfg.data.train.anchor_frames != memory.anchor_frames
            or (list(memory.history_sizes) == [96, 2, 1] and variant not in ('multiscale_long99', 'long99_less_temporal', 'long99_less_spatial'))):
        raise ValueError('RoboMME data/model memory layouts must match; use multiscale_long99 for [96,2,1]')
    if (cfg.eval_every != 0 or cfg.model.memory_backend != 'helios'
            or cfg.model.action_rope_spatial_mode != 'memorywam'
            or cfg.model.helios_config.observation_contract != CONTRACT
            or (cfg.model.proprio_dim, cfg.model.action_dit_config.action_dim) != (8, 8)):
        raise ValueError('Invalid RoboMME Anchor-Helios recipe')
    if os.getenv('ACCELERATE_USE_FSDP', '').lower() == 'true':
        raise ValueError('Use the provided ZeRO-1 launcher: generic FSDP export lacks Helios metadata')
    dataset = instantiate(cfg.data.train)
    horizon = memory.get('prediction_horizon', 16)
    if dataset.prediction_horizon != horizon or horizon not in (16, 32):
        raise ValueError('Dataset/model prediction horizons disagree')
    from fastwam.models.wan22.helpers.loader import _resolve_configs
    _, _, vae, _ = _resolve_configs(model_id=cfg.model.model_id,
                                   tokenizer_model_id=cfg.model.tokenizer_model_id,
                                   redirect_common_files=True)
    vae.download_if_necessary()
    vae_hash = sha256(vae.path)
    if vae_hash != dataset.manifest['metadata']['vae_model_sha256']:
        raise ValueError('Training VAE does not match the cached observation VAE')
    provenance = json.loads((dataset.prepared_root/'provenance.json').read_text())
    for name, digest in provenance['text_sha256'].items():
        if sha256(dataset.prepared_root/'text_cache'/name) != digest:
            raise ValueError('Prepared text asset changed')
    report = dict(benchmark='RoboMME', episodes=len(dataset.episodes), decisions=len(dataset),
                  semantic_tasks=sorted(dataset.task_episodes), vae_sha256=vae_hash,
                  stats_sha256=provenance['stats_sha256'],
                  assets_sha256=sha256(dataset.prepared_root/'provenance.json'),
                  phase_contract=CONTRACT, task_balanced=dataset.task_balanced)
    if dataset.sample_stride == 1:
        report['sampling'] = dict(
            schema_version=1, sample_stride=1, dataset_len=len(dataset),
            seed=int(cfg.seed), batch_size=int(cfg.batch_size),
            world_size=int(os.environ.get('WORLD_SIZE', os.environ.get('NUM_GPUS', '1'))),
            gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
            continuous_dataloader=bool(cfg.get('continuous_dataloader', False)),
            sampler_version='task_episode_frame_balanced_v1' if dataset.task_balanced else 'permutation_v1',
            probability='uniform_task_then_episode_then_start' if dataset.task_balanced else 'uniform_all_starts',
            cache=dataset.dense_cache.contract(),
            source_assets_sha256=report['assets_sha256'])
    if anchor_mode == PHASE_ANCHOR_MODE:
        report['anchor_contract'] = inference_contract(memory.anchor_frames, anchor_mode, view_layout)
    if view_layout != 'legacy':
        from fastwam.memory.robomme_views import view_spec
        report['view_contract'] = view_spec(view_layout)
    if horizon != 16:
        report.update(prediction_horizon=horizon, execution_horizon=16,
                      future_video_latents=horizon // 16)
    enabled = memory.get('past_action', {}).get('enabled', False)
    if enabled != hasattr(dataset, 'dreamdojo'):
        raise ValueError('Past-action model and DreamDojo dataset must be enabled together')
    if enabled:
        if (memory.past_action.get('target_mode') in ('uniform16_half8', 'past16_future2_half8')
                and (int(memory.past_action.get('past_query_slots', 16)) != dataset.past_query_slots
                     or int(memory.past_action.get('source_history_slots', 99)) != sum(dataset.history_sizes))):
            raise ValueError('Past-action dataset/model source or query counts disagree')
        if memory.past_action.get('target_mode', 'endpoint') != dataset.target_mode:
            raise ValueError('Past-action dataset/model target modes disagree')
        if memory.past_action.get('learning_mode', 'deterministic') != dataset.learning_mode:
            raise ValueError('Past-action dataset/model learning modes disagree')
        report['dreamdojo'] = dataset.dreamdojo_contract()
        existing = memory.past_action.get('label_contract')
        if existing is not None and dict(existing) != report['dreamdojo']:
            raise ValueError('Pinned DreamDojo label contract changed')
        with open_dict(cfg):
            cfg.model.helios_config.past_action.label_contract = report['dreamdojo']
    if memory.get('action_video_attention', 'history_only') == 'asv':
        if (memory.get('past_action', {}).get('target_mode') != 'past16_future2_half8'
                or horizon != 32 or memory.get('future_mask_aux', {}).get('enabled', False)):
            raise ValueError('ASV requires the RGB Push75 past/future flow recipe')
        init = memory.get('asv_init_checkpoint')
        if init:
            if cfg.get('resume'):
                raise ValueError('ASV weight initialization and resume are mutually exclusive')
            source = Path(init)
            if not source.is_file() or source.suffix != '.pt':
                raise ValueError('ASV initialization requires a portable weight checkpoint')
            source_cfg = OmegaConf.load(source.parents[2]/'config.yaml')
            for key in ('stats_sha256', 'assets_sha256', 'vae_sha256'):
                if source_cfg.robomme_provenance[key] != report[key]:
                    raise ValueError(f'ASV initialization data assets differ: {key}')
            with open_dict(cfg):
                memory.asv_source_data_fingerprint = source_cfg.model.helios_config.data_fingerprint
    if 'sampling' in report:
        behavior = OmegaConf.to_container(memory, resolve=True)
        behavior.pop('data_fingerprint', None)
        report['sampling']['model_memory_contract'] = behavior
        report['sampling']['episode_indices'] = [row['episode_index'] for row in dataset.episodes]
        report['sampling']['supervision_contract'] = {
            key: value for key, value in report.items() if key != 'sampling'}
    from fastwam.datasets.robomme_sampling_resume import validate_sampling_resume
    validate_sampling_resume(cfg.get('resume'), report.get('sampling'))
    fingerprint = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    with open_dict(cfg):
        cfg.robomme_provenance = report
        cfg.model.helios_config.data_fingerprint = fingerprint
    return report


@hydra.main(config_path='../../configs', config_name='train', version_base='1.3')
def main(cfg):
    report = validate_recipe(cfg)
    if int(os.getenv('RANK', '0')) == 0:
        print('ROBOMME_PREFLIGHT '+json.dumps(report), flush=True)
    if os.getenv('ROBOMME_PREFLIGHT_ONLY') == '1':
        return
    if os.getenv('ROBOMME_GPU_MEMORY_GIB'):
        import torch
        device = int(os.getenv('LOCAL_RANK', '0'))
        limit = float(os.environ['ROBOMME_GPU_MEMORY_GIB']) * 2**30
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(limit / total, 1.0), device)
    if int(os.getenv('RANK', '0')) == 0:
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        stats = Path(cfg.data.train.pretrained_norm_stats)
        target = output/'dataset_stats.json'
        if target.exists() and sha256(target) != sha256(stats):
            raise ValueError('Run normalization changed')
        shutil.copyfile(stats, target)
        (output/'robomme_contract.json').write_text(json.dumps(report, indent=2)+'\n')
    run_training(cfg)


if __name__ == '__main__':
    main()
