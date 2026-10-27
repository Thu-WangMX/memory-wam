# Training and evaluation updates (2026-09)

## Training

- `scripts/train_observe_nohistnoise.sh`: Observe-only training with official
  RoboTwin initialization, clean history conditioning, and configurable
  `LEARNING_RATE` / `LR_SCHEDULER` (defaults: `0.0001`, `cosine`).
  The trainer already supported cosine; these launchers expose the experiment.
- `scripts/prepare_ninetask_helios.py`: combine references to nine existing
  LeRobot datasets, continuous latent caches and text caches without copying
  or modifying source data. `scripts/validate_ninetask_helios.py` checks loading
  and collation across the nine tasks.
- `scripts/train_ninetask_helios.sh`: one mixed-data run, shared normalization,
  explicit `HISTORY_AUGMENTATION=true|false`, and no full RoboTwin initialization.
- `scripts/distributed_env.sh`: single-node and multi-node launch topology;
  `NUM_GPUS` is global, `GPUS_PER_NODE` is per node. Nodes share a run ID.
  Only node 0 uploads to TOS; each node writes a separate training log.
- `NUM_WORKERS` defaults to 16 per GPU; `SAVE_EVERY` configures checkpoint cadence.
- `RESUME_CHECKPOINT` in the mixed-task launcher accepts a portable weight file
  or training-state directory. Portable weights reset optimizer, scheduler and
  data progress; the requested epochs run again. Full-state loading follows
  Accelerate/DeepSpeed restrictions and is not a verified cross-world-size resume.
- Training state saving remains disabled by default in `rmbench_helios`.
  W&B/TOS credentials must be supplied through environment variables, never Git.

## Evaluation

- Optional fixed-duration dense action executor; see `rmbench_dense_execution.md`.
  Legacy execution remains the default.
- Restore Helios conditioning augmentation metadata from each checkpoint before
  validating its model contract, so no-history-noise checkpoints can be evaluated.
- Fixed scene-seed lists, scene diagnostics and shared evaluation queue support;
  see `../scripts/RMBENCH_DYNAMIC_QUEUE.md`.
- Queue policy `denoise50_then_top2=true` first completes all 50-step groups,
  then runs 10-step groups for the best two checkpoints (ties: later checkpoint).
  Keep `checkpoint_ordered=false` to parallelize different checkpoints. Existing
  completed seeds are retained. The queue currently targets 100 episodes/group.
- `TORCH_EXTENSIONS_DIR` can be overridden for machine-specific compiled caches.

Training/evaluation environments, simulator assets, datasets, checkpoint files,
queue credentials, logs and audit outputs are local dependencies, not included.
Existing launch scripts still contain deployment-specific paths; adjust those
paths when moving to another machine.
