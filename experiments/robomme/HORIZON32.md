# RoboMME predict32 / execute16 variants

The history clock and demo/execution VAE reset are unchanged. All new variants
predict 32 native joint actions, supervise two future video latents, and return
only the first 16 actions to the existing evaluator. The execution clock remains
16 observations per policy call. Video targets are the 9 sampled frames at
offsets 0,4,...,32; action labels use offsets 1,...,32. Tail labels are loss-masked.
Action RoPE uses two consecutive 16-step intervals, preserving i/17 within each.
Future video is a training branch and is omitted during deployment.

| Selector | Third expert | Long patch |
| --- | --- | --- |
| `past32` | Existing past 16 queries × 8 adjacent transitions | `[4,8,8]` |
| `pastfuture32` | Past 16 × 8 + future 2 × 8 | `[4,8,8]` |
| `pastfuture32_light` | Past 16 × 8 + future 2 × 8 | `[2,8,8]` |

Set `ROBOMME_LONG_COMPRESSION=spatial` for the light variant's `[4,4,4]`
alternative. The visual window stays `[96,2,1]`, with two fixed anchors.
The corresponding history token counts at latent size 24×20 are 606, 822, 1110.

As of 2026-10-22, new RoboMME recipes default to
`model.helios_config.anchor_mode=demo_execution_first`. The two fixed slots are
the first demo latent and the first execution latent, selected from the full
phase-separated prefix before cropping long99. They remain fixed throughout
execution. No-demo episodes use a zero demo slot masked out as a key in every
expert, including cached inference. The two slots retain their existing RoPE
positions 0/1; no extra type embeddings or extra visual tokens are introduced.
VAE reset at the demo/execution boundary is unchanged; cached training latents
and DreamDojo labels are reused.

Old saved configs/checkpoints without `anchor_mode` retain `global_first2`.
For a legacy-anchor control or an old training-state resume, explicitly pass
`model.helios_config.anchor_mode=global_first2`; dataset configuration follows
this override. Loading a checkpoint with different anchor semantics is rejected.

## Training

Submit **one command**, select four Volc workers with eight GPUs each:

```bash
bash /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison/experiments/robomme/train_horizon32.sh past32
# Second experiment:
bash /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison/experiments/robomme/train_horizon32.sh pastfuture32
# Third experiment:
bash /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison/experiments/robomme/train_horizon32.sh pastfuture32_light
```

Submit these as separate experiments. Defaults: 32 GPUs, bs16/GPU, GA1,
global batch 512, 20k steps, checkpoint every 2.5k. The established launcher
sets the code root and training venv, reads W&B/TOS credentials from
`runtime/memory_ablation_train.env`, and uploads after successful training.
Only node 0 runs the upload. No credentials are embedded in these files.
`ROBOMME_LAUNCH_DRY_RUN=1` prints the command without training or uploading.
For one 8-GPU worker, prefix `ROBOMME_TRAIN_GPUS=8`; bs16/GA1 then gives global
batch 128 (set GA4 to match global batch 512). Other existing launch overrides
such as `BATCH_SIZE`, `GRAD_ACCUM`, `MAX_STEPS`, `SAVE_EVERY` remain available.

## Third-expert target and attention

Each third-expert token adds a projected noisy 8×64 block to query, phase,
endpoint age, and interval-length features. The past/future variant additionally
has a direction embedding. Past slots use the original long99 temporal positions;
future slots are at +1 and +2 visual intervals. Future blocks sample destination
offsets 2,4,...,16 and 18,20,...,32 from the existing stride-1 DreamDojo cache.
These are adjacent-frame codes, not newly encoded stride-2 pairs.

Readers → keys (H includes anchors/current):

| Reader | H | noisy video F | past P | future Q | action A |
| --- | --- | --- | --- | --- | --- |
| H | yes | no | no | no | no |
| F | yes | yes | no | no | no |
| P | yes | no | yes | no | no |
| Q | yes | no | yes | yes | no |
| A | yes | no | yes | yes | yes |

P/Q are a single 18-token Transformer expert. Cache H once; recompute all P/Q
layer-aligned KV at each flow step. P, Q, A share scheduler time and Euler delta,
use independent noise, and update simultaneously from velocities evaluated at
the old states. No DreamDojo encoder or teacher labels are used at inference.

Future queries are always present. Episode length never enters their position,
length, attention mask, or validity metadata. Missing terminal labels affect only
the loss; their input remains noise instead of exposing clean padding. Past and
future losses each average over valid transitions per sample, then nonempty
samples; weights default to 0.1 each. Override the latter with
`model.helios_config.past_action.future_loss_weight=...` on past/future variants.

The 16×8 past targets are **16 Transformer tokens**, not 128 tokens: each noisy
8×64 block is flattened and projected once to hidden size 1024. Future adds two
tokens with the same shape. The action expert has one 8D target per token, hence
32 tokens for predict32 (16 for legacy predict16). Past/future contain at most
8192/1024 supervised scalars, versus 256 action scalars, but each loss is a
valid-element mean, not a sum. Counts therefore do not automatically multiply
branch weights. Keep past/future weights at 0.1 each initially; compare weighted
losses and gradients before choosing a larger or smaller auxiliary weight.

## Compatibility and verification

Existing 16-step configs/checkpoints retain their original contract. New
prediction horizons, target schemas, and compression profiles are checkpointed;
loading/resuming a different variant is rejected instead of silently changing
inference behavior. Switching selectors creates fresh independent experiments;
it does not convert optimizer state from another architecture/profile.

The existing `campaign.py` and `serve.py` use each run's saved config and enforce
its checkpoint contract. No special horizon flag is needed for evaluation.

CPU/real-data checks:

```bash
source experiments/robomme/env.sh
python -m pytest -q tests/test_robomme_horizon32.py tests/test_robomme_horizon32_launch.py
python scripts/check_robomme_horizon32.py --output runtime/horizon32_data_check
```

Full-size GPU forward/backward, update, and demo/no-demo policy replay:

```bash
python scripts/smoke_robomme_past_action.py --task robomme_helios_pastfuture32_light_flow \
  --output runtime/horizon32_gpu_smoke --memory-gib 65
```

The smoke utility checks full-model gradients and a subset of FP32-master
optimizer updates. Its memory use is not a production training-batch estimate.
