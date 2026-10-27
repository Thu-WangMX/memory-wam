# Every-frame RoboMME training starts

The audited 1,600 episodes now support 475,257 execution starts rather than
30,470 starts on the original stride-16 grid. A start t supervises a[t+1:t+33]
for predict32/execute16. The terminal observation has no future action and is
excluded; short tails keep the existing padding/loss masks.

`sample_stride=1` requires supplemental causal VAE caches for execution clock
offsets 1..15. For offset r>0, endpoints are `[execution_start,
execution_start+r, execution_start+r+16, ...]`. The first interval uses the
same phase-local linspace sampling as short demo intervals. Subsequent intervals
are 16 raw frames. Each stream has independent causal encoder state. The demo
and true execution-first anchor are shared from the original cache. The current
latent ends exactly at t; no floor/ceil substitution or future observation is
allowed. Offset zero uses the original cache exactly.

This is training clock-phase augmentation. Online control still observes every
frame and replans/executes 16 actions on its original offset-zero stream.
Prediction horizon, compression, phase anchors, losses, noise, and network
architecture do not change. Existing DreamDojo stride-one labels are reused.
Native RGB/action preprocessing remains unchanged in this patch.

## One-time cache preparation

Run on one machine; the script supports resumable per-episode/offset shards.
Only our output directory is written. Kevin's source data is read-only.
Use selected GPUs with spare capacity (default GPU 0, allocator cap 6 GiB).

```bash
ROBOMME_PREP_GPUS=0,1,2,3,4,5,6,7 bash \
  /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison/experiments/robomme/prepare_dense_latents.sh
```

Default output:
`/mnt/vepfs01/output/spidy.wang/runtime-robomme/phase-offset16-latents-v1`.
The finalized manifest is written only after every source/payload checksum,
clock, shape and provenance check passes for all 1,600 episodes x 16 offsets.
Dataset construction refuses incomplete caches. Additional latent payloads are
approximately 19.2 GiB (446,273 additional latents), plus serialization/metadata
overhead; no demos are duplicated.

## Training

Select 4 workers x 8 GPUs on Volc and submit this command once:

```bash
bash /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison/experiments/robomme/train_dense_horizon32.sh past32
```

The inherited recipe is BS16/GPU, GA1, 20,000 steps, save every 2,500. Replace
`past32` with `pastfuture32` or `pastfuture32_light` to select either other existing
variant. `ROBOMME_TRAIN_GPUS=8` selects the inherited single-node launcher.
The launchers use the existing environment, model assets, W&B and TOS setup;
no credentials are embedded in this new entry.

Any RoboMME dataset can select the new mode with `ROBOMME_SAMPLE_STRIDE=1` and
`ROBOMME_DENSE_LATENTS=<finalized cache>`. Set `ROBOMME_SAMPLE_STRIDE=16` to return
to the original sampling grid. The base config defaults to 16 so old commands,
checkpoints and already running jobs are unaffected.

The probability remains uniform task -> uniform episode -> uniform start.
Every frame is eligible; a balanced epoch is a reproducible draw with replacement,
not a guarantee that all starts appear exactly once. At global batch 512 and
GA1 the continuous sampler has ceil(475257/512)=929 steps per virtual epoch;
20k steps are about 21.53 such epochs. Keep step count fixed for comparisons.

Full-state resume requires the same sampling/cache, seed, batch/world/GA and
loader contract. An old stride-16 state must not silently resume its epoch/batch
cursor on this larger dataset. The snapshot is stored in trainer_state.json.
Portable weights retain the existing model checkpoint compatibility checks.
