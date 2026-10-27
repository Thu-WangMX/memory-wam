# RMBench memory ablations (2026-09-29)

Entry: `bash scripts/train_ninetask_memory_ablation.sh VARIANT`.
Defaults: 8 GPUs; anchor-only/dense19 BS4 GA4; full history BS1 GA16
(global batch 128 in all groups); 500 epochs,
1e-4 cosine, historical condition augmentation enabled, Wan + ActionDiT
initialization (no full RoboTwin initialization), weights every 5000 optimizer
steps and at the end, no optimizer-state checkpoint. All use the same nine-task
data, pooled normalization, stride16 decisions, stride4 RGB, continuous causal
VAE cache, horizon16 and replan16.

| Variant | Anchors | History including current | Projection | History tokens (24x20 latent) |
| --- | --- | --- | --- | --- |
| Original multiscale | 2 | 19 | (4,8,8)/(2,4,4)/(1,2,2) | 426 |
| anchor_only | 2 | current only | (1,2,2) | 360 |
| recent19_dense | 2 | same latest19 | all (1,2,2) | 2520 |
| full_history_dense | 2 | entire observed episode, minimum19 padded startup slots | all (1,2,2) | 120*(2+max(19,H)) |

The original 19 already includes current: it is 18 past decision latents plus
current, not 19 past plus current. Anchors are the same earliest causal latents
as before, not redefined RGB frames. They remain separate in every mode, even
when duplicated in the history prefix. Anchor-only retains current observation
and its original temporal RoPE index20; removing older tokens does not renumber
current/future. Dense19 retains the old temporal indices and independent long /
mid patch branches, but their kernels and strides become (1,2,2).

Full history extends dense19 beyond19 without eviction. Extra left padding for
batching is excluded as an attention key at every layer. Every example's video
and action RoPE is rebuilt at its own prefix length, so outputs do not depend
on other examples' lengths. Original startup zero slots are retained as in
dense19 (and are distinct from extra batching padding). The temporal RoPE table
retains the backbone's maximum supported range; no frame truncation is applied.

Masks are unchanged: clean memory/current sees only clean memory/current;
future sees memory and itself; action sees memory and action, never future.
This full-history mode recomputes a full-history prefill per decision and reuses
its per-layer KV during denoising. It does NOT incrementally append stale
Transformer KV across decisions. Bidirectional history requires recomputing
old features after adding a frame. Switching to the preserved `fullkv` backend
would change that experiment's semantics and is deliberately not done here.

Checkpoints store the variant and patch/history contract. Deployment restores
these structural fields before strict weight loading. Original checkpoints
without a variant retain their old contract and default behavior.

Memory: variable-length BS4 is supported, not silently micro-split. Full history
uses BS1 GA16 by user choice. The launcher enables
whole-joint-layer gradient checkpointing and tokenwise QKV/FFN chunking; this
changes activation storage/recomputation only, not the model's attention span.
Long-history runtime and memory are substantially larger than compressed
Helios. Validation artifacts are under `runtime/memory_ablation_validation_20260929`.

Controls: evaluate all variants with the same per-task valid seeds, denoising
steps, executor, and replan16. For strict training comparison, rerun the original
multiscale baseline with BS4/GA4 as well; equality of global batch does not make
all floating-point, data-order, or scheduling effects identical to old BS16/GA1.


Validation completed on 2026-09-29:
- 25 unit/contract checks passed (including full-batch padding isolation,
  per-example RoPE, cached inference equivalence and whole-layer gradient parity).
- Real pretrained model, real nine-task data (17,522 samples): forward/backward
  passed for anchor-only BS4 (24.39 GiB), dense19 BS4 (27.50 GiB),
  and full-history BS1 at153 latents (31.44 GiB peak allocated).
- These memory measurements exclude optimizer/distributed communication buffers.
  Full 8-GPU optimizer training has not been run as part of this validation.
- Hydra composition and shell syntax checked; all runs save weights each5000
  optimizer updates plus final. Expected budget is137 updates/epoch,68500 total.
- Fixed samples/sec reporting to include gradient accumulation.


## Additional layout ablations (2026-10-09)

Single-task entry: `bash scripts/train_memory_ablation_task.sh VARIANT TASK`.
Sequential entry: `bash scripts/train_memory_ablation_three_tasks.sh VARIANT`.
The latter trains press_button, swap_T, swap_blocks independently in that order.
Defaults for these four new variants: 8 GPUs, BS8, GA2 (global batch128),
500 epochs per task, 1e-4 cosine, history augmentation on,16 loader workers,
weights every2500 optimizer steps plus final, no optimizer state.
Swap T initializes the official complete RoboTwin checkpoint; Press Button and
Swap Blocks use the existing Wan + ActionDiT initialization.

| Variant | Long history | Mid history | Fixed anchors | Clean-prefix tokens at24x20 |
| --- | --- | --- | --- | --- |
| multiscale (baseline) |16 at(4,8,8)|2 at(2,4,4)|2 at(1,2,2)|426|
| wo_long |none|2 at(2,4,4)|2 at(1,2,2)|390|
| uniform_mid |16 at(2,4,4)|2 at(2,4,4)|2 at(1,2,2)|630|
| reverse_scale |oldest2 at(2,4,4)|nearest16 at(4,8,8)|2 at(1,2,2)|426|
| wo_anchor |16 at(4,8,8)|2 at(2,4,4)|none|186|

All counts include the current latent's120 tokens and exclude the future latent
and actions. The19-latent window is18 past +1 current. `uniform_mid` changes
scale only: Long/Mid remain independent learned Conv3d branches, not weight-tied.
`reverse_scale` uses history_sizes=[2,16,1]: oldest2 use(2,4,4), nearest16
use(4,8,8). Both temporal segments divide exactly by their temporal kernels,
so no temporal padding is needed. Spatial boundary padding follows the baseline.
The same18 past latents, anchors, current and426-token budget are retained.
Chronological order is preserved: oldest positions2..3, nearest4..19, current20.
The earlier experimental reverse profile[16,2,1] is rejected by validation rather
than silently interpreted as the new layout.
Current/future always use(1,2,2); anchors also remain dense when enabled.
`wo_anchor` removes the fixed prefix only: early observations can still appear
naturally in the rolling19-latent window until they age out.

Removing Long or anchors preserves the baseline RoPE indices of every retained
segment: wo_long history offset16; wo_anchor offset2. Current remains20,
future21; action RoPE remains aligned to current. The variant determines this
offset, so training and checkpoint-restored deployment cannot silently disagree.
The baseline's bidirectional clean-history mask remains unchanged. Clean tokens
cannot read future/actions; action reads clean tokens and actions, not future.
Removed Conv3d branches are frozen both at model creation and after the trainer
re-enables DiT gradients, avoiding unused trainable parameters under ZeRO.

The shared registry is `src/fastwam/memory/helios_history.py`; Bash reads it via
`scripts/memory_variant_args.py`. Checkpoints persist variant, sizes, patch
kernels, anchors and augmentation. Existing deployment restores these fields
before model construction and strict checkpoint loading. Old variants retain
their previous contracts, kernels and temporal coordinates.

Validation:64 CPU unit/contract tests passed, including startup lengths1/2/3,
window boundaries18/19 and longer histories, mask isolation, prefill equivalence,
backward, checkpoint round-trip, unchanged retained RoPE and legacy regression.
All4 variants x3 task launcher configurations were composed and checked.
Real-model validation reports: `runtime/memory_layout_ablations_20261009/`.

Cloud queue entry (sources the existing private credential env file):
`bash scripts/train_memory_layout_ablation.sh VARIANT`.
Use one of wo_long, uniform_mid, reverse_scale, wo_anchor per8-GPU queue.
Explicit BS/GA/epochs/save/workers/LR/scheduler/noise environment overrides are
preserved when sourcing credentials. The shell command itself contains no keys.

Real pretrained-model validation also passed for all4 new variants at BS8:
wo_long on Swap T (official RoboTwin initialization), uniform_mid and
the initial reverse_scale profile on Press Button, wo_anchor on Swap Blocks. Forward/backward
peak allocated memory was24.65-25.80 GiB; this excludes optimizer states and
8-GPU distributed buffers. These were validation batches, not full training
or simulator success-rate measurements. No large checkpoints were created.


Task-group split (2026-10-09): pass `--task-group press_swapT` to train
Press Button then Swap T, or `--task-group swap_blocks` to train only Swap Blocks.
Apply both groups independently for each of the four variants (eight8-GPU jobs).
Omitting the option retains the previous all-three-tasks entry for compatibility.
Example: `bash scripts/train_memory_layout_ablation.sh wo_long --task-group press_swapT`.
All other hyperparameters, per-task data and initialization rules are unchanged.

The revised reverse2+16 profile is checked separately in
`runtime/reverse_2_16_validation_20261009/`.

Revised reverse2+16:65 unit/contract tests and both split-queue configuration
checks passed. Both dataset and model receive history_sizes=[2,16,1].

Revised reverse2+16 real pretrained-model BS8 forward/backward also passed on
Press Button:426 clean-prefix tokens,25.37 GiB peak allocated memory. This
excludes optimizer states and distributed buffers; it is not a full training run.
See runtime/reverse_2_16_validation_20261009/real_validation.json.


Long compression ablations (2026-10-09):
- long_less_temporal: Long16 uses (2,8,8), producing 8x3x3=72 tokens.
- long_less_spatial: Long16 uses (4,4,4), producing 4x6x5=120 tokens.
Both retain Mid2=(2,4,4), Current1=(1,2,2), two fixed anchors and the
same history window/chronological coordinates/current20/future21/mask.
At latent spatial size24x20, total clean prefix lengths are462 and510
respectively, versus426 for baseline. They increase the token budget; this
is a compression-resolution ablation, not an equal-compute comparison.
Both support train_memory_layout_ablation.sh with --task-group press_swapT
or --task-group swap_blocks, preserving the existing task initialization rules.
Checkpoint contracts store and strictly restore each variant's actual kernels.
Validation artifacts: runtime/long_compression_validation_20261009/.

Long compression validation completed:89 unit/contract tests passed; all four
split-queue commands (six task configurations) passed. Real pretrained-model
BS8 forward/backward passed for both profiles, with462/510 clean tokens and
25.42/25.52 GiB peak allocated memory respectively. These batch checks exclude
optimizer/distributed memory and do not constitute an8-GPU training run.

All six layout ablation variants now default to BS16/GA1 per GPU (2026-10-09).
Both the layout entry and single-task defaults agree. Global batch remains128
on8 GPUs; epochs, save intervals, initialization and credential exports are unchanged.
Explicit BATCH_SIZE/GRAD_ACCUM environment overrides remain supported.
