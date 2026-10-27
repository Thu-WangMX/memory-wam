# RoboMME long99 + ordered past-action flow

Switchable tasks:

| Hydra task | P input | P objective |
| --- | --- | --- |
| `robomme_helios_past_action` | Time/phase queries | One adjacent-frame DreamDojo code at each decision endpoint |
| `robomme_helios_past_action_block16` | Time/phase/interval-length queries | Up to 16 ordered adjacent-frame codes per decision interval |
| `robomme_helios_past_action_flow` | Same queries plus a noisy ordered block and diffusion time | Flow velocity for the ordered block |
| `robomme_helios_past_action_sparse16x8_flow` | 16 sampled interval queries plus noisy half-rate blocks and diffusion time | Flow velocity for up to 8 adjacent transitions per selected interval |

All retain visual history `[96, 2, 1]` and two anchors. The original three tasks use
99 P Transformer tokens. Their interval head outputs `[B,99,16,64]`; it does not
expand attention to 1,584 P tokens. The sparse variant uses 16 P Transformer tokens
with a `[B,16,8,64]` head, at most 128 supervised transition codes.
Each 64D target concatenates the head and wrist DreamDojo `z_mu`, using the existing
frozen encoder's stride-1 cache and pinned per-dimension normalization.

## Uniform 16 by 8 variant

The shared offline/online layout first constructs the original long99 interval
clock, excluding intervals with no valid transition (initial frame, padding, and
the demo/execution jump). If more than 16 intervals remain, select interval ranks
with `rint(linspace(0, N-1, 16))`, retaining the earliest and latest available
intervals. Demo and execution share this budget; no phase quota or semantic label
is used. Short histories retain all available intervals without duplication and
left-pad unused query slots.

Inside each selected interval, take every second valid transition backwards
from the endpoint, then restore chronological order. A length-16 interval keeps
offsets 2,4,...,16; a length-3 interval keeps offsets 1,3. Short outputs are
left-aligned and tail-masked. Each selected code remains an adjacent-frame
`E(i-1,i)` label from the existing cache: frames are not re-encoded at stride 2.

The original interval duration, endpoint age, phase and original long99 visual
slot are preserved. Batched RoPE uses the original visual slot, not sparse query
rank 0..15. Visual history and anchors are unchanged. P/A flow time, independent
noise, attention, per-valid-transition loss normalization and joint Euler updates
are unchanged. Sparse P input/output projections have width 512 rather than 1024.

The new target schema and sampling contract are saved in checkpoints and checked
at training/deployment. The old 99-query checkpoint/state cannot be resumed as
this variant. Keep old launches for old checkpoints; the new entry starts fresh.

Sparse launchers:

```bash
# One worker with eight GPUs; default bs16, GA1, 20k steps, save every 2.5k.
bash experiments/robomme/train_past_action_sparse16x8_flow_8gpu.sh

# Submit once on Volc with four workers of eight GPUs each.
# Default bs16, GA1, global batch 512, 20k steps, save every 2.5k.
bash experiments/robomme/train_past_action_sparse16x8_flow_32gpu.sh
```

Both use the existing training environment and W&B/TOS setup. With
`ROBOMME_LAUNCH_DRY_RUN=1` they print the distributed command without training.
No label extraction or cache replacement is needed.

## Ordered labels

For consecutive visual decisions at raw frames 32 and 48, the latter query targets
`E(32,33), E(33,34), ..., E(47,48)` in that order.
Short intervals are left aligned and tail masked. Initial frames, left padding,
and the demo-to-execution transition do not contribute to loss or P attention keys.
The predecessor is determined before cropping to the last 99 visual positions.
No target ends after the currently observed frame. Demo and execution VAE streams
remain separate; their observed latent prefixes remain in the shared memory clock.

## Flow definition

The P and A branches share the action scheduler's sampled time:

```
sigma = timestep / num_train_timesteps
P_sigma = (1 - sigma) * normalized_z + sigma * independent_gaussian_noise
target_velocity_P = noise - normalized_z
```

All invalid P coordinates are zero before input projection, in the velocity target,
and after each inference update. The ordered block is flattened only for learned
input/output projections: width 1024 for the full block, or 512 for sparse16x8.
The existing independent Transformer expert receives time modulation, phase,
relative endpoint age, and observed interval length.

P loss averages MSE over 64 dimensions, then over valid transitions per example.
Nonempty examples receive the same scheduler time weighting as actions and are
averaged; default P loss weight is 0.1. Empty examples contribute zero.
Logged `past_action_velocity_mse_demo/execution` measure velocity error, not
clean-code reconstruction error.

Unlike the deterministic variants, FM training DOES use corrupted teacher labels
as the P diffusion state. This is explicitly recorded in the checkpoint contract.
There is no clean-teacher-only action conditioning and no teacher encoder/cache at
inference. The method is a joint conditional flow model, not a deterministic
label-free reconstruction forward.

## Attention and inference

Let H be observed visual memory, F the training future-video branch, P past-action
flow tokens, and A action flow tokens. A row indicates a reader:

| reader | H | F | P | A |
| --- | --- | --- | --- | --- |
| H | yes | no | no | no |
| F | yes | yes | no | no |
| P | yes | no | valid keys | no |
| A | yes | no | valid keys | yes |

P reads language but not action-only proprioception. `PAST_ACTION_READ=false`
disables A reading P for the auxiliary-only control.

Inference:

1. Prefill observed H once.
2. Draw independent A and P Gaussian states, using the requested generator seed.
3. At each shared scheduler time, run P against H and construct its layer-aligned
   KV; run A against H and that P KV.
4. Compute BOTH velocities from the old states, then update BOTH with the same
   Euler delta. Repeat for `num_inference_steps`.

P KV is recomputed each noise step, never reused from the preceding time.
P does not need a separate outer denoising loop. It adds one P forward per action
denoising step compared with the deterministic version. F is omitted at inference.
Returned `past_action_prediction` is the final normalized P state, not velocity.

## Launch

Submit once on a worker with eight GPUs:

```bash
ROBOMME_LAUNCH_DRY_RUN=0 bash \
  /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison/experiments/robomme/train_past_action_flow_8gpu.sh
```

The wrapper sets the code directory and training venv, sources the existing
W&B/TOS credential environment, uses all 16 tasks, and defaults to:
8 GPUs, batch 16/GPU, gradient accumulation 4, global batch 512,
20,000 optimizer steps, save every 2,000, bf16, persistent workers, continuous
dataloader, seek decoding, and whole-layer checkpointing.
Checkpoint upload runs after successful training through the existing uploader.
The script starts a fresh run. It does not resume a deterministic checkpoint/state.
Experiment knobs such as `BATCH_SIZE`, `GRAD_ACCUM`, `MAX_STEPS`, `SAVE_EVERY`,
`PAST_ACTION_LOSS_WEIGHT`, and `PAST_ACTION_READ` remain overridable.

The endpoint and deterministic block launchers are respectively
`train_past_action_8gpu.sh` and `train_past_action_block16_8gpu.sh`.
Model, data, label, and inference contracts reject incompatible resume/evaluation.

## Verification

Core tests exercise ordered coverage, short intervals, phase boundaries, rollover,
loss masks, noise direction, independent parameters, gradients, empty histories,
checkpoint compatibility, and multi-step full-joint/cached inference parity.

The real-model smoke utility accepts `--task robomme_helios_past_action_flow`.
It audits labels in all 16 tasks, checks all model gradients, performs a small
FP32-master optimizer update, and replays demo/no-demo inputs through the policy.
It releases large gradients during the smoke step; its peak memory is not a
measurement of full optimizer training capacity at the default batch size.
