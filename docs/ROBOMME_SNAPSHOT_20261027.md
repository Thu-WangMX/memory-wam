# RoboMME code snapshot — 2026-10-27

This repository contains the current FastWAM + anchor + Helios source snapshot,
including Push75, long444, 288, ASV, past/future latent-action flow matching,
dense sampling, compact history, video seek, persistent loading and exact future-clip caching.

## Entry points
- experiments/robomme/README.md: training, evaluation and data contracts.
- experiments/robomme/train_push75.sh: Push75 baseline.
- experiments/robomme/train_push75_444_16gpu.sh: long444 variant.
- experiments/robomme/train_push75_asv_8gpu.sh and train_push75_288_8gpu.sh.
- ops/robomme_server/: snapshot of deployed checkpoint monitoring and dual-host evaluation scripts. Future automatic evaluations default to 5 denoising steps; historic 10-step evaluations are separate.

## External dependencies and deployment
This is a source backup of the working deployment, not a self-contained dataset or environment.
The launchers retain the original absolute shared-storage paths.
Training environment: /mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv.
Training data: /mnt/vepfs02/output/kevin_wang/memorywam/data/robomme_all16_anchor1_event_k8_v1_full/lerobot/multitask.
Visual caches: /mnt/vepfs01/output/spidy.wang/runtime-robomme/two-view-square256-v1/{base,prepared,dense}.
Latent-action labels: /mnt/vepfs01/output/spidy.wang/runtime-robomme/dreamdojo_labels_v1/cache/head_wrist64_stride1.

Weights, datasets, virtual environments, evaluation videos, logs, server queue state and credentials are excluded.
Supply private W&B/TOS configuration in the deployment environment; runtime/memory_ablation_train.env is deliberately absent.
The ASV launcher requires its source 17.5k checkpoint.
ASV/288 launchers retain a deployment readiness gate at runtime/speed_optimization_20261026/READY.json; on a new machine review the validated installation and adapt this gate before launching, otherwise it waits.
Server scripts in ops/robomme_server retain their original runtime-robomme/launchers paths and are deployment references, not automatically started by cloning.
runtime/python contains locally installed dependencies in the original deployment and is not vendored here; install dependencies in your environment.
Consult the configured environment rather than assuming pyproject.toml reproduces every installed CUDA package version.

Existing A100 idle reservations are not an atomic GPU reservation mechanism. They are released at evaluation startup, so uninterrupted GPU ownership is not guaranteed.

## Verification
Python source and RoboMME shell syntax checks passed. 227 tests passed across views, dense sampling, long75/444 and WSD scheduling, using the deployment venv plus its runtime/python dependency directory. No full training run was launched for this backup.
