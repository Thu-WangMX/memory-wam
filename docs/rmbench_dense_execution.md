# RMBench dense execution

The original executor uses TOPP position/velocity trajectories with a variable
number of physics steps. It does not teleport joint positions.

Enable fixed-duration execution before running the normal evaluation script:

```bash
export RMBENCH_ACTION_EXECUTOR=dense
export RMBENCH_DENSE_SUBSTEPS=15
export RMBENCH_DENSE_STATE_SOURCE=drive_target
# For a control-only comparison, keep official physics:
unset RMBENCH_BATTERY_CONTACT_OFFSET RMBENCH_BATTERY_TABLE_FRICTION
bash scripts/eval_comparison.sh <your existing evaluation overrides>
```

Use a distinct output directory for every execution configuration. To restore
original execution, set RMBENCH_ACTION_EXECUTOR=legacy (also the default).
The settings are read once when the policy is initialized; existing processes
are unaffected. Training and action normalization are unchanged.

Each denormalized 14-D action is ordered left arm 6, left gripper, right arm 6,
right gripper. Dense execution commands the final arm position plus constant
velocity (target - start)/(substeps * actual scene timestep) at each substep.
Gripper drive commands are linearly interpolated. Physics, not qpos assignment,
determines the actual motion. Endpoint arrival is not guaranteed.

At dt=1/250, 15 substeps means 0.06 simulation seconds per waypoint (16.67 Hz).
Keep the current model's 16 predicted / 16 executed waypoints, or 0.96 simulation
seconds per complete chunk. This does not imply wall-clock real-time inference.
Success can terminate a waypoint early. Episode limits and video frames remain
indexed by policy waypoints, not physics substeps. History sampling is unchanged.

Reference: https://github.com/OpenBMB/SimpleMemVLA/blob/main/rmbench_sim/rmbench_env.py
Its get_*_arm_jointState helpers return drive targets, not measured qpos; the
same behavior is the default here. RMBENCH_DENSE_STATE_SOURCE=measured explicitly
uses get_*_arm_real_jointState for arm velocity calculations instead. Gripper
start remains the wrapper's gripper command state in both modes.

Validation: tests/test_rmbench_dense_executor.py checks 15 substeps, both arm
velocities, gripper ramps, success latching, step limits, video hooks, invalid
inputs, measured-state selection, and legacy dispatch. Full-policy rollout
success-rate validation must be done separately.
