# RMBench shared evaluation queue

Coordinator: `rmbench_eval_queue.py`; SQLite stays on local disk, workers communicate over authenticated HTTP.
Workers: `rmbench_queue_worker.py`; one supervisor and one model per GPU. After a group finishes, GPU can pick another unfinished group.
Integration: `third_party/RoboTwin/script/rmbench_queue_client.py`, enabled only by RMBENCH_QUEUE_URL/TOKEN/GROUP.

Group identity includes task and checkpoint; launch manifest also fixes inference settings. Completed imported results are deduplicated by scene seed. Each group targets 100 completed episodes, including policy successes and failures. Expert validation failures consume no completion quota; runtime errors are retried and must not be called invalid scenes.

Leases last 300 seconds. Heartbeats every 20 seconds require recent forward progress; 240 seconds without progress terminates the worker. Supervisor retries up to three errors per group, and stops a worker whose log stalls for 900 seconds. Exhausted retries remain incomplete, never fabricated as policy failures. All completion commits check lease tokens. Duplicate commits are idempotent.

Fixed seed candidates are preferred; after invalid candidates are exhausted, additional seeds are issued in ascending order. Different checkpoint groups may therefore complete on different seed sets: compare common seeds as well as aggregate success rate.

`eval_launch_logs/dynamic_queue_latest.json` locates the live run; coordinator credential config remains local and private. Shared run directory contains per-GPU logs and status snapshots. `seed_records.json` publishes recovery records every 30 seconds. Do not directly open the live database on a shared mount. Already running legacy workers do not hot-reload queue logic.

With `denoise50_then_top2=true` and `checkpoint_ordered=false`, finish every 50-step group first, rank completed success rates, then evaluate only the top two matching 10-step groups. Ties favor the later checkpoint. Already completed groups are not repeated. Queue configs and logs under `eval_launch_logs/` are excluded from Git.
