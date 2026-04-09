# COPILOT.md

Current working guide for coding agents in this repository.

## Scope

This repository is currently focused on a minimal MPC-to-imitation workflow:

1. run MPC experts in Gym-Khana,
2. collect trajectory datasets,
3. train downstream sequence models outside or alongside this repo.

Out of scope unless explicitly requested:
- PPO/RL pipelines
- legacy training/test/CI surfaces removed in cleanup

## Canonical Runtime

Container-first execution is the default:
- `docker compose run --rm app ...`
- run from `/app` with `PYTHONPATH=/app`

## Canonical Entry Points

- `src/collect_mpc_data.py`: single-map MPC data collection
- `src/collect_mpc_multimap.py`: map/direction sweep orchestration
- `src/kmpc_race.py`: visual kinematic MPC run
- `src/stmpc_race.py`: visual single-track MPC run

## Canonical Configs

- `configs/collect_mpc_default.yaml`
- `configs/collect_mpc_test.yaml`
- `configs/collect_mpc_multimap_fullscale.yaml`

## Important Current Behavior

- `src/collect_mpc_multimap.py` now supports `collector_overrides` via deep merge into `base_collector_config`.
- `run.stop_on_error: false` means failed combos are skipped and the run continues.
- `execution.mode: parallel` with `max_workers: N` runs up to N map/direction combos concurrently.
- When parallel mode is enabled, acados MPC solvers are automatically pre-compiled sequentially before collection starts (automatic, no manual steps).
- `lidar` collection is integrated: when `lidar.enabled=true`, scans are extracted (360 beams), clipped [0, 15.0m], binned (360→60 via min-pooling), and stored as float16 NPZ arrays.

## Next Technical Steps (Priority Order)

1. **Integrate LSTM training notebook**: Load NPZ datasets and train sequence models on (state, lidar_scan) → action mapping.
2. **Parallelize across GPUs**: If multiple GPUs available, extend `max_workers` or use GPU-aware scheduling in collect scripts.
3. **Add model-agnostic observation transformations**: Support other sensor modalities (camera, radar) via pluggable lidar-style extraction functions.

## Validation Gate

After any core collection/orchestration change, run:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_multimap.py --config /app/configs/collect_mpc_multimap_test.yaml"
```

Test config uses 3 maps × 2 directions × 2 episodes for quick turnaround (~15 min).
