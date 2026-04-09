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
- `execution.mode` currently executes sequentially; non-sequential values are not yet implemented.

## Next Technical Step

Implement true parallel combo execution in `src/collect_mpc_multimap.py` using `execution.max_workers`.

Target outcome:
- run multiple map/direction combos concurrently,
- preserve per-combo config/output isolation,
- keep manifest logging and failure accounting stable.

## Validation Gate

After any core collection/orchestration change, run:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_multimap.py --help"
```

And run at least one real collection command against a YAML config before declaring the change complete.
