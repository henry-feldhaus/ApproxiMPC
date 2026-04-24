# Current Context

## Active Goal

Compare one LSTM controller run and one MPC controller run on the same Gymkhana map using a shared `.npz` schema, then debug why the LSTM baseline is underperforming.

## Current Baseline

- Main workflow note: `COMPARISON_WORKFLOW.md`
- Default LSTM eval config: `configs/lstm_eval_default.yaml`
- Current default map: `Spielberg`
- Current track direction: `normal`
- Current lidar bin count: `60`
- Current path lookahead: `[1.0, 2.0, 3.0, 5.0, 8.0]`
- Canonical runtime is container-first from `/app` with `PYTHONPATH=/app`
- Canonical collection entry points:
  - `src/collect_mpc_data.py`
  - `src/collect_mpc_multimap.py`
  - `src/kmpc_race.py`
  - `src/stmpc_race.py`

## What Was Just Completed

- Added a repo-local notes area under `.codex-notes/`.
- Added and documented the LSTM-vs-MPC comparison workflow.
- Added `src/convert_mpc_to_lstm_log.py` and `configs/mpc_compare_default.yaml`.
- Added `configs/collect_mpc_first_comparison.yaml` for one-off comparison collection runs.
- Completed the first end-to-end comparison pass:
  - LSTM eval log produced successfully
  - MPC dataset produced successfully
  - MPC dataset converted into the shared comparison schema
- Updated the Docker image so the dev container includes ONNX/LSTM runtime dependencies.

## Immediate Next Step

Debug why the LSTM rollout is much slower and weaker than the MPC baseline.

Highest-priority checks:

1. verify feature ordering and scaling assumptions at runtime,
2. confirm the scaler compatibility warning is not distorting inference,
3. check start pose / startup speed / action clipping behavior,
4. confirm the path-feature contract matches the trained model.

## Notes

- Current output layout for the first comparison pass:
  - raw MPC dataset: `outputs/mpc_set/stmpc_Spielberg_first_comparison.npz`
  - converted MPC comparison log: `outputs/mpc_eval_logs/*.npz`
  - LSTM comparison log: `outputs/lstm_eval_logs/*.npz`
- Shared schema between the LSTM and MPC comparison logs is now confirmed.
- Important multimap behavior:
  - `src/collect_mpc_multimap.py` supports `collector_overrides` via deep merge into `base_collector_config`
  - `run.stop_on_error: false` skips failed combos and continues
  - `execution.mode: parallel` with `max_workers` runs combos concurrently
  - parallel combos use isolated acados build tags to avoid codegen/link collisions
- Known follow-ups:
  - `MinMaxScaler` pickle version mismatch warning
  - Gym reset observation-space warning during LSTM eval
  - LSTM failed to complete a lap and hit the `6000`-step cap
