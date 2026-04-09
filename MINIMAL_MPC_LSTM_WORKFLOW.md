# Minimal MPC -> Sequence Model Workflow

This is the shortest supported path for generating expert driving data and training a sequence policy.

## 1) Validate MPC Runtime

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/kmpc_race.py"
```

Success criteria:
- simulator launches,
- controller tracks centerline stably,
- no acados/runtime errors.

## 2) Collect Single-Map Data

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_data.py --config /app/configs/collect_mpc_default.yaml"
```

Optional perturbation-enabled run:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_data.py --config /app/configs/collect_mpc_test.yaml"
```

## 3) Collect Multi-Map Training Data

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_multimap.py --config /app/configs/collect_mpc_multimap_fullscale.yaml"
```

This performs a map/direction sweep and writes a run manifest plus per-combo outputs.

## 4) Train LSTM (or other sequence model)

Train outside this document's scope using generated NPZ datasets in:
- `outputs/datasets/` (single-map)
- `outputs/datasets/multimap_training/` (multi-map)

Recommended split:
- train: most maps/directions,
- validation: held-out episodes on seen maps,
- test: held-out map/direction combos.

## Dataset Contents

Saved arrays include expert and executed actions with perturbation labels:
- observations/state vectors,
- expert actions,
- executed actions,
- perturbation flags,
- noise vectors (when enabled),
- rewards and termination indicators.

## Notes

- Keep `render: false` for large batch runs.
- `run.stop_on_error: false` is best for long unattended sweeps.
- Parallel multimap execution is the next planned acceleration step.
