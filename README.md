[![gymkhana](https://img.shields.io/pypi/v/gymkhana)](https://pypi.org/project/gymkhana/)
[![python_version](https://img.shields.io/badge/Python-%3E=3.10-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

# ApproxiMPC: Minimal MPC Data Collection

This repository is maintained as a focused MPC data-generation pipeline for imitation learning:

1. run MPC controllers in Gym-Khana,
2. collect trajectories into NPZ datasets,
3. train sequence models (for example LSTM) downstream.

## What This Repo Contains

Primary entrypoints:
- `src/kmpc_race.py`: visual kinematic MPC run.
- `src/stmpc_race.py`: visual single-track MPC run.
- `src/collect_mpc_data.py`: single-map data collection.
- `src/collect_mpc_multimap.py`: multi-map orchestration.

Primary configs:
- `configs/collect_mpc_default.yaml`: default single-run collector config.
- `configs/collect_mpc_test.yaml`: perturbation-enabled single-run test config.
- `configs/collect_mpc_multimap_fullscale.yaml`: full batch collection plan.

## Quick Start (Docker, Recommended)

Build:

```bash
docker compose build app
```

Run kinematic MPC GUI:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/kmpc_race.py"
```

Collect a single-map dataset (default config):

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_data.py"
```

Collect a full multi-map training dataset:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_multimap.py --config /app/configs/collect_mpc_multimap_fullscale.yaml"
```

When `execution.mode=parallel` is enabled, acados MPC solvers are automatically pre-compiled sequentially before parallel collection begins (this is a one-time cost, no manual steps needed).

## Data Output

Collector outputs are compressed NPZ datasets plus JSON metadata.

Typical fields:
- `observations`
- `lidar_scans` / `next_lidar_scans` (when `lidar.enabled=true`)
- `expert_actions`
- `executed_actions`
- `is_perturbed`
- `noise_vectors`
- rewards and termination flags

Default output root is configured in YAML under `output.output_dir`.

## Configuration Guide

All collection behavior is YAML-driven.

Key sections in collector configs:
- `run`: episodes, step limits, rendering, lap termination
- `env`: map, direction, timestep, integrator
- `controller`: `kmpc` or `stmpc` and controller-specific settings
- `lidar`: lidar feature collection, clipping, and scan binning controls
- `perturbation`: probabilistic action perturbation
- `dagger`: dual-action labeling for recovery supervision
- `output`: output directory and filename

In multi-map configs:
- `run.maps` and `run.directions` define the sweep order
- `episodes_per_combo` controls run count per map/direction
- `collector_overrides` applies deep overrides onto the base collector config

## Baseline Validation

Use this to verify the main MPC path remains healthy after code changes:

```bash
docker compose run --rm app bash -c 'cd /app && PYTHONPATH=/app python - <<"PY"
import gymnasium as gym
import numpy as np
import gymkhana
from mpc.gym_bridge import KMPCGymBridge

config = {
    "map": "Spielberg",
    "num_agents": 1,
    "timestep": 0.01,
    "integrator": "rk4",
    "model": "ks",
    "control_input": ["speed", "steering_angle"],
    "observation_config": {"type": "kinematic_state"},
    "normalize_act": False,
    "normalize_obs": False,
    "training_mode": "race",
    "track_direction": "normal",
    "max_episode_steps": 2000,
}

env = gym.make("gymkhana:gymkhana-v0", config=config, render_mode="human")
bridge = KMPCGymBridge(env, ref_speed=4.0)
x0, y0, yaw0 = bridge.get_start_pose()
obs, _ = env.reset(options={"poses": np.array([[x0, y0, yaw0]])})

steps = 0
while steps < 400:
    action = bridge.get_action(obs)
    obs, _, terminated, truncated, _ = env.step(action)
    env.render()
    steps += 1
    if terminated or truncated:
        break

env.close()
print(f"MPC GUI baseline complete: steps={steps}")
PY'
```

## Notes

- This codebase is intentionally MPC-focused and container-first.
- Historical RL/training/test infrastructure is out of scope for this workflow.
