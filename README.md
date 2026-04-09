[![gymkhana](https://img.shields.io/pypi/v/gymkhana)](https://pypi.org/project/gymkhana/)
[![python_version](https://img.shields.io/badge/Python-%3E=3.10-purple)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

# Gym-Khana (Minimal MPC Workflow)

This repository is currently maintained as a minimal, container-first workflow for:

1. Running MPC in simulation.
2. Collecting trajectories.
3. Supporting downstream imitation learning (for example, LSTM training outside this repo).

The RL training/test/CI surfaces were intentionally removed to keep the codebase focused on MPC data generation.

## Quick Start (Container-First)

Build the image:

```bash
docker compose build app
```

If you just pulled Dockerfile changes, rebuild to pick up baked dependencies (including acados `t_renderer`):

```bash
docker compose build --no-cache app
```

Run a shell in the container:

```bash
docker compose run --rm app
```

Run the kinematic MPC GUI example:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/kmpc_race.py"
```

Run the single-track MPC GUI example:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/stmpc_race.py"
```

Collect transition data with kinematic MPC (clean expert behavior):

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_data.py"
```

Collect with perturbation injection and DAgger dual-action labeling:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_data.py --config /app/configs/collect_mpc_test.yaml"
```

Collect with custom configuration file:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_data.py --config /path/to/custom.yaml --render"
```

### Data Collection Configuration

Edit `configs/collect_mpc_default.yaml` to customize collection behavior:
- `run`: episodes, max steps, rendering, reset behavior
- `env`: map, timestep, integrator, vehicle model
- `controller`: MPC reference speed
- `perturbation`: probabilistic noise injection (80% clean, 20% perturbed)
- `dagger`: dual-action labeling for recovery learning
- `output`: dataset directory and filename

Datasets are saved as compressed NPZ files with metadata JSON. Fields include `expert_actions`, `executed_actions`, `is_perturbed` flag, and `noise_vectors` for imitation learning.

## Baseline Verification Command

Use this command to verify the main MPC path after changes:

```bash
docker compose run --rm app bash -c 'cd /app && PYTHONPATH=/app python - <<"PY"
import gymnasium as gym
import numpy as np
import gymkhana
import sys

sys.path.insert(0, "/app/examples/controllers")
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
total_reward = 0.0
while steps < 400:
    action = bridge.get_action(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    env.render()
    steps += 1
    if terminated or truncated:
        break

print(f"MPC GUI baseline complete: steps={steps}, total_reward={total_reward:.2f}")
env.close()
PY'
```

Expected baseline metric in this trimmed state:
- `steps=400`
- `total_reward=74.70`

## Project Layout (Current)

Primary files for the current workflow:

- `src/kmpc_race.py`: Interactive kinematic MPC runner.
- `src/stmpc_race.py`: Interactive single-track MPC runner.
- `src/collect_mpc_data.py`: MPC data collection with YAML configuration.
- `configs/collect_mpc_default.yaml`: Default data collection parameters.
- `gymkhana/envs/gymkhana_env.py`: Gymnasium environment core.
- `examples/controllers/mpc/gym_bridge.py`: MPC-environment bridge for action computation.

## Notes

- This repository is intended to be run in Docker/devcontainer environments.
- The Docker image installs acados `t_renderer` during build, so GUI/MPC runs should not prompt for an interactive tera download after rebuild.
