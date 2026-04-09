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

Run a shell in the container:

```bash
docker compose run --rm app
```

Run the kinematic MPC GUI example:

```bash
docker compose run --rm app bash -c "cd /app/examples && PYTHONPATH=/app python kmpc_race_example.py"
```

Run the single-track MPC GUI example:

```bash
docker compose run --rm app bash -c "cd /app/examples && PYTHONPATH=/app python stmpc_race_example.py"
```

## Baseline Verification Command

Use this command to verify the main MPC path after changes:

```bash
docker compose run --rm app bash -c 'cd /app && yes y | python -c "from acados_template.utils import get_tera; print(get_tera())" && PYTHONPATH=/app python - <<"PY"
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

- `gymkhana/envs/gymkhana_env.py`: Environment core.
- `examples/kmpc_race_example.py`: Main kinematic MPC runner.
- `examples/stmpc_race_example.py`: Main single-track MPC runner.
- `examples/controllers/mpc/gym_bridge.py`: MPC-to-environment bridge.
- `MINIMAL_MPC_LSTM_WORKFLOW.md`: Minimal workflow notes.
- `COPILOT.md`: Agent-oriented status and validation notes.

## Notes

- This repository is intended to be run in Docker/devcontainer environments.
- If acados asks to install the tera renderer in an ephemeral container, use the `yes y | ... get_tera` bootstrap shown above in the same command session.
