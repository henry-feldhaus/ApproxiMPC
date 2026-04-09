# COPILOT.md

This file documents the current working state of this repository for coding agents.

## Repository Intent (Current)

This repo is now being used for a minimal supervised pipeline:

1. Run MPC in simulation across maps.
2. Collect trajectories (observations, actions, rewards, done flags).
3. Train an LSTM to imitate MPC behavior.

Not in scope for this workflow:
- RL training loops
- Wandb workflows
- PPO training/evaluation scripts
- Curriculum learning infrastructure

## Execution Model: Container-First

This repository should be treated as container-first (no required local Python environment).

Primary runtime:
- `docker compose run --rm app ...`

Container details:
- acados is built in the image at `/opt/acados`
- `acados_template` is installed in the container
- source is mounted at `/app`
- X11 socket is mounted for GUI rendering (`/tmp/.X11-unix`)

Important practical notes:
- Default to container `python` commands (no local venv assumptions).
- `uv` can be used inside the container when faster installs are needed.
- For scripts under `examples/`, use `PYTHONPATH=/app` or run scripts from `/app` to avoid import resolution issues.

Example:
```bash
docker compose run --rm app bash -c "cd /app/examples && PYTHONPATH=/app python simple_gui_test.py"
```

## Verified Current Status

### Core environment
- Gym environment registration exists: `gymkhana-v0` in `gymkhana/__init__.py`.
- Main env implementation: `gymkhana/envs/gymkhana_env.py`.
- Default config currently points to:
  - `map: Spielberg`
  - `model: st`
  - `control_input: ["speed", "steering_angle"]`
  - `training_mode: race`

### Working baseline run
- `examples/simple_gui_test.py` runs in container with:
  - `PYTHONPATH=/app`
  - container `python`
- Environment stepping and rendering path are functional with correct action shape (`np.array([[steer, speed]])`).

### MPC code availability
- MPC bridge code exists in `examples/controllers/mpc/gym_bridge.py`.
- Both bridges are present:
  - `KMPCGymBridge`
  - `STMPCGymBridge`
- acados-backed MPC modules are present under `examples/controllers/mpc/kmpc/` and `examples/controllers/mpc/stmpc/`.

## Structural Drift After Trimming

The repository no longer has a top-level `train/` package directory, but many scripts still import it.

Affected areas include:
- Several example scripts
- Some tests
- Analysis scripts

Common broken import patterns:
- `from train.config.env_config import ...`
- `from controllers...` (instead of `from examples.controllers...` when run from repo root)

This means many historical examples are currently stale and should not be treated as canonical for the minimal MPC->LSTM path until updated.

## Recommended Canonical Path (Now)

For MPC->LSTM work, treat these as primary components:

1. Environment core
   - `gymkhana/envs/gymkhana_env.py`
2. MPC interfaces
   - `examples/controllers/mpc/gym_bridge.py`
3. MPC controllers
   - `examples/controllers/mpc/kmpc/`
   - `examples/controllers/mpc/stmpc/`
4. Minimal workflow guide
   - `MINIMAL_MPC_LSTM_WORKFLOW.md`

## Known Container Compatibility Fixes

This repo includes Dockerfile patches for NVIDIA container compatibility:

- Gymnasium Atari wrapper import patch to avoid cv2 dnn typing breakage.
- cv2 typing fallback patch for missing `cv2.dnn.DictValue`.

These patches are required for reliable `import gymnasium` in this image family.

## What to Ignore for This Experiment

Unless explicitly requested, ignore RL-centric legacy surfaces:
- legacy scripts under `examples/legacy/`
- training scripts/docs that reference removed `train/` modules
- PPO/recovery pipeline instructions

## Immediate Next Technical Tasks (for future edits)

1. Add a dedicated `examples/collect_mpc_trajectories.py` script for sequential data collection.
2. Keep all trajectory generation commands container-first and path-stable.
3. Validate one GUI MPC lap before batch collection.

## Stage 1 Status (Completed)

- Primary MPC entrypoints were decoupled from removed `train.config` dependencies:
   - `examples/kmpc_race_example.py`
   - `examples/stmpc_race_example.py`
- Both scripts now build self-contained env configs and use `gymkhana:gymkhana-v0` directly.
- Both scripts bypass stale `examples/controllers/__init__.py` import side effects by importing MPC bridge via:
   - `sys.path` insert of `examples/controllers`
   - `from mpc.gym_bridge import ...`

## Functional Verification Gate (Before/After Trimming)

Use this as the required checkpoint before and after each trimming stage:

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

Latest baseline result (pre-trim Stage 1):
- `steps=400`
- `total_reward=74.70`

Latest baseline result (post-Stage-1 edits):
- `steps=400`
- `total_reward=74.70`

Additional Stage-1 sanity checks:
- `import kmpc_race_example` passes in container.
- `import stmpc_race_example` passes in container.

## Stage 2 Status (Completed)

- Stage-2 trim was implemented as a safe quarantine (move, not delete).
- Moved to `examples/legacy/`:
   - `examples/analysis/` (entire folder)
   - `examples/braking.py`
   - `examples/controller_example.py`
   - `examples/drift_debug.py`
   - `examples/random_trackgen.py`
   - `examples/run_in_empty_track.py`
   - `examples/test_inference_rate.py`
   - `examples/video_recording.py`

Active top-level examples are now intentionally minimal:
- `examples/kmpc_race_example.py`
- `examples/stmpc_race_example.py`
- `examples/simple_gui_test.py`
- `examples/waypoint_follow.py`
- `examples/examples_utils.py`

Latest baseline result (post-Stage-2 trim):
- `steps=400`
- `total_reward=74.70`

Additional Stage-2 sanity checks:
- `import kmpc_race_example` passes in container.
- `import stmpc_race_example` passes in container.

## Stage 3 Status (In Progress: Hard Cleanup)

- Requested hard cleanup is now applied directly (no quarantine):
   - `tests/` removed.
   - legacy analysis/example surfaces removed from top-level active workflow.
   - `.github/` workflows removed (CI/lint/publish automation removed).
- `README.md` was rewritten to match the minimal MPC, container-first workflow.
- `pyproject.toml` test configuration and pytest dev dependencies were removed to match the deleted test suite.

Pending Stage-3 gate before commit:
- Re-run the canonical MPC GUI verification command and confirm baseline remains stable.
