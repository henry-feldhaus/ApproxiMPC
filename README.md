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
- `LSTM_training/scripts/pytorch_to_onnx_converter.py`: export a trained LSTM checkpoint to ONNX.

Primary configs:
- `configs/collect_mpc_default.yaml`: default single-run collector config.
- `configs/collect_mpc_test.yaml`: perturbation-enabled single-run test config.
- `configs/collect_mpc_multimap_fullscale.yaml`: full batch collection plan.

## Getting Started

### Prerequisites

- **Docker** (recommended) with Docker Compose for isolated environment
- **4+ GB free disk space** for datasets and build artifacts

### Development Container Setup

The recommended approach is to use Docker for a consistent, reproducible environment.

**1. Build the Docker image:**

```bash
docker compose build app
```

This builds a NVIDIA PyTorch container with all MPC and simulation dependencies pre-installed.

**2. Verify the setup by running a quick MPC test:**

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python - <<"PY"
import gymnasium as gym
import gymkhana
print("✓ Gymnasium and Gymkhana imports successful")
PY"
```

### Data Collection Workflows

ApproxiMPC uses YAML-driven configuration for all data collection. Collections produce compressed NPZ datasets with metadata in JSON.

#### Quick Validation (5-10 min)

Test the full pipeline with minimal data:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_multimap.py --config /app/configs/collect_mpc_multimap_test.yaml"
```

This runs 3 maps × 2 directions × 2 episodes with 4 parallel workers.

#### Single-Map Collection (10-15 min)

Collect from one track with default settings:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_data.py"
```

Output: `outputs/datasets/kmpc_Spielberg_<timestamp>.npz` + `.json`

#### Full-Scale Multi-Map Collection (1.5-2 hours)

Generate comprehensive training dataset across 7 test tracks with perturbations and DAgger labels:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/collect_mpc_multimap.py --config /app/configs/collect_mpc_multimap_fullscale.yaml"
```

This runs:
- **7 maps**: Spielberg, Budapest, Monza, Spa, Silverstone, Melbourne, Montreal
- **2 directions each** (normal + reverse)
- **10 episodes per combo** for rich diversity
- **Parallel execution** with 4 concurrent collectors
- **Lidar data** (360 beams → 60 bins, clipped 0-15m)
- **Perturbations** (20% stochastic steering shoves)
- **DAgger labels** (expert + perturbed action pairs)

Output: `outputs/datasets/multimap_training/run_<timestamp>/` containing all map/direction/episode datasets.

#### Interactive MPC Visualization

Run the kinematic MPC controller in GUI mode (requires display):

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/kmpc_race.py"
```

Or single-track dynamic MPC:

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=/app python src/stmpc_race.py"
```

### Configuring Data Collection

All collection parameters are in YAML. Three canonical configs provided:

**`configs/collect_mpc_default.yaml`** (base/reference):
- Single map, single direction, default settings
- Lidar disabled (backward compatible)
- No perturbations or DAgger
- Use this as a template to customize

**`configs/collect_mpc_test.yaml`** (validation):
- Single map, 2 episodes for quick smoke testing
- Lidar, perturbations, and DAgger enabled
- ~5 min runtime

**`configs/collect_mpc_multimap_fullscale.yaml`** (production):
- 7 maps × 2 directions × 10 episodes
- Parallel execution (4 workers)
- All features enabled (lidar, perturbation, DAgger)

### Customizing Configs

Edit any YAML to override parameters:

```yaml
run:
  maps:
    - Spielberg        # which tracks
    - Budapest
  episodes_per_combo: 5 # how many laps per track

env:
  track_direction: normal # or "reverse" for counter-clockwise

controller:
  mode: kmpc           # "kmpc" or "stmpc"
  ref_speed: 4.0       # reference velocity (m/s)

lidar:
  enabled: true        # collect lidar scans
  clip_max: 15.0       # clip range (0-15m recommended for racing)
  binning:
    n_bins: 60         # downsample 360 beams to 60

perturbation:
  enabled: true        # inject stochastic steering shoves
  probability: 0.2     # 20% of steps perturbed

dagger:
  enabled: true        # record expert recovery labels
```

### Understanding Dataset Output

Each collection produces an NPZ file (compressed NumPy array) + JSON metadata.

NPZ contents:
- `state_vector` (N, 5) float32: [pose_x, pose_y, theta, delta, vx]
- `action` (N, 2) float32: [steer_cmd, throttle_cmd]
- `lidar_scans` (N, 60) float16: binned/clipped lidar (when enabled)
- `is_perturbed` (N,) bool: which steps had perturbations
- `dagger_actions` (N, 2) float32: expert recovery labels (when enabled)

JSON metadata:
- Collection parameters (map, direction, episodes, controller type, etc.)
- Lidar processing details (clipping, binning, FOV)
- Episode summaries (reward, lap count, termination reason)

### Training with Collected Data

Datasets are organized by map/direction for easy curriculum or full-batch training:

```
outputs/datasets/multimap_training/run_<timestamp>/
├── Spielberg/
│   ├── normal/
│   │   ├── kmpc_Spielberg_normal.npz    (all episodes concatenated)
│   │   └── kmpc_Spielberg_normal.json   (metadata)
│   └── reverse/
├── Budapest/
└── ... (other maps)
```

**Load data in Python:**

```python
import numpy as np
import json

# Load single map/direction combo
data = np.load("outputs/datasets/multimap_training/run.../Spielberg/normal/kmpc_Spielberg_normal.npz")
states = data["state_vector"]           # shape (N, 5)
actions = data["action"]                # shape (N, 2)
lidar = data["lidar_scans"]             # shape (N, 60) if enabled

with open("...json") as f:
    metadata = json.load(f)
    clip_max = metadata["lidar"]["clip_max"]  # your clipping threshold
```

**Downstream training** (LSTM, Transformer, etc.) typically uses:
- Input: concatenated [states, lidar] or just states
- Target: actions (or dagger_actions for imitation learning)
- Framework: PyTorch, TensorFlow, etc. (external to this repo)

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

### Exporting an LSTM Model to ONNX

Use the script under `LSTM_training/scripts` to export a trained checkpoint into a `.onnx` file:

```bash
cd /app
PYTHONPATH=/app python LSTM_training/scripts/pytorch_to_onnx_converter.py
```

By default this reads the model bundle in `LSTM_training/models/4-16-25 Models/LSTM_1B_128D_Pred_1/` and writes `LSTM_1B_128D.onnx` next to the checkpoint.

Optional flags:
- `--config`: path to the model config JSON.
- `--output`: where to write the ONNX file.
- `--opset`: ONNX opset version to export.

The exporter only writes the ONNX file. It does not start the simulator or run inference.

## LSTM Training and ONNX Export

### Exporting a Trained LSTM Model to ONNX

To export a trained LSTM checkpoint to ONNX format for F1Tenth simulation, you can use either the config/output method or the new convenient `--model_dir` option:

**Option 1: Using --model_dir (recommended)**

```bash
python LSTM_training/scripts/pytorch_to_onnx_converter.py --model_dir "LSTM_training/models/4-16-25 Models/LSTM_1B_128D_Pred_1"
```

This will automatically find the config and export the ONNX and scaler files to your current directory.

**Option 2: Manual config/output**

```bash
python LSTM_training/scripts/pytorch_to_onnx_converter.py \
  --config LSTM_training/models/4-16-25\ Models/LSTM_1B_128D_Pred_1/LSTM_1B_128D_config.json \
  --output LSTM_1B_128D.onnx
```

- `--config`: Path to the model config JSON (references all artifact files)
- `--output`: Path for the ONNX file to write
- `--opset`: (Optional) ONNX opset version (default: 17)
- `--device`: (Optional) Device to use (default: cpu)

**Note:** The script will also export the input/output scalers as `<model_name>_input_scaler.pkl` and `<model_name>_target_scaler.pkl` alongside the ONNX file.

---

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
