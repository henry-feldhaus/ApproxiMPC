# Minimal MPC → LSTM Training Workflow

## Goal
Train an LSTM to replicate **kinematic MPC controller behavior** across multiple maps, without RL or drifting complexity.

## Key Insight
- **You have MPC as the expert** (acados-based controller)
- **No RL needed**: supervised learning from MPC trajectories
- **No drifting**: kinematic model (simpler, faster)
- **No complex normalization**: LSTM learns raw values
- **No parallel training**: sequential data collection is fine

---

## Phase 1: Verify MPC Works (GUI Test)

### 1.1 Run MPC Example with Visualization
```bash
docker compose run --rm app bash
cd examples
uv run python kmpc_race_example.py  # or stmpc_race_example.py
```

**What to look for:**
- Car follows centerline smoothly
- No crashes or erratic behavior
- acados solver runs without errors

**If successful**: MPC is correctly integrated and acados is available.

---

## Phase 2: Build Minimal Data Collector

### 2.1 Create `examples/collect_mpc_trajectories.py`

**Purpose**: Loop through environments, run MPC, save state-action pairs.

**Template**:
```python
import gymnasium as gym
import numpy as np
from pathlib import Path
from examples.controllers.mpc.gym_bridge import KMPCGymBridge  # or STMPC

# Kinematic model config (no drift)
config = {
    "model": "kinematic",
    "track": "Skid-Pad",
    "control_input": ["accl", "steering_angle"],
    "max_episode_steps": 1000,
}

env = gym.make("gymkhana:gymkhana-v0", config=config)
controller = KMPCGymBridge(env)  # MPC interface

trajectories = {
    "observations": [],
    "actions": [],
    "rewards": [],
    "dones": [],
}

obs, info = env.reset()
for step in range(1000):
    action = controller.compute_action(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    trajectories["observations"].append(obs)
    trajectories["actions"].append(action)
    trajectories["rewards"].append(reward)
    trajectories["dones"].append(done or truncated)
    
    if done or truncated:
        break

# Save
output_dir = Path("data/mpc_trajectories")
output_dir.mkdir(parents=True, exist_ok=True)
np.savez(output_dir / "trajectory_001.npz", **trajectories)
print(f"Saved: {output_dir / 'trajectory_001.npz'}")

env.close()
```

### 2.2 Test on Single Map with GUI
```bash
# Modify config: add render_mode="human"
uv run python examples/collect_mpc_trajectories.py
```

**Validation**: Watch the car drive and see trajectory saved to disk.

---

## Phase 3: Batch Collection

### 3.1 Extend to Multiple Maps
```python
MAPS = ["Skid-Pad", "Indy", "MoscowLvl3", "MoscowLvl4", "MoscowLvl5"]
for episode in range(5):
    for map_name in MAPS:
        config["track"] = map_name
        # Run collection loop
        # Save to data/mpc_trajectories/trajectory_<map>_<episode>.npz
```

**Expected output**: ~20-50 trajectory files, ~10-100MB total.

---

## Phase 4: LSTM Training

### 4.1 Update Notebook
```python
# Load collected trajectories
trajectories = []
for npz_file in Path("data/mpc_trajectories").glob("*.npz"):
    data = np.load(npz_file)
    trajectories.append(data)

# Stack into training dataset
X = np.concatenate([t["observations"] for t in trajectories])
y = np.concatenate([t["actions"] for t in trajectories])

# Train LSTM (existing notebook logic)
# model = LSTM(...)
# model.fit(X, y, ...)
```

### 4.2 Validation
Test LSTM on held-out test map:
```python
# Load test env (e.g., "Test-Track")
env.reset()
for step in range(500):
    lstm_action = lstm_model.predict(obs)
    obs, reward, done, trunc, info = env.step(lstm_action)
    
    # Compare with MPC on same track (in separate eval)
    # Check: do rewards align? Does car stay on track?
```

---

## Expected Timeline

| Phase | Time | Blocker |
|-------|------|---------|
| 1. Verify MPC | 5 min | acados_template import |
| 2. Build collector | 1 hour | MPC API understanding |
| 3. Batch collect | 2-4 hours | Wall-clock (running 25 episodes) |
| 4. LSTM train | 1-2 hours | Data pipeline + hyperparameter tuning |
| **Total** | **~6-10 hours** | Test on real hardware |

---

## Files to Create/Modify

1. ✅ `examples/collect_mpc_trajectories.py` (new, ~100 lines)
2. ✅ `notebooks/LSTM_training.ipynb` (update data loading section)
3. ✅ `data/mpc_trajectories/` (directory for saving `.npz` files)

---

## Comparison: Minimal vs Full

| Aspect | Minimal | Full PPO Setup |
|--------|---------|----------------|
| Config complexity | 5 lines | 50+ lines (env + RL hyperparams) |
| Training time | 2-4 hours | 50+ hours (50M timesteps) |
| Code size | ~200 lines | 2000+ lines |
| Wandb needed | No | Yes |
| Parallel envs | 1 (sequential) | 16 (CPU cores) |
| Callbacks | None | 5+ (checkpoint, eval, etc.) |
| Drifting model | No (kinematic) | Yes (STD with PAC2002) |

---

## Troubleshooting

**acados_template not found**:
```bash
python -c "import acados_template; print(acados_template.__file__)"
# If fails, reinstall in venv:
pip install -e /opt/acados/interfaces/acados_template
```

**MPC controller not responding**:
- Check acados solver settings (may need tuning for sim-to-trajectory)
- Verify gym config matches controller expectations

**LSTM overfitting**:
- Collect more maps/episodes
- Add dropout/regularization

---

## Next: Phase 1 Checkpoint
Once `kmpc_race_example.py` runs without errors, move to Phase 2 (data collector).
