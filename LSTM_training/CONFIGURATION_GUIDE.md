# ApproxiMPC LSTM Training: Configuration & Settings Guide

This guide documents all toggleable configuration parameters for data collection, preprocessing, and model training. Settings are organized by **location** (YAML configs or Jupyter notebooks) and **functional area**.

---

## 📍 Table of Contents

1. [MPC Data Collection (YAML)](#mpc-data-collection-yaml)
2. [Dataset Creation (Notebook)](#dataset-creation-notebook)
3. [LSTM Training (Notebook)](#lstm-training-notebook)
4. [Quick Reference](#quick-reference)

---

## MPC Data Collection (YAML)

### File: `configs/collect_mpc_multimap_fullscale.yaml`

These settings control how simulation data is collected for LSTM training.

#### **LiDAR Collection**
```yaml
lidar:
  enabled: true                    # Enable/disable LiDAR scan collection
  env_num_beams: 360              # Number of simulated LiDAR beams (360 = 1° per beam)
  clip_min: 0.0                   # Minimum range (m), values below are clipped to this
  clip_max: 15.0                  # Maximum range (m), values above are clipped to this
  store_raw: false                # false = save only binned data; true = also save raw 360-beam scans
  dtype: float16                  # Data type for storage (float16 = more compact)
  binning:
    enabled: true                 # Enable spatial binning/downsampling of beams
    n_bins: 60                    # Output bins per scan (360 → 60 beams)
    method: min                   # "min" = keep closest obstacle, emphasizes nearby threats
```

**Explanation:**
- `enabled`: Controls whether LiDAR is collected at all. If `false`, LiDAR keys will be empty.
- `env_num_beams` & `clip_min/max`: Simulate realistic LiDAR hardware specs (limited range, finite resolution).
- `binning`: Reduces data volume by 6× (360 → 60 bins). LiDAR augmentation in training will add realistic noise to these binned scans.
- **When to adjust:** Increase `n_bins` (60 → 120) for finer LiDAR resolution; increase `clip_max` (15.0 → 25.0) for larger track visibility.

---

#### **Perturbation (Recovery Training)**
```yaml
perturbation:
  enabled: true                   # Enable/disable steering shove perturbations
  probability: 0.25               # Probability of starting a perturbation event (subject to cooldown)
  shove_magnitude: 0.2            # Steering angle perturbation (radians); 0.2 rad ≈ 11.5°
  min_steps_between_shoves: 500   # Cooldown: minimum steps between perturbation event starts
  hold_steps_min: 10              # Hold the same perturbation for at least this many steps
  hold_steps_max: 30              # Hold the same perturbation for at most this many steps
```

**Explanation:**
- `enabled`: If `false`, no perturbations applied; MPC will track reference perfectly (poor recovery training).
- `probability`: Event trigger probability when cooldown allows; lower (0.1) = fewer disturbances, higher (0.3) = more frequent disturbances.
- `shove_magnitude`: Larger values create more severe disturbances. 0.2 rad is a moderate disturbance setting.
- `min_steps_between_shoves`: Prevents event starts from clustering. At 100 Hz, 500 steps = 5.0 s between event starts.
- `hold_steps_min/max`: Once triggered, the same noise vector is held for a random duration in `[hold_steps_min, hold_steps_max]` to create persistent disturbance bursts.
- **When to adjust:** 
  - Increase magnitude if recovery behavior is weak in deployment
  - Decrease hold range if disturbances cause frequent boundary terminations
  - Increase cooldown further to reduce disturbance frequency in narrow/technical maps

---

#### **DAgger Configuration**
```yaml
dagger:
  enabled: true                   # Enable DAgger (record both expert + executed actions for later replay)
  record_noise_vectors: true      # Record the Gaussian perturbation applied (for analysis/replay)
```

**Explanation:**
- `enabled`: If `true`, saves both `expert_actions` (clean MPC) and `executed_actions` (with perturbations). Used for recovery-specific supervision later.
- `record_noise_vectors`: Stores the actual perturbation noise vector added, enabling exact reproduction of scenarios.

---

#### **Run Configuration**
```yaml
run:
  maps:                           # Which maps to collect data from
    - Spielberg
    - Budapest
    - Monza
    - Spa
    - Silverstone
    - Melbourne
    - Montreal
  directions:                     # Direction variants per map
    - normal
    - reverse
  episodes_per_combo: 10          # Number of episodes per (map, direction) pair
  render: false                   # Disable visualization for faster unattended collection
  stop_on_error: false            # Continue on failed episodes instead of halting entire run
```

---

### File: `configs/collect_mpc_default.yaml` (Base Config)

Inherited by multimap configs. Key toggles:

```yaml
path_features:
  enabled: true                   # Enable centerline-derived path features
  lookahead_m: [1.0, 2.0, 3.0, 5.0, 8.0]  # Distances (m) for curvature lookahead
  include_current_kappa: true     # Include curvature at current position
  dtype: float32                  # Data type for storage
```

**Explanation:**
- Path features add road geometry context to steering/speed supervision. Helps LSTM understand why steering changes are needed.
- Lookahead distances: 1m = very local, 8m = long-horizon planning.
- **When to adjust:** Add smaller values (0.5m) for tight track corners; add larger (10m+) for high-speed straights.

---

## Dataset Creation (Notebook)

### File: `LSTM_training/notebooks/dataset_creation.ipynb`

Settings for processing raw NPZ files into training-ready episode sequences.

#### **Cell 5: EMA Smoothing Toggle**
```python
APPLY_EMA_SMOOTHING = True        # Enable causal exponential moving average on steering targets
EMA_ALPHA = 0.2                   # Smoothing factor: 0 = no smoothing, 1 = current value only
```

**Explanation:**
- `APPLY_EMA_SMOOTHING`: When `True`, reduces high-frequency jitter in steering targets from bang-bang actuator chatter.
- `EMA_ALPHA`: 
  - **0.1** = aggressive smoothing, may remove important steering changes
  - **0.2** (default) = balanced, removes jitter while preserving steering intent
  - **0.3** = light smoothing, closer to raw MPC output
- Applied **per-episode** (no cross-episode contamination).
- Operates only on steering (channel 0); velocity (channel 1) unchanged.
- **When to adjust:** Increase alpha if LSTM still overfits to micro-oscillations; decrease if steering changes become too sluggish.

---

#### **Processing Logic** (No Toggles, But Good to Know)
```python
# Drop x,y from observations: keep [delta, linear_vel_x, pose_theta]
observations = data["observations"][:, 2:]  # (N, 3)
lidar_scans = data["lidar_scans"]          # (N, 60)
inputs = np.concatenate([observations, lidar_scans], axis=1)  # (N, 63)

targets = data["expert_actions"]  # (N, 2) from MPC
```

**Note:** Output always uses `expert_actions` (clean MPC), not `executed_actions`. Perturbations are part of the **input recovery context**, not the supervision signal.

---

## LSTM Training (Notebook)

### File: `LSTM_training/notebooks/LSTM_training.ipynb`

#### **Section 0: Setup - Data Loading & Preprocessing**

##### Cell: Configuration Parameters (Top of notebook)
```python
# Data settings
SEQ_LENGTH = 100                  # Number of historical timesteps model sees (1 second at 100 Hz)
PREDICT_LENGTH = 1                # Steps into future to predict (always 1 for single-step prediction)
BATCH_SIZE = 64                   # Samples per gradient update

# Split strategy
SPLIT_MODE = "map"               # "map" = 70/15/15 by unique maps (cross-map generalization)
                                  # "episode" = 70/15/15 by random episodes (allows same map in train/test)
```

**Explanation:**
- `SEQ_LENGTH`: Longer = more context, but more memory and computation. 100 steps = 1s history.
- `SPLIT_MODE = "map"` ensures test set sees **only new maps**, validating generalization. Use for final evaluation.
- `SPLIT_MODE = "episode"` randomizes by episode within maps; quicker to verify training works.

---

##### Cell: Data Augmentation Toggle
```python
ENABLE_LIDAR_AUGMENTATION = True  # Enable training-only input noise
```

**When `True` (augmentation enabled):**
```python
if ENABLE_LIDAR_AUGMENTATION:
    from data_transforms import LiDARNoise
    train_augment = LiDARNoise(
        gaussian_sigma_m=0.1,     # Std dev of range noise (meters)
        dropout_prob=0.05         # Probability of zeroing each beam (5%)
    )
    print(f"Training augmentation enabled: {train_augment}")
else:
    train_augment = None
```

**Explanation:**
- `gaussian_sigma_m=0.1`: Adds ±0.1m noise to each LiDAR range measurement (realistic sensor noise).
- `dropout_prob=0.05`: Simulates 5% beam failures or occlusions (realistic failure modes).
- Augmentation **only applied to training data**, never to validation/test (prevents data leakage).
- **When to adjust:**
  - Increase sigma (0.1 → 0.2) if LSTM overfits to clean sensor data
  - Increase dropout (0.05 → 0.15) to stress-test robustness
  - Set both to 0 if deployment sensors are very clean

---

#### **Section 0: Model Configuration**
```python
MODEL_TYPE = 'LSTM'               # 'LSTM', 'GRU', or 'RNN'
RNN_HIDDEN_DIM = 64               # Hidden state size (larger = more capacity, more parameters)
NUM_RNN_LAYERS = 1                # Number of stacked recurrent layers (1 or 2)
USE_EMBEDDING = True              # Linear embedding layer before RNN (reduces input dim)
EMBEDDING_DIM = 30                # Embedding output size (input_dim=63 → 30)
FC_LAYER_SIZES = [64, 32, 16]     # Fully connected head after RNN
DROPOUT = 0.2                     # Dropout rate in RNN and FC layers
```

**Explanation:**
- Larger `RNN_HIDDEN_DIM` increases model capacity but also parameters and training time.
- `EMBEDDING_DIM = 30`: Compresses 63-dim input (3 kinematics + 60 LiDAR) into 30 dims, reducing overfitting.
- `NUM_RNN_LAYERS = 1`: Single layer keeps model simple and fast. Use 2 for more complex patterns.

---

#### **Section 0: Training Hyperparameters**
```python
EPOCHS = 5                        # Number of training passes (small for quick testing)
LEARNING_RATE = 1e-3              # Adam optimizer learning rate
WEIGHT_DECAY = 1e-5               # L2 regularization
MAX_GRAD_NORM = 1.0               # Gradient clipping threshold (prevents RNN explosions)
EARLY_STOPPING_PATIENCE = 4       # Stop if val loss doesn't improve for 4 epochs
SCHEDULER_STEP_SIZE = 3           # Every 3 epochs, multiply LR by gamma
SCHEDULER_GAMMA = 0.5             # LR multiplier (halve every 3 epochs)
STEERING_LOSS_WEIGHT = 1.0        # Relative loss weight for steering
VELOCITY_LOSS_WEIGHT = 1.0        # Relative loss weight for velocity
```

**Explanation:**
- `LEARNING_RATE`: 1e-3 is standard for Adam. Decrease to 5e-4 if training diverges; increase to 2e-3 if learning is slow.
- `EARLY_STOPPING_PATIENCE`: Stops training early if no validation improvement. Prevents overfitting.
- `STEERING_LOSS_WEIGHT > VELOCITY_LOSS_WEIGHT` would emphasize steering accuracy (common in racing); currently balanced.
- **When to adjust:**
  - If model underfits: increase EPOCHS, decrease LEARNING_RATE, add layers
  - If model overfits: decrease EPOCHS, increase WEIGHT_DECAY, enable augmentation

---

#### **Section 1: Data Loading**

##### Cell: PKL Path & Augmentation
```python
# Locate dataset
PKL_PATH = "datasets/multimap_lstm_sequences_no_collisions.pkl"

# Optional: Apply input augmentation only during training
ENABLE_LIDAR_AUGMENTATION = True
if ENABLE_LIDAR_AUGMENTATION:
    train_augment = LiDARNoise(gaussian_sigma_m=0.1, dropout_prob=0.05)
else:
    train_augment = None

# Build loaders (augmentation only applied to train_loader)
train_loader, val_loader, test_loader, test_episodes, input_scaler, target_scaler = load_and_prep_data(
    pkl_path=PKL_PATH,
    seq_length=SEQ_LENGTH,
    predict_length=PREDICT_LENGTH,
    batch_size=BATCH_SIZE,
    split_mode=SPLIT_MODE,
    train_transform=train_augment,  # <- Only training data is augmented
)
```

**Key Behavior:**
- `train_loader`: Augmentation applied to every batch ✓
- `val_loader`: No augmentation ✗ (clean validation)
- `test_loader`: No augmentation ✗ (clean test, final evaluation)

---

#### **Section 2: Model Architecture**
```python
model = F1TenthSequenceModel(
    input_dim=63,                  # 3 kinematics + 60 LiDAR bins
    rnn_hidden_dim=RNN_HIDDEN_DIM,
    output_dim=2,                  # [steering, velocity]
    model_type=MODEL_TYPE,
    num_rnn_layers=NUM_RNN_LAYERS,
    use_embedding=USE_EMBEDDING,
    embedding_dim=EMBEDDING_DIM,
    fc_layer_sizes=FC_LAYER_SIZES,
    dropout=DROPOUT,
)
```

**No toggles here, but these determine model capacity.** Adjust via Section 0 parameters above.

---

#### **Section 3: Loss & Optimization**
```python
criterion = WeightedRMSELoss(
    steering_weight=STEERING_LOSS_WEIGHT,
    velocity_weight=VELOCITY_LOSS_WEIGHT,
)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
```

**No new toggles; parameters from Section 0 apply here.**

---

#### **Section 4: Training Loop**
```python
def train_pipeline(...):
    for epoch in range(epochs):
        # Training phase: augmentation applied via train_loader
        model.train()
        for batch_x, batch_y in train_loader:  # batch_x includes augmented LiDAR
            ...
        
        # Validation & test: no augmentation
        model.eval()
        avg_val_loss = evaluate_model(model, val_loader, ...)
        avg_test_loss = evaluate_model(model, test_loader, ...)
```

**Key:** Augmentation happens **automatically in the DataLoader** during training; no additional toggles needed in the loop itself.

---

#### **Section 5: Analytics & Visualization**
```python
# Plotting functions (no toggles, but useful for understanding):
plot_learning_curves(history, model_name=SAVE_MODEL_NAME, save_dir=analytics_dir)
plot_predictions_vs_actual(model, test_loader, DEVICE, target_scaler=fitted_target_scaler, num_samples=200)
plot_test_episode_predictions(model, test_episodes, episode_index=0, device=DEVICE, target_scaler=fitted_target_scaler)

# Diagnostic functions:
diagnose_targets_and_baselines(PKL_PATH, seq_length=SEQ_LENGTH)
evaluate_per_target_metrics(model, test_loader, DEVICE, fitted_target_scaler)
```

**No toggles; run these cells for interpretability and debugging.**

---

#### **Section 6: Model Saving**
```python
save_outputs = save_model_and_artifacts(
    model=model,
    input_scaler=fitted_input_scaler,
    target_scaler=fitted_target_scaler,
    history=history,
    test_loader=test_loader,
    device=DEVICE,
    model_name=SAVE_MODEL_NAME,  # Saved under models/{SAVE_MODEL_NAME}/{date}/
    save_config=MODEL_CONFIG,
    seq_length_for_latency=SEQ_LENGTH,
    jetson_available_fraction=0.35,  # Assumes 35% of Jetson Nano compute available
)
```

**`jetson_available_fraction`:** Estimates inference latency assuming 35% of Jetson Nano compute is available (rest used by ROS, localization, etc.). Adjust to 0.5 for more optimistic estimate.

---

## Quick Reference

| Setting | Location | Default | Purpose | Adjustment |
|---------|----------|---------|---------|------------|
| **EMA Steering Smoothing** | dataset_creation.ipynb Cell 5 | α=0.2 | Reduce bang-bang actuator chatter | ↑ Alpha for more smoothing |
| **LiDAR Augmentation** | LSTM_training.ipynb Cell (loading) | Enabled | Add realistic sensor noise | σ=0.1m, dropout=5% |
| **Perturbation Magnitude** | collect_mpc_multimap_fullscale.yaml | 0.2 rad | Steering disturbance strength | ↑ for harder recovery, ↓ for easier |
| **Perturbation Frequency** | collect_mpc_multimap_fullscale.yaml | 0.25 (event probability) | How often disturbance events start | ↑ for more recovery examples |
| **Perturbation Persistence** | collect_mpc_multimap_fullscale.yaml | 10-30 steps | Duration of each disturbance burst | ↑ for longer off-center recoveries |
| **LiDAR Bins** | collect_mpc_multimap_fullscale.yaml | 60 | Resolution of LiDAR input | ↑ for fine detail, ↓ for speed |
| **SEQ_LENGTH** | LSTM_training.ipynb Section 0 | 100 | Context window (seconds) | 1 sec = 100 steps at 100 Hz |
| **Split Mode** | LSTM_training.ipynb Section 0 | "map" | Cross-map generalization test | "episode" for quick validation |
| **RNN Hidden Dim** | LSTM_training.ipynb Section 0 | 64 | Model capacity | ↑ for complex patterns, ↓ for speed |
| **Early Stopping** | LSTM_training.ipynb Section 0 | 4 epochs | Overfit prevention | ↑ for longer training, ↓ for quicker stop |

---

## 🚀 Recommended Workflows

### **Quick Validation Run** (Minutes)
```python
# dataset_creation.ipynb
APPLY_EMA_SMOOTHING = True  # ← Toggle ON

# LSTM_training.ipynb
SPLIT_MODE = "episode"      # ← Random splits (faster)
ENABLE_LIDAR_AUGMENTATION = True  # ← ON
EPOCHS = 5                  # ← Short run
```

### **Production Training** (Hours)
```python
# dataset_creation.ipynb
APPLY_EMA_SMOOTHING = True  # ← Always ON for real data

# LSTM_training.ipynb
SPLIT_MODE = "map"          # ← Cross-map validation
ENABLE_LIDAR_AUGMENTATION = True  # ← ON
EPOCHS = 30                 # ← Full training
```

### **Minimal Setup** (Debug)
```python
# dataset_creation.ipynb
APPLY_EMA_SMOOTHING = False # ← Raw data only

# LSTM_training.ipynb
SPLIT_MODE = "episode"      # ← Fast splits
ENABLE_LIDAR_AUGMENTATION = False  # ← No augmentation
EPOCHS = 2                  # ← Sanity check
```

---

## 📊 Current Settings Summary

**Active Configuration (Enabled):**
- EMA Smoothing: α=0.2 (dataset_creation.ipynb)
- LiDAR Augmentation: σ=0.1m, dropout=5% (LSTM_training.ipynb)
- Perturbations: 0.2 rad magnitude, 0.25 event probability, 500-step cooldown, 10-30 step hold (collect_mpc_multimap_fullscale.yaml)
- Path Features: Centerline curvature lookahead at 1,2,3,5,8m (collect_mpc_default.yaml)

All features work together to **reduce LSTM overfit** while maintaining **recovery learning capability**.
