# Next Steps

## Immediate Next Steps

- Investigate why the LSTM baseline fails to complete a lap and moves much more slowly than MPC.
- Verify runtime feature ordering and scaling against the trained model contract.
- Check whether the scaler version mismatch is materially affecting inference outputs.
- Confirm the path-feature and startup-speed behavior during rollout.

## Bugs Found During First Run

- `src/lstm_eval.py` failed in the dev container because `onnxruntime` was not installed in the image.
- LSTM ONNX export/runtime had a sequence-length mismatch:
  - exporter defaulted to `10`
  - runtime defaulted to `100`
  - actual training baseline expects `100`
- `scikit-learn` scaler pickle version mismatch warning appeared:
  - saved with `1.8.0`
  - container currently uses `1.2.0`
  - first run can proceed, but this should be aligned later.
- `gymnasium` reset warning appeared during LSTM eval:
  - `reset()` observation was reported as outside the declared observation space
  - first run completed, but the env/config contract should be checked later.

## First End-to-End Comparison Result

- LSTM comparison log generated successfully.
- MPC raw dataset generated successfully.
- MPC dataset conversion into shared comparison schema succeeded.
- Shared schema now matches between LSTM and MPC logs:
  - `collision_flag`
  - `completion_flag`
  - `distance_to_goal`
  - `distance_to_start`
  - `goal_center`
  - `goal_radius`
  - `has_left_start_zone`
  - `lidar`
  - `model_name`
  - `speed_command`
  - `start_center`
  - `start_radius`
  - `steering_angle`
  - `step_id`
  - `timestamps`
  - `velocity`
  - `x`
  - `y`
  - `yaw`

## New Investigation Items

- LSTM eval hit the max step budget (`6000`) instead of completing a lap.
- LSTM qualitative performance appears much slower than MPC and only covered a small fraction of the track.
- Need to confirm whether this is expected model behavior versus an integration/config mismatch.
- Current raw MPC comparison output path is `outputs/mpc_set/`.

Likely things to check:

- LSTM feature ordering and scaling assumptions
- start pose / start speed behavior
- action clipping and startup-speed override behavior
- path-feature contract alignment with the trained model
- artifact compatibility, especially scaler version mismatch
