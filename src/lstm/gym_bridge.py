from __future__ import annotations

import numpy as np

from .artifacts import resolve_lstm_artifacts
from .features import PathFeatureLookup, process_scan
from .inference import LSTMInferenceModel


class LSTMGymBridge:
    def __init__(self, env, cfg: dict):
        self.track = env.unwrapped.track
        self.cfg = cfg

        artifact_cfg = cfg["artifacts"]
        self.artifacts = resolve_lstm_artifacts(
            model_config_path=artifact_cfg["model_config_path"],
            onnx_model_path=artifact_cfg.get("onnx_model_path") or None,
            input_scaler_path=artifact_cfg.get("input_scaler_path") or None,
            target_scaler_path=artifact_cfg.get("target_scaler_path") or None,
            auto_export_if_missing=bool(artifact_cfg.get("auto_export_if_missing", True)),
            export_device=str(artifact_cfg.get("export_device", "cpu")),
            base_dir=cfg.get("__project_root__"),
        )
        self.model = LSTMInferenceModel(self.artifacts)

        lidar_cfg = cfg["lidar"]
        self.clip_min = float(lidar_cfg["clip_min"])
        self.clip_max = float(lidar_cfg["clip_max"])
        self.n_bins = int(lidar_cfg["n_bins"])
        self.bin_method = str(lidar_cfg.get("binning_method", "min")).lower()

        self.max_speed = float(cfg["runtime"].get("max_speed", 8.0))
        self.max_steering = float(cfg["runtime"].get("max_steering_angle", 0.4))
        self.startup_speed = float(cfg["runtime"].get("startup_speed", 1.5))

        base_dim = 3 + self.n_bins
        self.path_feature_count = self.model.input_dim - base_dim
        if self.path_feature_count < 0:
            raise ValueError(f"Model input_dim={self.model.input_dim} is smaller than base feature dim {base_dim}.")

        lookahead_m = cfg["path_features"].get("lookahead_m", [])
        if self.path_feature_count == 0:
            self.path_lookup = None
        else:
            if len(lookahead_m) != self.path_feature_count:
                raise ValueError(
                    f"Model input_dim={self.model.input_dim} implies {self.path_feature_count} path features, "
                    f"but config provides {len(lookahead_m)} lookahead distances."
                )
            self.path_lookup = PathFeatureLookup(self.track, lookahead_m)

        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_lidar = np.zeros(self.n_bins, dtype=np.float32)

    def reset(self) -> None:
        self.model.reset()
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_lidar = np.zeros(self.n_bins, dtype=np.float32)

    def get_action(self, obs: dict) -> np.ndarray:
        agent_obs = obs["agent_0"]
        pose_x = float(agent_obs["pose_x"])
        pose_y = float(agent_obs["pose_y"])
        delta = float(agent_obs["delta"])
        linear_vel_x = float(agent_obs["linear_vel_x"])
        pose_theta = float(agent_obs["pose_theta"])

        lidar = process_scan(
            scan=np.asarray(agent_obs["scan"], dtype=np.float32),
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            n_bins=self.n_bins,
            method=self.bin_method,
        )
        self.last_lidar = lidar

        feature_parts = [np.array([delta, linear_vel_x, pose_theta], dtype=np.float32), lidar]
        if self.path_lookup is not None:
            feature_parts.append(self.path_lookup.get_curvature_lookahead(pose_x, pose_y))
        feature_step = np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False)

        pred = self.model.predict(feature_step)
        steering = float(np.clip(pred[0], -self.max_steering, self.max_steering))
        speed = float(np.clip(pred[1], 0.0, self.max_speed))
        if linear_vel_x < self.startup_speed:
            speed = max(speed, self.startup_speed)
        self.last_action = np.array([steering, speed], dtype=np.float32)
        return self.last_action[None, :]

    def get_start_pose(self) -> tuple[float, float, float]:
        cl = self.track.centerline
        return float(cl.xs[0]), float(cl.ys[0]), float(cl.yaws[0])
