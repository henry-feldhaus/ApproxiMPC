"""Run standalone ONNX LSTM evaluation in Gymkhana and save ROS-style comparison logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

import gymkhana  # noqa: F401
from gymkhana.envs import GKEnv

from lstm.gym_bridge import LSTMGymBridge
from lstm.logger import LSTMRunLogger


class LapTracker:
    def __init__(self, track, wrap_threshold_ratio: float = 0.2):
        self.track = track
        self.track_length = float(track.centerline.spline.s[-1])
        self.wrap_threshold_ratio = float(wrap_threshold_ratio)
        self.prev_s = None
        self.lap_count = 0

    def reset(self, x: float, y: float) -> None:
        s, _ = self.track.centerline.spline.calc_arclength_inaccurate(x, y)
        self.prev_s = float(s) % self.track_length
        self.lap_count = 0

    def update(self, x: float, y: float) -> tuple[int, bool]:
        if self.prev_s is None:
            self.reset(x, y)
            return self.lap_count, False

        s, _ = self.track.centerline.spline.calc_arclength_inaccurate(x, y)
        current_s = float(s) % self.track_length
        high = self.track_length * (1.0 - self.wrap_threshold_ratio)
        low = self.track_length * self.wrap_threshold_ratio
        wrapped = self.prev_s >= high and current_s <= low
        if wrapped:
            self.lap_count += 1
        self.prev_s = current_s
        return self.lap_count, wrapped


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["__config_path__"] = str(config_path)
    cfg["__project_root__"] = str(Path(__file__).resolve().parent.parent)
    return cfg


def get_default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "lstm_eval_default.yaml"


def build_env_config(cfg: dict) -> dict:
    env_cfg = cfg["env"]
    lidar_cfg = cfg["lidar"]
    return {
        "map": env_cfg["map"],
        "num_agents": 1,
        "timestep": float(env_cfg.get("timestep", 0.01)),
        "integrator": str(env_cfg.get("integrator", "rk4")),
        "model": str(env_cfg.get("model", "std")),
        "control_input": ["speed", "steering_angle"],
        "observation_config": {
            "type": "features",
            "features": ["scan", "pose_x", "pose_y", "delta", "linear_vel_x", "pose_theta"],
        },
        "normalize_act": False,
        "normalize_obs": False,
        "training_mode": str(env_cfg.get("training_mode", "race")),
        "track_direction": str(env_cfg.get("track_direction", "normal")),
        "max_episode_steps": int(cfg["run"]["steps_per_episode"]),
        "num_beams": int(lidar_cfg["env_num_beams"]),
        "params": GKEnv.f1tenth_std_vehicle_params(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone LSTM evaluation in Gymkhana.")
    parser.add_argument("--config", type=str, default=str(get_default_config_path()), help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_config = build_env_config(cfg)
    env = gym.make("gymkhana:gymkhana-v0", config=env_config, render_mode="human" if cfg["run"]["render"] else None)
    bridge = LSTMGymBridge(env, cfg)
    logger = LSTMRunLogger(cfg, bridge.artifacts.model_name)

    try:
        x0, y0, yaw0 = bridge.get_start_pose()
        if cfg["run"].get("use_default_reset", False):
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(options={"poses": np.array([[x0, y0, yaw0]], dtype=np.float64)})
        bridge.reset()

        lap_cfg = cfg["run"]["lap_termination"]
        lap_tracker = LapTracker(env.unwrapped.track, wrap_threshold_ratio=float(lap_cfg.get("wrap_threshold_ratio", 0.2)))
        lap_tracker.reset(float(obs["agent_0"]["pose_x"]), float(obs["agent_0"]["pose_y"]))

        timestep = float(env_config["timestep"])
        for step_idx in range(int(cfg["run"]["steps_per_episode"])):
            action = bridge.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            del reward

            pose_x = float(next_obs["agent_0"]["pose_x"])
            pose_y = float(next_obs["agent_0"]["pose_y"])
            yaw = float(next_obs["agent_0"]["pose_theta"])
            velocity = float(next_obs["agent_0"]["linear_vel_x"])
            lidar = bridge.last_lidar
            collision = bool(info.get("collision", False))

            logger.log_step(
                step_id=step_idx,
                timestamp_s=(step_idx + 1) * timestep,
                pose_x=pose_x,
                pose_y=pose_y,
                yaw=yaw,
                velocity=velocity,
                steering_command=float(action[0, 0]),
                speed_command=float(action[0, 1]),
                lidar=lidar,
                collision=collision,
            )

            obs = next_obs
            laps, _ = lap_tracker.update(pose_x, pose_y)
            if cfg["run"]["render"]:
                env.render()

            if lap_cfg.get("enabled", True) and laps >= int(lap_cfg.get("target_laps", 1)):
                break
            if terminated or truncated:
                break
    finally:
        output_path = logger.save()
        env.close()
        print(f"Saved LSTM evaluation log to: {output_path}")


if __name__ == "__main__":
    main()
