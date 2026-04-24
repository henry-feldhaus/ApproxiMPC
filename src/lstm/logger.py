from __future__ import annotations

import datetime
import math
from pathlib import Path

import numpy as np


class LSTMRunLogger:
    def __init__(self, cfg: dict, model_name: str):
        log_cfg = cfg["logging"]
        self.project_root = Path(cfg.get("__project_root__", ".")).resolve()
        self.lidar_points = int(cfg["lidar"]["n_bins"])
        self.model_name = model_name

        self.goal_center_param = (float(log_cfg["goal_center_x"]), float(log_cfg["goal_center_y"]))
        self.goal_offset = (float(log_cfg["goal_offset_x"]), float(log_cfg["goal_offset_y"]))
        self.use_goal_offset = bool(log_cfg["use_goal_offset"])
        self.goal_radius = float(log_cfg["goal_radius"])
        self.start_center_param = (float(log_cfg["start_center_x"]), float(log_cfg["start_center_y"]))
        self.use_start_pose_as_start_center = bool(log_cfg["use_start_pose_as_start_center"])
        self.start_radius = float(log_cfg["start_radius"])

        self.output_file = self._resolve_output_file(log_cfg["output_file"], model_name)
        self.goal_center = None
        self.start_center = None
        self.collision_flag = False
        self.completion_flag = False
        self.has_left_start_zone = False
        self.start_time = None
        self.completion_time = None

        self.timestamps = []
        self.step_ids = []
        self.x = []
        self.y = []
        self.yaw = []
        self.velocities = []
        self.steering_angles = []
        self.speed_commands = []
        self.collision_flags = []
        self.completion_flags = []
        self.has_left_start_zone_flags = []
        self.goal_distances = []
        self.start_distances = []
        self.lidar_data = []

    def _resolve_output_file(self, output_file: str, model_name: str) -> Path:
        output_path = Path(output_file).expanduser()
        if not output_path.is_absolute():
            output_path = self.project_root / output_path
        if output_path.is_dir():
            base_name = "lstm_eval_log"
            output_dir = output_path
        else:
            base_name = output_path.stem if output_path.stem else "lstm_eval_log"
            output_dir = output_path.parent if output_path.parent != Path("") else Path("outputs/logs")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace(" ", "_")
        return output_dir / f"{base_name}_{safe_model_name}_{timestamp}.npz"

    def _distance(self, p1, p2) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def initialize_start_goal(self, pose_x: float, pose_y: float) -> None:
        current_pose = (pose_x, pose_y)
        if self.start_center is None:
            if self.use_start_pose_as_start_center:
                self.start_center = current_pose
            else:
                self.start_center = self.start_center_param
            if self.use_goal_offset:
                self.goal_center = (
                    self.start_center[0] + self.goal_offset[0],
                    self.start_center[1] + self.goal_offset[1],
                )
            else:
                self.goal_center = self.goal_center_param

    def log_step(
        self,
        step_id: int,
        timestamp_s: float,
        pose_x: float,
        pose_y: float,
        yaw: float,
        velocity: float,
        steering_command: float,
        speed_command: float,
        lidar: np.ndarray,
        collision: bool,
    ) -> None:
        self.initialize_start_goal(pose_x, pose_y)
        goal_dist = self._distance((pose_x, pose_y), self.goal_center)
        start_dist = self._distance((pose_x, pose_y), self.start_center)

        if collision:
            self.collision_flag = True
        if not self.has_left_start_zone and start_dist > self.start_radius:
            self.has_left_start_zone = True
            if self.start_time is None:
                self.start_time = timestamp_s
        if self.has_left_start_zone and not self.completion_flag and goal_dist <= self.goal_radius:
            self.completion_flag = True
            self.completion_time = timestamp_s

        self.timestamps.append(timestamp_s)
        self.step_ids.append(step_id)
        self.x.append(pose_x)
        self.y.append(pose_y)
        self.yaw.append(yaw)
        self.velocities.append(velocity)
        self.steering_angles.append(steering_command)
        self.speed_commands.append(speed_command)
        self.collision_flags.append(int(self.collision_flag))
        self.completion_flags.append(int(self.completion_flag))
        self.has_left_start_zone_flags.append(int(self.has_left_start_zone))
        self.goal_distances.append(goal_dist)
        self.start_distances.append(start_dist)
        self.lidar_data.append(np.asarray(lidar, dtype=np.float32).copy())

    def save(self) -> Path:
        lidar_array = (
            np.stack(self.lidar_data, axis=0)
            if self.lidar_data
            else np.empty((0, self.lidar_points), dtype=np.float32)
        )
        np.savez_compressed(
            self.output_file,
            timestamps=np.array(self.timestamps, dtype=np.float64),
            step_id=np.array(self.step_ids, dtype=np.int64),
            x=np.array(self.x, dtype=np.float32),
            y=np.array(self.y, dtype=np.float32),
            yaw=np.array(self.yaw, dtype=np.float32),
            velocity=np.array(self.velocities, dtype=np.float32),
            steering_angle=np.array(self.steering_angles, dtype=np.float32),
            speed_command=np.array(self.speed_commands, dtype=np.float32),
            collision_flag=np.array(self.collision_flags, dtype=np.uint8),
            completion_flag=np.array(self.completion_flags, dtype=np.uint8),
            has_left_start_zone=np.array(self.has_left_start_zone_flags, dtype=np.uint8),
            distance_to_goal=np.array(self.goal_distances, dtype=np.float32),
            distance_to_start=np.array(self.start_distances, dtype=np.float32),
            lidar=lidar_array,
            model_name=np.array(self.model_name),
            goal_center=np.array(self.goal_center if self.goal_center is not None else [np.nan, np.nan], dtype=np.float32),
            start_center=np.array(self.start_center if self.start_center is not None else [np.nan, np.nan], dtype=np.float32),
            goal_radius=np.array(self.goal_radius, dtype=np.float32),
            start_radius=np.array(self.start_radius, dtype=np.float32),
        )
        return self.output_file
