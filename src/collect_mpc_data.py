"""Collect transition data by rolling out MPC controllers with advanced features.

Features:
  - YAML-based configuration management
  - Lap-based episode termination
  - Probabilistic perturbation injection (80% clean, 20% perturbed)
    - Optional persistent perturbation bursts (same shove held for multiple steps)
  - DAgger-style dual action labeling (expert vs. executed)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import yaml

import gymkhana  # noqa: F401  # ensures gym env registration
from mpc.gym_bridge import KMPCGymBridge, STMPCGymBridge

FEATURE_NAMES = [
    "pose_x",
    "pose_y",
    "delta",
    "linear_vel_x",
    "pose_theta",
]

ACTION_NAMES = ["steering_angle", "speed"]

VIDEO_VIEW_NAMES = {"global", "follow"}


class VideoRecorder:
    """Lazy OpenCV MP4 writer for RGB frames returned by env.render()."""

    def __init__(self, path: Path, fps: float):
        self.requested_path = path
        self.path = path
        self.fps = float(fps)
        self.writer = None
        self.codec = None

    def write(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        height, width = frame.shape[:2]
        if self.writer is None:
            self._open(width, height)

        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _open(self, width: int, height: int) -> None:
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)
        candidates = [
            (self.requested_path, "mp4v"),
            (self.requested_path, "avc1"),
            (self.requested_path, "H264"),
            (self.requested_path.with_suffix(".avi"), "MJPG"),
        ]
        for path, codec in candidates:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), self.fps, (width, height))
            if writer.isOpened():
                self.path = path
                self.codec = codec
                self.writer = writer
                if path != self.requested_path:
                    print(
                        f"WARNING: MP4 writer unavailable; recording {self.requested_path.name} "
                        f"as {path.name} with codec={codec}"
                    )
                return
            writer.release()

        raise RuntimeError(f"Failed to open video writer: {self.requested_path}")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class LapTracker:
    """Track lap completion from centerline arclength wrap-around."""

    def __init__(self, track: Any, wrap_threshold_ratio: float = 0.2):
        self.track = track
        self.track_length = float(track.centerline.spline.s[-1])
        self.wrap_threshold_ratio = float(wrap_threshold_ratio)
        self.prev_s = None
        self.lap_count = 0

    def reset(self, x: float, y: float) -> None:
        s, _ = self.track.centerline.spline.calc_arclength_inaccurate(x, y)
        self.prev_s = float(s) % self.track_length
        self.lap_count = 0

    def update(self, x: float, y: float) -> tuple[int, bool, float]:
        if self.prev_s is None:
            self.reset(x, y)
            return self.lap_count, False, float(self.prev_s)

        s, _ = self.track.centerline.spline.calc_arclength_inaccurate(x, y)
        current_s = float(s) % self.track_length

        high = self.track_length * (1.0 - self.wrap_threshold_ratio)
        low = self.track_length * self.wrap_threshold_ratio
        wrapped = self.prev_s >= high and current_s <= low
        if wrapped:
            self.lap_count += 1

        self.prev_s = current_s
        return self.lap_count, wrapped, current_s


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_default_config_path() -> Path:
    """Return path to default config file."""
    return Path(__file__).resolve().parent.parent / "configs" / "collect_mpc_default.yaml"


def get_lap_termination_config(run_cfg: dict) -> dict:
    lap_cfg = run_cfg.get("lap_termination", {})
    return {
        "enabled": bool(lap_cfg.get("enabled", True)),
        "target_laps": int(lap_cfg.get("target_laps", 1)),
        "method": str(lap_cfg.get("method", "s_wrap")),
        "wrap_threshold_ratio": float(lap_cfg.get("wrap_threshold_ratio", 0.2)),
        "auto_timeout": {
            "enabled": bool(lap_cfg.get("auto_timeout", {}).get("enabled", True)),
            "timeout_factor": float(lap_cfg.get("auto_timeout", {}).get("timeout_factor", 2.0)),
            "min_expected_speed_ratio": float(
                lap_cfg.get("auto_timeout", {}).get("min_expected_speed_ratio", 0.5)
            ),
        },
    }


def get_controller_config(cfg: dict) -> dict:
    """Normalize controller configuration from YAML."""
    ctrl_cfg = cfg.get("controller", {})
    mode = str(ctrl_cfg.get("mode", "kmpc")).lower().strip()
    if mode not in {"kmpc", "stmpc"}:
        raise ValueError(f"Unsupported controller.mode='{mode}'. Use 'kmpc' or 'stmpc'.")

    return {
        "mode": mode,
        "ref_speed": float(ctrl_cfg.get("ref_speed", 4.0)),
        "kmpc": ctrl_cfg.get("kmpc", {}),
        "stmpc": ctrl_cfg.get("stmpc", {}),
    }


def get_lidar_config(cfg: dict, controller_cfg: dict) -> dict:
    """Normalize lidar collection settings from YAML."""
    lidar_cfg = cfg.get("lidar", {})
    enabled = bool(lidar_cfg.get("enabled", False))
    mode = controller_cfg["mode"]

    if mode == "stmpc":
        required_features = [
            "scan",
            "pose_x",
            "pose_y",
            "delta",
            "linear_vel_x",
            "linear_vel_y",
            "pose_theta",
            "ang_vel_z",
            "beta",
        ]
    else:
        required_features = [
            "scan",
            "pose_x",
            "pose_y",
            "delta",
            "linear_vel_x",
            "pose_theta",
        ]

    binning_cfg = lidar_cfg.get("binning", {})
    n_bins = int(binning_cfg.get("n_bins", 60))
    if n_bins <= 0:
        raise ValueError("lidar.binning.n_bins must be > 0")

    method = str(binning_cfg.get("method", "min")).lower().strip()
    if method not in {"min", "mean"}:
        raise ValueError("lidar.binning.method must be one of: min, mean")

    dtype = str(lidar_cfg.get("dtype", "float16")).lower().strip()
    if dtype not in {"float16", "float32"}:
        raise ValueError("lidar.dtype must be one of: float16, float32")

    return {
        "enabled": enabled,
        "num_beams": int(lidar_cfg.get("env_num_beams", 360)),
        "clip_min": float(lidar_cfg.get("clip_min", 0.0)),
        "clip_max": float(lidar_cfg.get("clip_max", 30.0)),
        "store_raw": bool(lidar_cfg.get("store_raw", False)),
        "dtype": dtype,
        "feature_list": required_features,
        "binning": {
            "enabled": bool(binning_cfg.get("enabled", True)),
            "n_bins": n_bins,
            "method": method,
        },
    }


def get_path_feature_config(cfg: dict) -> dict:
    """Normalize centerline-derived path feature settings from YAML."""
    path_cfg = cfg.get("path_features", {})

    dtype = str(path_cfg.get("dtype", "float32")).lower().strip()
    if dtype not in {"float16", "float32"}:
        raise ValueError("path_features.dtype must be one of: float16, float32")

    lookahead_raw = path_cfg.get("lookahead_m", [1.0, 2.0, 3.0, 5.0, 8.0])
    if not isinstance(lookahead_raw, list) or len(lookahead_raw) == 0:
        raise ValueError("path_features.lookahead_m must be a non-empty list")

    lookahead_m = np.asarray([float(v) for v in lookahead_raw], dtype=np.float64)
    if np.any(lookahead_m < 0.0):
        raise ValueError("path_features.lookahead_m values must be >= 0")

    return {
        "enabled": bool(path_cfg.get("enabled", True)),
        "lookahead_m": lookahead_m,
        "include_current_kappa": bool(path_cfg.get("include_current_kappa", True)),
        "dtype": dtype,
    }


def get_video_config(cfg: dict, timestep: float) -> dict:
    """Normalize optional render-to-video settings."""
    video_cfg = cfg.get("run", {}).get("video", {})
    enabled = bool(video_cfg.get("enabled", False))

    views_raw = video_cfg.get("views", ["global"])
    if isinstance(views_raw, str):
        views = [views_raw]
    elif isinstance(views_raw, list):
        views = [str(v).lower().strip() for v in views_raw]
    else:
        raise ValueError("run.video.views must be a string or list of strings")

    if not views:
        raise ValueError("run.video.views must contain at least one view")
    invalid = [v for v in views if v not in VIDEO_VIEW_NAMES]
    if invalid:
        raise ValueError(
            f"run.video.views contains unsupported values {invalid}; "
            f"use one of {sorted(VIDEO_VIEW_NAMES)}"
        )

    fps = float(video_cfg.get("fps", 30.0))
    if fps <= 0.0:
        raise ValueError("run.video.fps must be > 0")

    sim_fps = 1.0 / float(timestep)
    frame_stride_raw = video_cfg.get("frame_stride", None)
    if frame_stride_raw is None:
        frame_stride = max(1, int(round(sim_fps / fps)))
    else:
        frame_stride = int(frame_stride_raw)
        if frame_stride <= 0:
            raise ValueError("run.video.frame_stride must be > 0")

    return {
        "enabled": enabled,
        "views": views,
        "fps": fps,
        "frame_stride": frame_stride,
    }


def get_stmpc_reset_config(stmpc_cfg: dict, ref_speed: float) -> dict:
    """Build STMPC reset configuration."""
    init_cfg = stmpc_cfg.get("initial_state", {})
    return {
        "reset_with_states": bool(stmpc_cfg.get("reset_with_states", True)),
        "delta": float(init_cfg.get("delta", 0.0)),
        "speed": float(init_cfg.get("speed", ref_speed)),
        "yaw_rate": float(init_cfg.get("yaw_rate", 0.0)),
        "slip_angle": float(init_cfg.get("slip_angle", 0.0)),
    }


def apply_stmpc_runtime_tuning(bridge: STMPCGymBridge, stmpc_cfg: dict) -> list[str]:
    """Apply optional runtime tuning overrides to STMPC config values."""
    tuning = stmpc_cfg.get("runtime_tuning", {})
    if not bool(tuning.get("enabled", False)):
        return []

    allowed_fields = [
        "qn",
        "qalpha",
        "qv",
        "qjerk",
        "qddelta",
        "a_min",
        "a_max",
        "v_min",
        "v_max",
        "ddelta_min",
        "ddelta_max",
        "jerk_min",
        "jerk_max",
        "alat_max",
        "track_safety_margin",
    ]
    applied = []
    ctrl_cfg = bridge.controller.stmpc_config

    for field in allowed_fields:
        if field in tuning:
            old_val = getattr(ctrl_cfg, field)
            new_val = float(tuning[field])
            setattr(ctrl_cfg, field, new_val)
            applied.append(f"{field}:{old_val}->{new_val}")
    return applied


def count_by_value(values: list[str]) -> dict[str, int]:
    """Return frequency table for string values."""
    counts = {}
    for val in values:
        counts[val] = counts.get(val, 0) + 1
    return counts


def set_render_view(env: gym.Env, view: str) -> None:
    """Force supported renderers into a named camera view before grabbing a frame."""
    renderer = getattr(env.unwrapped, "renderer", None)
    if renderer is None:
        return

    if view == "global":
        renderer.follow_agent_flag = False
        renderer.agent_to_follow = None
        renderer.active_map_renderer = "map"
    elif view == "follow":
        renderer.follow_agent_flag = True
        renderer.agent_to_follow = 0
        renderer.active_map_renderer = "car"
    else:
        raise ValueError(f"Unsupported render view: {view}")


def write_video_frames(env: gym.Env, recorders: dict[str, VideoRecorder]) -> None:
    """Render and append one frame for each requested view."""
    for view, recorder in recorders.items():
        set_render_view(env, view)
        recorder.write(env.render())


def get_collect_env_config(cfg: dict, max_episode_steps: int, controller_cfg: dict, lidar_cfg: dict) -> dict:
    """Build gymnasium environment config for MPC data collection."""
    from gymkhana.envs import GKEnv

    env_cfg = cfg["env"]
    mode = controller_cfg["mode"]

    if mode == "stmpc":
        stmpc_cfg = controller_cfg["stmpc"]
        model = str(stmpc_cfg.get("model", "std"))
        default_obs_type = str(stmpc_cfg.get("observation_type", "frenet_dynamic_state"))
        training_mode = str(stmpc_cfg.get("training_mode", "race"))
        use_std_params = bool(stmpc_cfg.get("use_std_vehicle_params", True))
    else:
        kmpc_cfg = controller_cfg["kmpc"]
        model = str(kmpc_cfg.get("model", "ks"))
        default_obs_type = str(kmpc_cfg.get("observation_type", "kinematic_state"))
        training_mode = "race"
        use_std_params = False

    if lidar_cfg["enabled"]:
        observation_config = {"type": "features", "features": lidar_cfg["feature_list"]}
    else:
        observation_config = {"type": default_obs_type}

    config = {
        "map": env_cfg["map"],
        "num_agents": 1,
        "timestep": float(env_cfg["timestep"]),
        "integrator": str(env_cfg["integrator"]),
        "model": model,
        "control_input": ["speed", "steering_angle"],
        "observation_config": observation_config,
        "normalize_act": False,
        "normalize_obs": False,
        "training_mode": training_mode,
        "track_direction": str(env_cfg.get("track_direction", "normal")),
        "max_episode_steps": max_episode_steps,
    }
    if lidar_cfg["enabled"]:
        config["num_beams"] = int(lidar_cfg["num_beams"])
    if use_std_params:
        config["params"] = GKEnv.f1tenth_std_vehicle_params()
    return config


def obs_to_vec(obs: dict) -> np.ndarray:
    """Extract 5-dim kinematic state vector from observation dict."""
    agent_obs = obs["agent_0"]
    return np.array(
        [
            float(agent_obs["pose_x"]),
            float(agent_obs["pose_y"]),
            float(agent_obs["delta"]),
            float(agent_obs["linear_vel_x"]),
            float(agent_obs["pose_theta"]),
        ],
        dtype=np.float32,
    )


def extract_scan(obs: dict) -> np.ndarray:
    """Extract lidar scan vector from observation dict."""
    if "agent_0" in obs and "scan" in obs["agent_0"]:
        return np.asarray(obs["agent_0"]["scan"], dtype=np.float32)
    if "scans" in obs:
        return np.asarray(obs["scans"][0], dtype=np.float32)
    raise KeyError("Lidar scan not found in observation. Ensure lidar.enabled=true and observation_config supports scan.")


def bin_scan(scan: np.ndarray, n_bins: int, method: str) -> np.ndarray:
    """Downsample lidar scan into fixed bins using min/mean pooling."""
    n_beams = int(scan.shape[0])
    if n_bins >= n_beams:
        return scan.astype(np.float32, copy=True)

    edges = np.linspace(0, n_beams, n_bins + 1, dtype=np.int32)
    binned = np.empty((n_bins,), dtype=np.float32)
    for i in range(n_bins):
        left = int(edges[i])
        right = max(int(edges[i + 1]), left + 1)
        segment = scan[left:right]
        if method == "mean":
            binned[i] = float(np.mean(segment))
        else:
            binned[i] = float(np.min(segment))
    return binned


def process_scan(scan: np.ndarray, lidar_cfg: dict) -> np.ndarray:
    """Apply clipping and optional binning to lidar scan."""
    clipped = np.clip(scan, lidar_cfg["clip_min"], lidar_cfg["clip_max"]).astype(np.float32, copy=False)
    if lidar_cfg["binning"]["enabled"]:
        return bin_scan(clipped, lidar_cfg["binning"]["n_bins"], lidar_cfg["binning"]["method"])
    return clipped


class PerturbationManager:
    """Manages probabilistic perturbation (shove) injection and cooldown."""

    def __init__(
        self,
        probability: float,
        shove_magnitude: float,
        min_steps_between: int,
        hold_steps_min: int,
        hold_steps_max: int,
    ):
        """
        Args:
            probability: Probability of perturbation at each step [0.0-1.0]
            shove_magnitude: Magnitude of steering shove (radians)
            min_steps_between: Minimum steps between perturbations (cooldown)
            hold_steps_min: Minimum number of steps to hold a shove once triggered
            hold_steps_max: Maximum number of steps to hold a shove once triggered
        """
        self.probability = probability
        self.shove_magnitude = shove_magnitude
        self.min_steps_between = min_steps_between
        self.last_shove_step = -min_steps_between  # Allow first shove immediately
        self.hold_steps_min = hold_steps_min
        self.hold_steps_max = hold_steps_max
        self.active_noise = np.zeros(2, dtype=np.float32)
        self.active_steps_remaining = 0

    def should_perturb(self, current_step: int) -> bool:
        """Determine if we should apply perturbation at this step."""
        if self.active_steps_remaining > 0:
            return True

        if current_step - self.last_shove_step < self.min_steps_between:
            return False
        return np.random.random() < self.probability

    def apply_shove(self, action: np.ndarray, current_step: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply steering shove to action and record noise vector.

        Returns:
            (perturbed_action, noise_vector)
        """
        if self.active_steps_remaining > 0:
            noise = self.active_noise
            self.active_steps_remaining -= 1
            return action + noise, noise

        noise = np.array([np.random.randn() * self.shove_magnitude, 0.0], dtype=np.float32)
        self.active_noise = noise
        hold_steps = int(np.random.randint(self.hold_steps_min, self.hold_steps_max + 1))
        self.active_steps_remaining = max(0, hold_steps - 1)
        self.last_shove_step = current_step
        return action + noise, noise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect MPC transitions with YAML config, lap termination, and DAgger labeling"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to YAML config file (default: {get_default_config_path()})",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (overrides config setting)",
    )
    return parser.parse_args()


def main() -> None:
    """Main data collection loop."""
    args = parse_args()

    # Load configuration
    config_path = args.config or get_default_config_path()
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        print(f"Default config location: {get_default_config_path()}")
        sys.exit(1)

    cfg = load_config(config_path)
    controller_cfg = get_controller_config(cfg)
    lidar_cfg = get_lidar_config(cfg, controller_cfg)
    path_cfg = get_path_feature_config(cfg)
    lap_cfg = get_lap_termination_config(cfg["run"])
    video_cfg = get_video_config(cfg, float(cfg["env"].get("timestep", 0.01)))
    if lap_cfg["method"] != "s_wrap":
        print(f"WARNING: Unsupported lap_termination.method='{lap_cfg['method']}', using 's_wrap'")
        lap_cfg["method"] = "s_wrap"

    # Override render from CLI if specified
    if args.render:
        cfg["run"]["render"] = True

    output_dir = Path(cfg["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if "output_filename" in cfg["output"] and cfg["output"]["output_filename"]:
        out_path = output_dir / cfg["output"]["output_filename"]
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = output_dir / f"{controller_cfg['mode']}_{cfg['env']['map']}_{ts}.npz"

    print(f"Loaded config from: {config_path}")
    print(f"Collection settings: {cfg['run']['episodes']} episodes, {cfg['run']['steps_per_episode']} max steps per")
    print(
        f"Controller mode: {controller_cfg['mode']}, ref_speed={controller_cfg['ref_speed']:.2f} m/s"
    )
    if lidar_cfg["enabled"]:
        binning_desc = (
            f"binning={lidar_cfg['binning']['method']}->{lidar_cfg['binning']['n_bins']}"
            if lidar_cfg["binning"]["enabled"]
            else "binning=disabled"
        )
        print(
            "Lidar collection: "
            f"enabled=True, env_num_beams={lidar_cfg['num_beams']}, {binning_desc}, dtype={lidar_cfg['dtype']}"
        )
    if lap_cfg["enabled"]:
        print(
            "Lap termination: "
            f"enabled=True, target_laps={lap_cfg['target_laps']}, method={lap_cfg['method']}"
        )
    if path_cfg["enabled"]:
        print(
            "Path features: "
            f"enabled=True, lookahead_m={path_cfg['lookahead_m'].tolist()}, "
            f"include_current_kappa={path_cfg['include_current_kappa']}, dtype={path_cfg['dtype']}"
        )
    if video_cfg["enabled"]:
        print(
            "Video recording: "
            f"enabled=True, views={video_cfg['views']}, fps={video_cfg['fps']:.1f}, "
            f"frame_stride={video_cfg['frame_stride']}"
        )

    # Initialize environment
    env_config = get_collect_env_config(cfg, cfg["run"]["steps_per_episode"], controller_cfg, lidar_cfg)
    render_mode = "rgb_array" if video_cfg["enabled"] else "human" if cfg["run"]["render"] else None
    env = gym.make("gymkhana:gymkhana-v0", config=env_config, render_mode=render_mode)
    if controller_cfg["mode"] == "stmpc":
        stmpc_cfg = controller_cfg["stmpc"]
        startup_speed_offset = float(stmpc_cfg.get("startup_speed_offset", 3.0))
        bridge = STMPCGymBridge(
            env,
            ref_speed=controller_cfg["ref_speed"],
            startup_speed_offset=startup_speed_offset,
        )
        tuning_applied = apply_stmpc_runtime_tuning(bridge, stmpc_cfg)
        if tuning_applied:
            print("STMPC runtime tuning overrides: " + ", ".join(tuning_applied))
        stmpc_reset_cfg = get_stmpc_reset_config(stmpc_cfg, controller_cfg["ref_speed"])
    else:
        speed_profile_cfg = controller_cfg["kmpc"].get("speed_profile", {})
        bridge = KMPCGymBridge(
            env,
            ref_speed=controller_cfg["ref_speed"],
            speed_profile=speed_profile_cfg,
        )
        stmpc_reset_cfg = None

    path_centerline_ss = None
    path_centerline_ks = None
    path_track_length = 0.0
    path_centerline_size = 0
    if path_cfg["enabled"]:
        centerline = env.unwrapped.track.centerline
        path_centerline_ss = np.asarray(centerline.ss, dtype=np.float64)
        path_centerline_ks = np.asarray(centerline.ks, dtype=np.float64)
        path_track_length = float(centerline.spline.s[-1])
        path_centerline_size = int(path_centerline_ss.shape[0])

        if path_centerline_ss.ndim != 1 or path_centerline_ks.ndim != 1:
            raise ValueError("Centerline ss/ks arrays must be 1-D")
        if path_centerline_size == 0 or path_centerline_size != int(path_centerline_ks.shape[0]):
            raise ValueError("Centerline ss/ks arrays must be non-empty and same length")
        if np.any(np.diff(path_centerline_ss) <= 0):
            raise ValueError("Centerline ss must be strictly increasing for searchsorted lookup")

        def lookup_path_curvature_features(s_anchor: float) -> tuple[int, np.ndarray, float]:
            s_anchor_wrapped = float(s_anchor) % path_track_length

            idx_anchor = int(np.searchsorted(path_centerline_ss, s_anchor_wrapped, side="left"))
            if idx_anchor >= path_centerline_size:
                idx_anchor = 0

            s_queries = np.mod(s_anchor_wrapped + path_cfg["lookahead_m"], path_track_length)
            idxs = np.searchsorted(path_centerline_ss, s_queries, side="left")
            idxs = np.where(idxs >= path_centerline_size, 0, idxs).astype(np.int32)
            kappa_values = path_centerline_ks[idxs]
            return idx_anchor, kappa_values, s_anchor_wrapped

    effective_steps_per_episode = int(cfg["run"]["steps_per_episode"])
    if lap_cfg["enabled"] and lap_cfg["auto_timeout"]["enabled"]:
        track_length = float(env.unwrapped.track.centerline.spline.s[-1])
        timestep = float(env_config["timestep"])
        ref_speed = float(controller_cfg["ref_speed"])
        min_speed_ratio = max(0.05, lap_cfg["auto_timeout"]["min_expected_speed_ratio"])
        expected_progress_per_step = max(ref_speed * timestep * min_speed_ratio, 1e-4)
        base_steps = (lap_cfg["target_laps"] * track_length) / expected_progress_per_step
        dynamic_timeout = int(np.ceil(base_steps * lap_cfg["auto_timeout"]["timeout_factor"]))
        effective_steps_per_episode = max(effective_steps_per_episode, dynamic_timeout)
        env.unwrapped.max_episode_steps = effective_steps_per_episode
        print(
            "Lap timeout auto-sizing: "
            f"track_length={track_length:.2f}, timeout_steps={effective_steps_per_episode}"
        )

    # Initialize perturbation manager if enabled
    perturb_cfg = cfg.get("perturbation", {})
    perturbation_mgr = None
    if bool(perturb_cfg.get("enabled", False)):
        hold_steps_min = int(perturb_cfg.get("hold_steps_min", 1))
        hold_steps_max = int(perturb_cfg.get("hold_steps_max", hold_steps_min))
        if hold_steps_min < 1:
            raise ValueError("perturbation.hold_steps_min must be >= 1")
        if hold_steps_max < hold_steps_min:
            raise ValueError("perturbation.hold_steps_max must be >= perturbation.hold_steps_min")

        perturbation_mgr = PerturbationManager(
            probability=float(perturb_cfg.get("probability", 0.2)),
            shove_magnitude=float(perturb_cfg.get("shove_magnitude", 0.3)),
            min_steps_between=int(perturb_cfg.get("min_steps_between_shoves", 0)),
            hold_steps_min=hold_steps_min,
            hold_steps_max=hold_steps_max,
        )
        print(
            "Perturbations enabled: "
            f"{float(perturb_cfg.get('probability', 0.2)):.1%} probability, "
            f"hold_steps=[{hold_steps_min}, {hold_steps_max}]"
        )

    # Data collection arrays
    observations = []
    next_observations = []
    expert_actions = []  # MPC-computed actions
    executed_actions = []  # Actions actually taken (may include noise)
    rewards = []
    terminations = []
    truncations = []
    episode_ids = []
    step_ids = []
    is_perturbed_steps = []
    collector_lap_counts = []
    env_lap_counts = []
    episode_end_reasons = []
    collision_flags = []
    boundary_flags = []
    stmpc_status_codes = [] if controller_cfg["mode"] == "stmpc" else None
    noise_vectors = [] if bool(perturb_cfg.get("enabled", False)) and cfg["dagger"]["record_noise_vectors"] else None
    lidar_observations = [] if lidar_cfg["enabled"] else None
    lidar_next_observations = [] if lidar_cfg["enabled"] else None
    raw_lidar_observations = [] if lidar_cfg["enabled"] and lidar_cfg["store_raw"] else None
    raw_lidar_next_observations = [] if lidar_cfg["enabled"] and lidar_cfg["store_raw"] else None
    path_curvature_lookahead = [] if path_cfg["enabled"] else None
    path_s_anchor = [] if path_cfg["enabled"] else None
    path_centerline_idx_anchor = [] if path_cfg["enabled"] else None
    path_kappa_current = [] if path_cfg["enabled"] and path_cfg["include_current_kappa"] else None
    video_recorders = {}
    if video_cfg["enabled"]:
        video_recorders = {
            view: VideoRecorder(out_path.with_name(f"{out_path.stem}_{view}.mp4"), video_cfg["fps"])
            for view in video_cfg["views"]
        }

    try:
        global_step = 0  # Track global step count across episodes
        for ep_idx in range(cfg["run"]["episodes"]):
            # Reset environment
            if cfg["run"]["use_default_reset"]:
                obs, _ = env.reset()
            else:
                x0, y0, yaw0 = bridge.get_start_pose()
                if controller_cfg["mode"] == "stmpc" and stmpc_reset_cfg and stmpc_reset_cfg["reset_with_states"]:
                    init_states = np.array(
                        [
                            [
                                x0,
                                y0,
                                stmpc_reset_cfg["delta"],
                                stmpc_reset_cfg["speed"],
                                yaw0,
                                stmpc_reset_cfg["yaw_rate"],
                                stmpc_reset_cfg["slip_angle"],
                            ]
                        ],
                        dtype=np.float64,
                    )
                    obs, _ = env.reset(options={"states": init_states})
                else:
                    obs, _ = env.reset(options={"poses": np.array([[x0, y0, yaw0]])})

            if hasattr(bridge, "init_from_obs"):
                bridge.init_from_obs(obs)

            ep_reward = 0.0
            ep_step = 0
            end_reason = "unknown"
            lap_tracker = LapTracker(env.unwrapped.track, wrap_threshold_ratio=lap_cfg["wrap_threshold_ratio"])
            lap_tracker.reset(float(obs["agent_0"]["pose_x"]), float(obs["agent_0"]["pose_y"]))
            current_s_for_teacher = float(lap_tracker.prev_s)
            collector_laps = 0
            last_env_laps = 0
            last_collision = False
            last_boundary = False
            terminated = False
            truncated = False
            for step_idx in range(effective_steps_per_episode):
                obs_vec = obs_to_vec(obs)
                if path_cfg["enabled"]:
                    idx_anchor, kappa_lookahead_vals, s_anchor = lookup_path_curvature_features(current_s_for_teacher)
                    path_s_anchor.append(s_anchor)
                    path_centerline_idx_anchor.append(idx_anchor)
                    path_curvature_lookahead.append(kappa_lookahead_vals.astype(np.float32, copy=False))
                    if path_kappa_current is not None:
                        path_kappa_current.append(float(path_centerline_ks[idx_anchor]))
                if lidar_cfg["enabled"]:
                    scan = extract_scan(obs)
                    lidar_observations.append(process_scan(scan, lidar_cfg))
                    if raw_lidar_observations is not None:
                        raw_lidar_observations.append(scan.astype(np.float32, copy=False))
                expert_action = bridge.get_action(obs)
                expert_action_vec = expert_action[0].astype(np.float32, copy=False)

                # Determine if this step should be perturbed
                is_perturbed = False
                noise = np.zeros(2, dtype=np.float32)
                if perturbation_mgr and perturbation_mgr.should_perturb(global_step):
                    executed_action_vec, noise = perturbation_mgr.apply_shove(expert_action_vec, global_step)
                    is_perturbed = True
                else:
                    executed_action_vec = expert_action_vec

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(
                    np.array([executed_action_vec])
                )
                next_obs_vec = obs_to_vec(next_obs)
                if lidar_cfg["enabled"]:
                    next_scan = extract_scan(next_obs)
                    lidar_next_observations.append(process_scan(next_scan, lidar_cfg))
                    if raw_lidar_next_observations is not None:
                        raw_lidar_next_observations.append(next_scan.astype(np.float32, copy=False))
                collector_laps, wrapped, _ = lap_tracker.update(
                    float(next_obs["agent_0"]["pose_x"]),
                    float(next_obs["agent_0"]["pose_y"]),
                )
                current_s_for_teacher = float(lap_tracker.prev_s)
                last_env_laps = int(float(info.get("lap_counts", 0)))
                last_collision = bool(info.get("collision", False))
                last_boundary = bool(info.get("boundary_exceeded", False))

                if video_recorders and global_step % video_cfg["frame_stride"] == 0:
                    write_video_frames(env, video_recorders)
                elif cfg["run"]["render"]:
                    env.render()

                # Record transition
                observations.append(obs_vec)
                next_observations.append(next_obs_vec)
                expert_actions.append(expert_action_vec)
                executed_actions.append(executed_action_vec)
                rewards.append(np.float32(reward))
                terminations.append(bool(terminated))
                truncations.append(bool(truncated))
                is_perturbed_steps.append(bool(is_perturbed))
                collector_lap_counts.append(collector_laps)
                env_lap_counts.append(last_env_laps)
                collision_flags.append(last_collision)
                boundary_flags.append(last_boundary)
                episode_ids.append(ep_idx)
                step_ids.append(step_idx)
                if stmpc_status_codes is not None:
                    stmpc_status_codes.append(int(getattr(bridge, "last_status", 0)))

                if noise_vectors is not None:
                    noise_vectors.append(noise)

                ep_reward += reward
                obs = next_obs
                ep_step = step_idx + 1
                global_step += 1

                if lap_cfg["enabled"] and collector_laps >= lap_cfg["target_laps"]:
                    end_reason = "lap_target_reached"
                    print(
                        f"  Episode {ep_idx + 1}: collector_laps={collector_laps} reached target "
                        f"{lap_cfg['target_laps']}, terminating"
                    )
                    break

                if terminated or truncated:
                    if terminated:
                        if last_boundary:
                            end_reason = "boundary_terminated"
                        elif last_collision:
                            end_reason = "collision_terminated"
                        else:
                            end_reason = "env_terminated"
                    elif truncated:
                        end_reason = "env_truncated"
                    break

            if end_reason == "unknown":
                end_reason = "step_limit_reached"
            episode_end_reasons.append(end_reason)

            print(
                f"Episode {ep_idx + 1}/{cfg['run']['episodes']}: "
                f"steps={ep_step}, collector_laps={collector_laps}, env_laps={last_env_laps}, "
                f"reward={ep_reward:.2f}, terminated={terminated}, truncated={truncated}, "
                f"collision={last_collision}, boundary={last_boundary}, "
                f"end_reason={end_reason}"
            )

    finally:
        for recorder in video_recorders.values():
            recorder.close()
        env.close()

    video_paths = {
        view: str(recorder.path)
        for view, recorder in video_recorders.items()
    } if video_cfg["enabled"] else {}
    video_codecs = {
        view: recorder.codec
        for view, recorder in video_recorders.items()
    } if video_cfg["enabled"] else {}

    # Save dataset
    # Stack arrays
    obs_arr = np.stack(observations).astype(np.float32)
    next_obs_arr = np.stack(next_observations).astype(np.float32)
    expert_act_arr = np.stack(expert_actions).astype(np.float32)
    executed_act_arr = np.stack(executed_actions).astype(np.float32)
    rew_arr = np.asarray(rewards, dtype=np.float32)
    term_arr = np.asarray(terminations, dtype=np.bool_)
    trunc_arr = np.asarray(truncations, dtype=np.bool_)
    ep_arr = np.asarray(episode_ids, dtype=np.int32)
    step_arr = np.asarray(step_ids, dtype=np.int32)
    perturbed_arr = np.asarray(is_perturbed_steps, dtype=np.bool_)
    collector_laps_arr = np.asarray(collector_lap_counts, dtype=np.int32)
    env_laps_arr = np.asarray(env_lap_counts, dtype=np.int32)
    collision_arr = np.asarray(collision_flags, dtype=np.bool_)
    boundary_arr = np.asarray(boundary_flags, dtype=np.bool_)

    lidar_dtype = np.float16 if lidar_cfg["dtype"] == "float16" else np.float32
    lidar_arr = None
    lidar_next_arr = None
    raw_lidar_arr = None
    raw_lidar_next_arr = None
    path_curvature_lookahead_arr = None
    path_s_anchor_arr = None
    path_centerline_idx_anchor_arr = None
    path_kappa_current_arr = None
    if lidar_cfg["enabled"]:
        lidar_arr = np.stack(lidar_observations).astype(lidar_dtype)
        lidar_next_arr = np.stack(lidar_next_observations).astype(lidar_dtype)
        if raw_lidar_observations is not None:
            raw_lidar_arr = np.stack(raw_lidar_observations).astype(lidar_dtype)
            raw_lidar_next_arr = np.stack(raw_lidar_next_observations).astype(lidar_dtype)

    if path_cfg["enabled"]:
        path_dtype = np.float16 if path_cfg["dtype"] == "float16" else np.float32
        path_curvature_lookahead_arr = np.stack(path_curvature_lookahead).astype(path_dtype)
        path_s_anchor_arr = np.asarray(path_s_anchor, dtype=path_dtype)
        path_centerline_idx_anchor_arr = np.asarray(path_centerline_idx_anchor, dtype=np.int32)
        if path_kappa_current is not None:
            path_kappa_current_arr = np.asarray(path_kappa_current, dtype=path_dtype)

        n_transitions = int(obs_arr.shape[0])
        if path_curvature_lookahead_arr.shape[0] != n_transitions:
            raise ValueError("Path curvature lookahead rows must match transition count")
        if path_s_anchor_arr.shape[0] != n_transitions:
            raise ValueError("Path s anchors must match transition count")
        if path_centerline_idx_anchor_arr.shape[0] != n_transitions:
            raise ValueError("Path centerline idx anchors must match transition count")
        if path_kappa_current_arr is not None and path_kappa_current_arr.shape[0] != n_transitions:
            raise ValueError("Path current curvature values must match transition count")

    # Build save dict
    save_dict = {
        "observations": obs_arr,
        "next_observations": next_obs_arr,
        "expert_actions": expert_act_arr,
        "executed_actions": executed_act_arr,
        "rewards": rew_arr,
        "terminations": term_arr,
        "truncations": trunc_arr,
        "is_perturbed": perturbed_arr,
        "collector_lap_counts": collector_laps_arr,
        "env_lap_counts": env_laps_arr,
        "collision_flags": collision_arr,
        "boundary_flags": boundary_arr,
        "episode_ids": ep_arr,
        "step_ids": step_arr,
        "feature_names": np.asarray(FEATURE_NAMES),
        "action_names": np.asarray(ACTION_NAMES),
    }

    if stmpc_status_codes is not None:
        save_dict["stmpc_status_codes"] = np.asarray(stmpc_status_codes, dtype=np.int32)

    if noise_vectors is not None and bool(perturb_cfg.get("enabled", False)):
        noise_arr = np.stack(noise_vectors).astype(np.float32)
        save_dict["noise_vectors"] = noise_arr

    if lidar_cfg["enabled"] and lidar_arr is not None and lidar_next_arr is not None:
        save_dict["lidar_scans"] = lidar_arr
        save_dict["next_lidar_scans"] = lidar_next_arr
        save_dict["lidar_names"] = np.asarray(
            [f"lidar_{i}" for i in range(int(lidar_arr.shape[1]))],
            dtype="U32",
        )
        if raw_lidar_arr is not None and raw_lidar_next_arr is not None:
            save_dict["raw_lidar_scans"] = raw_lidar_arr
            save_dict["next_raw_lidar_scans"] = raw_lidar_next_arr

    if path_cfg["enabled"] and path_curvature_lookahead_arr is not None:
        def _fmt_kappa_name(distance_m: float) -> str:
            token = f"{float(distance_m):.3f}".rstrip("0").rstrip(".").replace(".", "p")
            return f"kappa_{token}m"

        save_dict["path_curvature_lookahead"] = path_curvature_lookahead_arr
        save_dict["path_curvature_lookahead_m"] = path_cfg["lookahead_m"].astype(path_curvature_lookahead_arr.dtype)
        save_dict["path_curvature_lookahead_names"] = np.asarray(
            [_fmt_kappa_name(d) for d in path_cfg["lookahead_m"]],
            dtype="U32",
        )
        save_dict["path_s_anchor"] = path_s_anchor_arr
        save_dict["path_centerline_idx_anchor"] = path_centerline_idx_anchor_arr
        if path_kappa_current_arr is not None:
            save_dict["path_kappa_current"] = path_kappa_current_arr

    np.savez_compressed(out_path, **save_dict)

    # Save metadata
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "map": cfg["env"]["map"],
        "track_direction": env_config["track_direction"],
        "vehicle_model": env_config["model"],
        "episodes_requested": cfg["run"]["episodes"],
        "steps_per_episode": cfg["run"]["steps_per_episode"],
        "effective_steps_per_episode": effective_steps_per_episode,
        "lap_termination": lap_cfg,
        "controller_mode": controller_cfg["mode"],
        "ref_speed": controller_cfg["ref_speed"],
        "controller_config": {
            "kmpc": controller_cfg["kmpc"],
            "stmpc": controller_cfg["stmpc"],
        },
        "stmpc_reset_config": stmpc_reset_cfg,
        "perturbation_enabled": bool(perturb_cfg.get("enabled", False)),
        "perturbation_probability": float(perturb_cfg.get("probability", 0.2)) if bool(perturb_cfg.get("enabled", False)) else None,
        "perturbation_shove_magnitude": float(perturb_cfg.get("shove_magnitude", 0.3)) if bool(perturb_cfg.get("enabled", False)) else None,
        "perturbation_min_steps_between_shoves": int(perturb_cfg.get("min_steps_between_shoves", 0)) if bool(perturb_cfg.get("enabled", False)) else None,
        "perturbation_hold_steps_min": int(perturb_cfg.get("hold_steps_min", 1)) if bool(perturb_cfg.get("enabled", False)) else None,
        "perturbation_hold_steps_max": int(perturb_cfg.get("hold_steps_max", int(perturb_cfg.get("hold_steps_min", 1)))) if bool(perturb_cfg.get("enabled", False)) else None,
        "dagger_enabled": cfg["dagger"]["enabled"],
        "num_transitions": int(obs_arr.shape[0]),
        "num_perturbed_steps": int(np.sum(perturbed_arr)),
        "num_collision_steps": int(np.sum(collision_arr)),
        "num_boundary_steps": int(np.sum(boundary_arr)),
        "episode_end_reasons": episode_end_reasons,
        "episode_end_reason_counts": count_by_value(episode_end_reasons),
        "obs_shape": list(obs_arr.shape),
        "expert_act_shape": list(expert_act_arr.shape),
        "executed_act_shape": list(executed_act_arr.shape),
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "lidar": {
            "enabled": lidar_cfg["enabled"],
            "env_num_beams": lidar_cfg["num_beams"],
            "clip_min": lidar_cfg["clip_min"],
            "clip_max": lidar_cfg["clip_max"],
            "store_raw": lidar_cfg["store_raw"],
            "dtype": lidar_cfg["dtype"],
            "binning": lidar_cfg["binning"],
            "lidar_shape": list(lidar_arr.shape) if lidar_arr is not None else None,
            "next_lidar_shape": list(lidar_next_arr.shape) if lidar_next_arr is not None else None,
            "raw_lidar_shape": list(raw_lidar_arr.shape) if raw_lidar_arr is not None else None,
            "next_raw_lidar_shape": list(raw_lidar_next_arr.shape) if raw_lidar_next_arr is not None else None,
        },
        "video": {
            "enabled": video_cfg["enabled"],
            "views": video_cfg["views"],
            "fps": video_cfg["fps"],
            "frame_stride": video_cfg["frame_stride"],
            "paths": video_paths,
            "codecs": video_codecs,
        },
    }

    if path_cfg["enabled"] and path_curvature_lookahead_arr is not None:
        metadata["path_features"] = {
            "enabled": True,
            "lookahead_m": [float(v) for v in path_cfg["lookahead_m"].tolist()],
            "include_current_kappa": path_cfg["include_current_kappa"],
            "dtype": path_cfg["dtype"],
            "path_curvature_lookahead_shape": list(path_curvature_lookahead_arr.shape),
            "path_s_anchor_shape": list(path_s_anchor_arr.shape),
            "path_centerline_idx_anchor_shape": list(path_centerline_idx_anchor_arr.shape),
            "path_kappa_current_shape": list(path_kappa_current_arr.shape) if path_kappa_current_arr is not None else None,
        }

    if stmpc_status_codes is not None:
        status_arr = np.asarray(stmpc_status_codes, dtype=np.int32)
        metadata["stmpc_solver_fail_steps"] = int(np.sum(status_arr != 0))

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nDataset saved: {out_path}")
    print(f"Metadata saved: {meta_path}")
    if video_cfg["enabled"]:
        for view, path in video_paths.items():
            print(f"Video saved ({view}): {path}")
    print(
        f"Total transitions: {obs_arr.shape[0]} "
        f"({int(np.sum(perturbed_arr))} perturbed, {int(np.sum(~perturbed_arr))} clean)"
    )


if __name__ == "__main__":
    main()
