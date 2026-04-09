"""Collect transition data by rolling out MPC controllers with advanced features.

Features:
  - YAML-based configuration management
  - Lap-based episode termination
  - Probabilistic perturbation injection (80% clean, 20% perturbed)
  - DAgger-style dual action labeling (expert vs. executed)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

import gymkhana  # noqa: F401  # ensures gym env registration

# Avoid importing through controllers package __init__, which currently pulls stale RL modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "controllers"))
from mpc.gym_bridge import KMPCGymBridge, STMPCGymBridge

FEATURE_NAMES = [
    "pose_x",
    "pose_y",
    "delta",
    "linear_vel_x",
    "pose_theta",
]

ACTION_NAMES = ["steering_angle", "speed"]


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


def get_collect_env_config(cfg: dict, max_episode_steps: int, controller_cfg: dict) -> dict:
    """Build gymnasium environment config for MPC data collection."""
    from gymkhana.envs import GKEnv

    env_cfg = cfg["env"]
    mode = controller_cfg["mode"]

    if mode == "stmpc":
        stmpc_cfg = controller_cfg["stmpc"]
        model = str(stmpc_cfg.get("model", "std"))
        obs_type = str(stmpc_cfg.get("observation_type", "frenet_dynamic_state"))
        training_mode = str(stmpc_cfg.get("training_mode", "race"))
        use_std_params = bool(stmpc_cfg.get("use_std_vehicle_params", True))
    else:
        kmpc_cfg = controller_cfg["kmpc"]
        model = str(kmpc_cfg.get("model", "ks"))
        obs_type = str(kmpc_cfg.get("observation_type", "kinematic_state"))
        training_mode = "race"
        use_std_params = False

    config = {
        "map": env_cfg["map"],
        "num_agents": 1,
        "timestep": float(env_cfg["timestep"]),
        "integrator": str(env_cfg["integrator"]),
        "model": model,
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": obs_type},
        "normalize_act": False,
        "normalize_obs": False,
        "training_mode": training_mode,
        "track_direction": "normal",
        "max_episode_steps": max_episode_steps,
    }
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


class PerturbationManager:
    """Manages probabilistic perturbation (shove) injection and cooldown."""

    def __init__(self, probability: float, shove_magnitude: float, min_steps_between: int):
        """
        Args:
            probability: Probability of perturbation at each step [0.0-1.0]
            shove_magnitude: Magnitude of steering shove (radians)
            min_steps_between: Minimum steps between perturbations (cooldown)
        """
        self.probability = probability
        self.shove_magnitude = shove_magnitude
        self.min_steps_between = min_steps_between
        self.last_shove_step = -min_steps_between  # Allow first shove immediately

    def should_perturb(self, current_step: int) -> bool:
        """Determine if we should apply perturbation at this step."""
        if current_step - self.last_shove_step < self.min_steps_between:
            return False
        return np.random.random() < self.probability

    def apply_shove(self, action: np.ndarray, current_step: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply steering shove to action and record noise vector.

        Returns:
            (perturbed_action, noise_vector)
        """
        noise = np.array([np.random.randn() * self.shove_magnitude, 0.0], dtype=np.float32)
        perturbed = action + noise
        self.last_shove_step = current_step
        return perturbed, noise


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
    lap_cfg = get_lap_termination_config(cfg["run"])
    if lap_cfg["method"] != "s_wrap":
        print(f"WARNING: Unsupported lap_termination.method='{lap_cfg['method']}', using 's_wrap'")
        lap_cfg["method"] = "s_wrap"

    # Override render from CLI if specified
    if args.render:
        cfg["run"]["render"] = True

    print(f"Loaded config from: {config_path}")
    print(f"Collection settings: {cfg['run']['episodes']} episodes, {cfg['run']['steps_per_episode']} max steps per")
    print(
        f"Controller mode: {controller_cfg['mode']}, ref_speed={controller_cfg['ref_speed']:.2f} m/s"
    )
    if lap_cfg["enabled"]:
        print(
            "Lap termination: "
            f"enabled=True, target_laps={lap_cfg['target_laps']}, method={lap_cfg['method']}"
        )

    # Initialize environment
    env_config = get_collect_env_config(cfg, cfg["run"]["steps_per_episode"], controller_cfg)
    render_mode = "human" if cfg["run"]["render"] else None
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
    perturbation_mgr = None
    if cfg["perturbation"]["enabled"]:
        perturbation_mgr = PerturbationManager(
            probability=cfg["perturbation"]["probability"],
            shove_magnitude=cfg["perturbation"]["shove_magnitude"],
            min_steps_between=cfg["perturbation"]["min_steps_between_shoves"],
        )
        print(f"Perturbations enabled: {cfg['perturbation']['probability']:.1%} probability")

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
    noise_vectors = [] if cfg["perturbation"]["enabled"] and cfg["dagger"]["record_noise_vectors"] else None

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
            collector_laps = 0
            last_env_laps = 0
            last_collision = False
            last_boundary = False
            terminated = False
            truncated = False
            for step_idx in range(effective_steps_per_episode):
                obs_vec = obs_to_vec(obs)
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
                collector_laps, wrapped, _ = lap_tracker.update(
                    float(next_obs["agent_0"]["pose_x"]),
                    float(next_obs["agent_0"]["pose_y"]),
                )
                last_env_laps = int(float(info.get("lap_counts", 0)))
                last_collision = bool(info.get("collision", False))
                last_boundary = bool(info.get("boundary_exceeded", False))

                if cfg["run"]["render"]:
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
        env.close()

    # Save dataset
    output_dir = Path(cfg["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if "output_filename" in cfg["output"] and cfg["output"]["output_filename"]:
        out_path = output_dir / cfg["output"]["output_filename"]
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = output_dir / f"{controller_cfg['mode']}_{cfg['env']['map']}_{ts}.npz"

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

    if noise_vectors is not None and cfg["perturbation"]["enabled"]:
        noise_arr = np.stack(noise_vectors).astype(np.float32)
        save_dict["noise_vectors"] = noise_arr

    np.savez_compressed(out_path, **save_dict)

    # Save metadata
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "map": cfg["env"]["map"],
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
        "perturbation_enabled": cfg["perturbation"]["enabled"],
        "perturbation_probability": cfg["perturbation"]["probability"] if cfg["perturbation"]["enabled"] else None,
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
    }

    if stmpc_status_codes is not None:
        status_arr = np.asarray(stmpc_status_codes, dtype=np.int32)
        metadata["stmpc_solver_fail_steps"] = int(np.sum(status_arr != 0))

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nDataset saved: {out_path}")
    print(f"Metadata saved: {meta_path}")
    print(
        f"Total transitions: {obs_arr.shape[0]} "
        f"({int(np.sum(perturbed_arr))} perturbed, {int(np.sum(~perturbed_arr))} clean)"
    )


if __name__ == "__main__":
    main()
