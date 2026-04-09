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
from mpc.gym_bridge import KMPCGymBridge

FEATURE_NAMES = [
    "pose_x",
    "pose_y",
    "delta",
    "linear_vel_x",
    "pose_theta",
]

ACTION_NAMES = ["steering_angle", "speed"]


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_default_config_path() -> Path:
    """Return path to default config file."""
    return Path(__file__).resolve().parent.parent / "configs" / "collect_mpc_default.yaml"


def get_kmpc_collect_config(map_name: str, max_episode_steps: int) -> dict:
    """Build gymnasium environment config for KMPC data collection."""
    return {
        "map": map_name,
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
        "max_episode_steps": max_episode_steps,
    }


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
        description="Collect KMPC transitions with YAML config, lap termination, and DAgger labeling"
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

    # Override render from CLI if specified
    if args.render:
        cfg["run"]["render"] = True

    print(f"Loaded config from: {config_path}")
    print(f"Collection settings: {cfg['run']['episodes']} episodes, {cfg['run']['steps_per_episode']} max steps per")

    # Initialize environment
    env_config = get_kmpc_collect_config(cfg["env"]["map"], cfg["run"]["steps_per_episode"])
    render_mode = "human" if cfg["run"]["render"] else None
    env = gym.make("gymkhana:gymkhana-v0", config=env_config, render_mode=render_mode)
    bridge = KMPCGymBridge(env, ref_speed=cfg["controller"]["ref_speed"])

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
    noise_vectors = [] if cfg["perturbation"]["enabled"] and cfg["dagger"]["record_noise_vectors"] else None

    try:
        global_step = 0  # Track global step count across episodes
        for ep_idx in range(cfg["run"]["episodes"]):
            # Reset environment
            if cfg["run"]["use_default_reset"]:
                obs, _ = env.reset()
            else:
                x0, y0, yaw0 = bridge.get_start_pose()
                obs, _ = env.reset(options={"poses": np.array([[x0, y0, yaw0]])})

            ep_reward = 0.0
            ep_step = 0

            for step_idx in range(cfg["run"]["steps_per_episode"]):
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
                episode_ids.append(ep_idx)
                step_ids.append(step_idx)

                if noise_vectors is not None:
                    noise_vectors.append(noise)

                ep_reward += reward
                obs = next_obs
                ep_step = step_idx + 1
                global_step += 1

                # Check lap-based termination
                current_laps = info.get("lap_counts", 0)
                target_laps = 1  # Collect one lap per episode
                if current_laps >= target_laps:
                    print(f"  Episode {ep_idx + 1}: Lap {current_laps} completed, terminating")
                    break

                if terminated or truncated:
                    break

            print(
                f"Episode {ep_idx + 1}/{cfg['run']['episodes']}: "
                f"steps={ep_step}, laps={info.get('lap_counts', 0)}, "
                f"reward={ep_reward:.2f}, terminated={terminated}, truncated={truncated}"
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
        out_path = output_dir / f"kmpc_{cfg['env']['map']}_{ts}.npz"

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
        "episode_ids": ep_arr,
        "step_ids": step_arr,
        "feature_names": np.asarray(FEATURE_NAMES),
        "action_names": np.asarray(ACTION_NAMES),
    }

    if noise_vectors is not None and cfg["perturbation"]["enabled"]:
        noise_arr = np.stack(noise_vectors).astype(np.float32)
        save_dict["noise_vectors"] = noise_arr

    np.savez_compressed(out_path, **save_dict)

    # Save metadata
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "map": cfg["env"]["map"],
        "vehicle_model": cfg["env"]["model"],
        "episodes_requested": cfg["run"]["episodes"],
        "steps_per_episode": cfg["run"]["steps_per_episode"],
        "ref_speed": cfg["controller"]["ref_speed"],
        "perturbation_enabled": cfg["perturbation"]["enabled"],
        "perturbation_probability": cfg["perturbation"]["probability"] if cfg["perturbation"]["enabled"] else None,
        "dagger_enabled": cfg["dagger"]["enabled"],
        "num_transitions": int(obs_arr.shape[0]),
        "num_perturbed_steps": int(np.sum(perturbed_arr)),
        "obs_shape": list(obs_arr.shape),
        "expert_act_shape": list(expert_act_arr.shape),
        "executed_act_shape": list(executed_act_arr.shape),
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
    }

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


if __name__ == "__main__":
    main()
