"""Convert one episode from an MPC dataset NPZ into the shared comparison log schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from lstm.logger import LSTMRunLogger


def get_default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "mpc_compare_default.yaml"


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["__config_path__"] = str(config_path)
    cfg["__project_root__"] = str(Path(__file__).resolve().parent.parent)
    return cfg


def resolve_path(project_root: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_npz_dict(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def require_key(data: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Required key '{key}' not found in dataset")
    return data[key]


def select_episode_mask(data: dict[str, np.ndarray], episode_index: int) -> np.ndarray:
    episode_ids = require_key(data, "episode_ids")
    if episode_ids.ndim != 1:
        raise ValueError("episode_ids must be a 1D array")

    unique_ids = np.unique(episode_ids.astype(np.int64, copy=False))
    if unique_ids.size == 0:
        raise ValueError("Dataset contains no episodes")
    if episode_index < 0 or episode_index >= unique_ids.size:
        raise IndexError(
            f"episode_index={episode_index} is out of range for {unique_ids.size} available episodes"
        )

    selected_id = int(unique_ids[episode_index])
    return episode_ids == selected_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one episode from an MPC dataset NPZ into the shared comparison schema."
    )
    parser.add_argument("--config", type=str, default=str(get_default_config_path()), help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["__project_root__"])
    input_cfg = cfg["input"]

    dataset_path = resolve_path(project_root, input_cfg["dataset_path"])
    episode_index = int(input_cfg.get("episode_index", 0))
    timestep = float(cfg["env"]["timestep"])
    model_name = str(cfg["output"].get("model_name", "mpc"))

    data = load_npz_dict(dataset_path)
    episode_mask = select_episode_mask(data, episode_index)
    indices = np.flatnonzero(episode_mask)
    if indices.size == 0:
        raise ValueError(f"No rows found for episode_index={episode_index}")

    observations = require_key(data, "observations")[episode_mask]
    expert_actions = require_key(data, "expert_actions")[episode_mask]
    step_ids = require_key(data, "step_ids")[episode_mask]
    collision_flags = require_key(data, "collision_flags")[episode_mask]

    if "lidar_scans" in data:
        lidar = data["lidar_scans"][episode_mask]
    else:
        lidar = np.empty((indices.size, 0), dtype=np.float32)

    if observations.ndim != 2 or observations.shape[1] < 5:
        raise ValueError("observations must have shape (N, >=5)")
    if expert_actions.ndim != 2 or expert_actions.shape[1] < 2:
        raise ValueError("expert_actions must have shape (N, >=2)")
    if lidar.ndim != 2:
        raise ValueError("lidar array must have shape (N, M)")

    logger_cfg = {
        "__project_root__": cfg["__project_root__"],
        "lidar": {"n_bins": int(lidar.shape[1])},
        "logging": {
            "output_file": cfg["output"]["output_file"],
            **cfg["logging"],
        },
    }
    logger = LSTMRunLogger(logger_cfg, model_name)

    for row_idx in range(indices.size):
        obs_row = observations[row_idx]
        action_row = expert_actions[row_idx]
        step_id = int(step_ids[row_idx])

        logger.log_step(
            step_id=step_id,
            timestamp_s=(float(step_id) + 1.0) * timestep,
            pose_x=float(obs_row[0]),
            pose_y=float(obs_row[1]),
            yaw=float(obs_row[4]),
            velocity=float(obs_row[3]),
            steering_command=float(action_row[0]),
            speed_command=float(action_row[1]),
            lidar=np.asarray(lidar[row_idx], dtype=np.float32),
            collision=bool(collision_flags[row_idx]),
        )

    output_path = logger.save()
    print(f"Converted MPC episode {episode_index} from {dataset_path} to: {output_path}")


if __name__ == "__main__":
    main()
