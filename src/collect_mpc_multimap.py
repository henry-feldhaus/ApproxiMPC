"""Orchestrate MPC data collection across maps and alternating directions from YAML."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def get_default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "collect_mpc_multimap_fullscale.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-map MPC collection from YAML")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to multi-map YAML config (default: {get_default_config_path()})",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_list_of_strings(name: str, values: object) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    normalized = [str(v).strip() for v in values]
    if any(not v for v in normalized):
        raise ValueError(f"{name} contains empty entries")
    return normalized


def build_combo_sequence(maps: list[str], directions: list[str]) -> list[tuple[str, str]]:
    combos: list[tuple[str, str]] = []
    for idx, map_name in enumerate(maps):
        ordered_dirs = directions if idx % 2 == 0 else list(reversed(directions))
        for direction in ordered_dirs:
            combos.append((map_name, direction))
    return combos


def build_run_dir(cfg: dict, repo_root: Path) -> Path:
    output_cfg = cfg.get("output", {})
    root_dir = Path(str(output_cfg.get("root_dir", "outputs/datasets/multimap")))
    if not root_dir.is_absolute():
        root_dir = repo_root / root_dir

    run_name = output_cfg.get("run_name")
    if run_name is None or str(run_name).strip().lower() in {"", "null", "none"}:
        run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_name = str(run_name).strip()

    run_dir = root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    return run_dir


def run_combo(
    collector_script: Path,
    collector_cfg: dict,
    combo_cfg_path: Path,
    combo_id: int,
    map_name: str,
    direction: str,
    episodes_to_run: int,
    render: bool,
    video_cfg: dict | None,
    combo_output_dir: Path,
    output_filename: str | None = None,
) -> subprocess.CompletedProcess:
    cfg = copy.deepcopy(collector_cfg)
    cfg.setdefault("env", {})
    cfg.setdefault("run", {})
    cfg.setdefault("output", {})

    cfg["env"]["map"] = map_name
    cfg["env"]["track_direction"] = direction
    cfg["run"]["episodes"] = int(episodes_to_run)
    cfg["run"]["render"] = bool(render)
    if video_cfg is not None:
        cfg["run"]["video"] = copy.deepcopy(video_cfg)
    cfg["output"]["output_dir"] = str(combo_output_dir)
    if output_filename is None:
        output_filename = f"{cfg['controller']['mode']}_{map_name}_{direction}.npz"
    cfg["output"]["output_filename"] = output_filename

    combo_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    env = dict(os.environ)
    env["APPROXIMPC_ACADOS_BUILD_TAG"] = f"combo_{combo_id}"

    return subprocess.run(
        [sys.executable, str(collector_script), "--config", str(combo_cfg_path)],
        check=False,
        env=env,
    )


def build_output_filename(controller_mode: str, map_name: str, direction: str) -> str:
    return f"{controller_mode}_{map_name}_{direction}.npz"


def get_combo_output_paths(combo_output_dir: Path, output_filename: str) -> tuple[Path, Path]:
    npz_path = combo_output_dir / Path(output_filename)
    return npz_path, npz_path.with_suffix(".json")


def load_npz_dict(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def count_canonical_episodes(canonical_npz: Path) -> int:
    if not canonical_npz.exists():
        return 0
    data = load_npz_dict(canonical_npz)
    if "episode_ids" not in data:
        return 0
    return int(len(np.unique(data["episode_ids"])))


def is_successful_attempt(meta: dict) -> bool:
    end_reasons = meta.get("episode_end_reasons") or []
    if len(end_reasons) != 1 or end_reasons[0] != "lap_target_reached":
        return False
    collisions = int(meta.get("num_collision_steps", 0))
    boundaries = int(meta.get("num_boundary_steps", 0))
    return collisions == 0 and boundaries == 0


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def append_attempt_to_canonical(
    canonical_npz: Path,
    canonical_meta_path: Path,
    attempt_npz: Path,
    attempt_meta_path: Path,
) -> None:
    attempt_data = load_npz_dict(attempt_npz)
    attempt_meta = json.loads(attempt_meta_path.read_text())

    if not canonical_npz.exists() or not canonical_meta_path.exists():
        attempt_npz.replace(canonical_npz)
        attempt_meta_path.replace(canonical_meta_path)
        return

    canonical_data = load_npz_dict(canonical_npz)
    canonical_meta = json.loads(canonical_meta_path.read_text())

    old_ids = canonical_data.get("episode_ids", np.asarray([], dtype=np.int32))
    new_ids = attempt_data.get("episode_ids", np.asarray([], dtype=np.int32))
    if old_ids.size > 0 and new_ids.size > 0:
        attempt_data["episode_ids"] = new_ids + int(np.max(old_ids) + 1)

    n_old = int(old_ids.shape[0])
    n_new = int(new_ids.shape[0])
    merged: dict[str, np.ndarray] = {}
    for key in set(canonical_data.keys()) | set(attempt_data.keys()):
        if key not in canonical_data:
            merged[key] = attempt_data[key]
            continue
        if key not in attempt_data:
            merged[key] = canonical_data[key]
            continue

        old_arr = canonical_data[key]
        new_arr = attempt_data[key]
        if (
            isinstance(old_arr, np.ndarray)
            and isinstance(new_arr, np.ndarray)
            and old_arr.ndim > 0
            and new_arr.ndim > 0
            and old_arr.shape[0] == n_old
            and new_arr.shape[0] == n_new
        ):
            merged[key] = np.concatenate([old_arr, new_arr], axis=0)
        else:
            merged[key] = canonical_data[key]

    np.savez_compressed(canonical_npz, **merged)

    merged_meta = dict(canonical_meta)
    merged_reasons = (canonical_meta.get("episode_end_reasons") or []) + (attempt_meta.get("episode_end_reasons") or [])
    merged_meta["episode_end_reasons"] = merged_reasons
    merged_meta["episode_end_reason_counts"] = dict(Counter(merged_reasons))
    if "observations" in merged:
        merged_meta["obs_shape"] = list(merged["observations"].shape)
        merged_meta["num_transitions"] = int(merged["observations"].shape[0])
    if "expert_actions" in merged:
        merged_meta["expert_act_shape"] = list(merged["expert_actions"].shape)
    if "executed_actions" in merged:
        merged_meta["executed_act_shape"] = list(merged["executed_actions"].shape)
    if "is_perturbed" in merged:
        merged_meta["num_perturbed_steps"] = int(np.sum(merged["is_perturbed"]))
    if "collision_flags" in merged:
        merged_meta["num_collision_steps"] = int(np.sum(merged["collision_flags"]))
    if "boundary_flags" in merged:
        merged_meta["num_boundary_steps"] = int(np.sum(merged["boundary_flags"]))

    merged_meta["created_utc"] = datetime.now(timezone.utc).isoformat()
    canonical_meta_path.write_text(json.dumps(merged_meta, indent=2))

    remove_if_exists(attempt_npz)
    remove_if_exists(attempt_meta_path)


def read_combo_meta(combo_output_dir: Path, output_filename: str) -> dict | None:
    try:
        _, meta_path = get_combo_output_paths(combo_output_dir, output_filename)
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config) if args.config else get_default_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Multi-map config not found: {cfg_path}")

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_yaml(cfg_path)

    base_collector_cfg_path = Path(str(cfg["base_collector_config"]))
    if not base_collector_cfg_path.is_absolute():
        base_collector_cfg_path = repo_root / base_collector_cfg_path
    if not base_collector_cfg_path.exists():
        raise FileNotFoundError(f"Base collector config not found: {base_collector_cfg_path}")

    collector_script = Path(__file__).resolve().parent / "collect_mpc_data.py"
    if not collector_script.exists():
        raise FileNotFoundError(f"Collector script not found: {collector_script}")

    base_collector_cfg = load_yaml(base_collector_cfg_path)

    # Apply collector-level overrides from multimap config
    overrides = cfg.get("collector_overrides", {})
    if overrides:
        def deep_merge(base: dict, updates: dict) -> dict:
            """Recursively merge updates into base, leaving base untouched."""
            result = copy.deepcopy(base)
            for key, value in updates.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
            return result
        base_collector_cfg = deep_merge(base_collector_cfg, overrides)

    run_cfg = cfg.get("run", {})
    maps = ensure_list_of_strings("run.maps", run_cfg.get("maps", []))
    directions = ensure_list_of_strings("run.directions", run_cfg.get("directions", []))
    episodes_per_combo = int(run_cfg.get("episodes_per_combo", 1))
    render = bool(run_cfg.get("render", False))
    video_cfg = copy.deepcopy(run_cfg.get("video")) if "video" in run_cfg else None
    stop_on_error = bool(run_cfg.get("stop_on_error", True))
    retry_until_success = bool(run_cfg.get("retry_until_success", True))
    max_attempts_per_combo = int(run_cfg.get("max_attempts_per_combo", episodes_per_combo * 3))

    mode = str(cfg.get("execution", {}).get("mode", "sequential")).lower()
    max_workers = None
    if mode == "parallel":
        max_workers = int(cfg.get("execution", {}).get("max_workers", 2))
        if max_workers < 1:
            raise ValueError(f"execution.max_workers must be >= 1, got {max_workers}")

    combos = build_combo_sequence(maps, directions)
    run_dir = build_run_dir(cfg, repo_root)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "multi_map_config": str(cfg_path),
        "base_collector_config": str(base_collector_cfg_path),
        "execution_mode_requested": mode,
        "execution_mode_used": mode if mode in {"sequential", "parallel"} else "sequential",
        "max_workers": max_workers,
        "maps": maps,
        "directions": directions,
        "episodes_per_combo": episodes_per_combo,
        "render": render,
        "video": video_cfg,
        "retry_until_success": retry_until_success,
        "max_attempts_per_combo": max_attempts_per_combo,
        "combos": [],
    }

    print(f"Run directory: {run_dir}")
    print(f"Total combos: {len(combos)}")
    if mode == "parallel":
        print(f"Execution mode: parallel with max_workers={max_workers}")
    else:
        print(f"Execution mode: sequential")
    if retry_until_success:
        print(f"Retry until success: enabled (max {max_attempts_per_combo} attempts per combo)")

    # Prepare all combo tasks
    combo_tasks = []
    for idx, (map_name, direction) in enumerate(combos, start=1):
        combo_name = f"{map_name}_{direction}"
        combo_output_dir = run_dir / map_name / direction
        combo_output_dir.mkdir(parents=True, exist_ok=True)
        combo_cfg_path = run_dir / "configs" / f"collect_{idx:02d}_{combo_name}.yaml"
        combo_tasks.append((idx, map_name, direction, combo_cfg_path, combo_output_dir))

    # Track successful episodes per combo
    combo_success_counts = {idx: 0 for idx, _, _, _, _ in combo_tasks}
    combo_attempt_counts = {idx: 0 for idx, _, _, _, _ in combo_tasks}
    combo_final_results = {}

    def run_single_combo_with_retries(
        idx: int, map_name: str, direction: str, combo_cfg_path: Path, combo_output_dir: Path
    ) -> dict:
        """Run one combo, appending only successful single-episode attempts."""
        canonical_filename = build_output_filename(base_collector_cfg["controller"]["mode"], map_name, direction)
        canonical_npz, canonical_meta = get_combo_output_paths(combo_output_dir, canonical_filename)
        canonical_stem = canonical_npz.stem

        for stale in combo_output_dir.glob(f"{canonical_stem}.attempt_*.npz"):
            stale.unlink()
        for stale in combo_output_dir.glob(f"{canonical_stem}.attempt_*.json"):
            stale.unlink()

        if not retry_until_success:
            proc = run_combo(
                collector_script=collector_script,
                collector_cfg=base_collector_cfg,
                combo_cfg_path=combo_cfg_path,
                combo_id=idx,
                map_name=map_name,
                direction=direction,
                episodes_to_run=episodes_per_combo,
                render=render,
                video_cfg=video_cfg,
                combo_output_dir=combo_output_dir,
                output_filename=canonical_filename,
            )
            return {
                "exit_code": int(proc.returncode),
                "successful_episodes": count_canonical_episodes(canonical_npz),
                "attempts": 1,
                "proc": proc,
            }

        current_success = count_canonical_episodes(canonical_npz)
        attempt = 0
        while current_success < episodes_per_combo and attempt < max_attempts_per_combo:
            attempt += 1
            combo_attempt_counts[idx] = attempt

            print(
                f"[{idx}/{len(combos)}] {map_name}_{direction}: attempt {attempt}/{max_attempts_per_combo}, "
                f"need {episodes_per_combo} successful episodes (have {current_success})"
            )

            attempt_filename = f"{canonical_stem}.attempt_{attempt:03d}.npz"
            attempt_npz = combo_output_dir / attempt_filename
            attempt_meta = attempt_npz.with_suffix(".json")

            proc = run_combo(
                collector_script=collector_script,
                collector_cfg=base_collector_cfg,
                combo_cfg_path=combo_cfg_path,
                combo_id=idx,
                map_name=map_name,
                direction=direction,
                episodes_to_run=1,
                render=render,
                video_cfg=video_cfg,
                combo_output_dir=combo_output_dir,
                output_filename=attempt_filename,
            )

            if proc.returncode != 0:
                print(f"  → collector failed with exit code {proc.returncode}")
                remove_if_exists(attempt_npz)
                remove_if_exists(attempt_meta)
                continue

            if not attempt_meta.exists() or not attempt_npz.exists():
                print("  → attempt output missing; discarded")
                continue

            attempt_meta_obj = json.loads(attempt_meta.read_text())
            if is_successful_attempt(attempt_meta_obj):
                append_attempt_to_canonical(canonical_npz, canonical_meta, attempt_npz, attempt_meta)
                current_success = count_canonical_episodes(canonical_npz)
                combo_success_counts[idx] = current_success
                print(f"  → accepted successful episode (total accumulated: {current_success})")
            else:
                print("  → rejected failed/collision episode")
                remove_if_exists(attempt_npz)
                remove_if_exists(attempt_meta)

        if current_success >= episodes_per_combo:
            return {
                "exit_code": 0,
                "successful_episodes": current_success,
                "attempts": attempt,
                "proc": None,
            }

        print(
            f"  ⚠ Max attempts ({max_attempts_per_combo}) reached with only {current_success} "
            f"successful episodes (needed {episodes_per_combo})"
        )
        return {
            "exit_code": -1,
            "successful_episodes": current_success,
            "attempts": attempt,
            "proc": None,
        }

    # Execute combos (sequential or parallel)
    if mode == "parallel":
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_combo_with_retries,
                    idx=idx,
                    map_name=map_name,
                    direction=direction,
                    combo_cfg_path=combo_cfg_path,
                    combo_output_dir=combo_output_dir,
                ): (idx, map_name, direction)
                for idx, map_name, direction, combo_cfg_path, combo_output_dir in combo_tasks
            }
            for future in as_completed(futures):
                idx, map_name, direction = futures[future]
                try:
                    result = future.result()
                    combo_final_results[idx] = result
                    status = "ok" if result["exit_code"] == 0 else "insufficient"
                    print(f"[{idx}] {map_name}_{direction}: {status} ({result['successful_episodes']} successful in {result['attempts']} attempt(s))")
                except Exception as e:
                    print(f"[{idx}] {map_name}_{direction}: exception: {e}")
                    combo_final_results[idx] = {"exit_code": -1, "successful_episodes": 0, "attempts": 0, "proc": None}
    else:
        # Sequential execution with retries
        for idx, map_name, direction, combo_cfg_path, combo_output_dir in combo_tasks:
            result = run_single_combo_with_retries(idx, map_name, direction, combo_cfg_path, combo_output_dir)
            combo_final_results[idx] = result
            if result["exit_code"] != 0 and stop_on_error and retry_until_success:
                print("Stopping due to insufficient successful episodes and run.stop_on_error=true")
                break


    # Build manifest from results
    for idx, map_name, direction, combo_cfg_path, combo_output_dir in combo_tasks:
        result = combo_final_results.get(idx, {"exit_code": -1, "successful_episodes": 0, "attempts": 0, "proc": None})
        exit_code = result.get("exit_code", -1)
        successful_episodes = result.get("successful_episodes", 0)
        attempts = result.get("attempts", 0)

        output_filename = build_output_filename(base_collector_cfg["controller"]["mode"], map_name, direction)
        meta = read_combo_meta(combo_output_dir, output_filename)
        end_reasons = (meta or {}).get("episode_end_reasons") or []
        end_reason = end_reasons[-1] if end_reasons else None
        collisions = (meta or {}).get("num_collision_steps")
        boundaries = (meta or {}).get("num_boundary_steps")

        if (
            exit_code == 0
            and successful_episodes == episodes_per_combo
            and int(collisions or 0) == 0
            and int(boundaries or 0) == 0
        ):
            status = "ok"
        elif retry_until_success:
            status = "insufficient"
        else:
            status = "failed"

        combo_status = {
            "index": idx,
            "map": map_name,
            "direction": direction,
            "collector_config": str(combo_cfg_path),
            "output_dir": str(combo_output_dir),
            "exit_code": exit_code,
            "status": status,
            "episode_end_reason": end_reason,
            "num_collision_steps": collisions,
            "num_boundary_steps": boundaries,
            "successful_episodes": successful_episodes,
            "target_episodes": episodes_per_combo,
            "attempts": attempts,
        }
        manifest["combos"].append(combo_status)

        if exit_code != 0 and stop_on_error and mode == "sequential":
            print("Stopping due to failure and run.stop_on_error=true")
            break

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    n_ok = sum(1 for c in manifest["combos"] if c["status"] == "ok")
    n_insufficient = sum(1 for c in manifest["combos"] if c["status"] == "insufficient")
    n_fail = sum(1 for c in manifest["combos"] if c["status"] not in ("ok", "insufficient"))
    print(
        f"Finished multi-map run: ok={n_ok}, insufficient={n_insufficient}, failed={n_fail} "
        f"({sum(c['successful_episodes'] for c in manifest['combos'])}/{sum(c['target_episodes'] for c in manifest['combos'])} successful episodes)"
    )
    print(f"Manifest: {manifest_path}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
