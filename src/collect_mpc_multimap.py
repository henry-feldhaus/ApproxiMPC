"""Orchestrate MPC data collection across maps and alternating directions from YAML."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

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
    episodes_per_combo: int,
    render: bool,
    combo_output_dir: Path,
) -> subprocess.CompletedProcess:
    cfg = copy.deepcopy(collector_cfg)
    cfg.setdefault("env", {})
    cfg.setdefault("run", {})
    cfg.setdefault("output", {})

    cfg["env"]["map"] = map_name
    cfg["env"]["track_direction"] = direction
    cfg["run"]["episodes"] = int(episodes_per_combo)
    cfg["run"]["render"] = bool(render)
    cfg["output"]["output_dir"] = str(combo_output_dir)
    cfg["output"]["output_filename"] = f"{cfg['controller']['mode']}_{map_name}_{direction}.npz"

    combo_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    env = dict(os.environ)
    env["APPROXIMPC_ACADOS_BUILD_TAG"] = f"combo_{combo_id}"

    return subprocess.run(
        [sys.executable, str(collector_script), "--config", str(combo_cfg_path)],
        check=False,
        env=env,
    )


def get_combo_quality_status(combo_cfg_path: Path, combo_output_dir: Path) -> tuple[str, dict | None]:
    """Return combo status from collector metadata if available.

    Status values:
      - "ok": run reached lap target with no collision/boundary termination.
      - "failed/collision": collision, boundary, or non-lap terminal outcome.
      - "failed": metadata missing/corrupt.
    """
    try:
        combo_cfg = load_yaml(combo_cfg_path)
        output_filename = combo_cfg.get("output", {}).get("output_filename")
        if not output_filename:
            return "failed", None

        meta_path = combo_output_dir / Path(str(output_filename)).with_suffix(".json")
        if not meta_path.exists():
            return "failed", None

        meta = json.loads(meta_path.read_text())
        end_reasons = meta.get("episode_end_reasons") or []
        end_reason = end_reasons[0] if end_reasons else None
        collisions = int(meta.get("num_collision_steps", 0))
        boundaries = int(meta.get("num_boundary_steps", 0))

        if end_reason == "lap_target_reached" and collisions == 0 and boundaries == 0:
            return "ok", meta
        return "failed: collision", meta
    except Exception:
        return "failed", None


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
    stop_on_error = bool(run_cfg.get("stop_on_error", True))

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
        "combos": [],
    }

    print(f"Run directory: {run_dir}")
    print(f"Total combos: {len(combos)}")
    if mode == "parallel":
        print(f"Execution mode: parallel with max_workers={max_workers}")
    else:
        print(f"Execution mode: sequential")

    # Prepare all combo tasks
    combo_tasks = []
    for idx, (map_name, direction) in enumerate(combos, start=1):
        combo_name = f"{map_name}_{direction}"
        combo_output_dir = run_dir / map_name / direction
        combo_output_dir.mkdir(parents=True, exist_ok=True)
        combo_cfg_path = run_dir / "configs" / f"collect_{idx:02d}_{combo_name}.yaml"
        combo_tasks.append((idx, map_name, direction, combo_cfg_path, combo_output_dir))

    # Execute combos (sequential or parallel)
    combo_results = {}
    if mode == "parallel":
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_combo,
                    collector_script=collector_script,
                    collector_cfg=base_collector_cfg,
                    combo_cfg_path=combo_cfg_path,
                    combo_id=idx,
                    map_name=map_name,
                    direction=direction,
                    episodes_per_combo=episodes_per_combo,
                    render=render,
                    combo_output_dir=combo_output_dir,
                ): (idx, map_name, direction, combo_cfg_path, combo_output_dir)
                for idx, map_name, direction, combo_cfg_path, combo_output_dir in combo_tasks
            }
            for future in as_completed(futures):
                idx, map_name, direction, combo_cfg_path, combo_output_dir = futures[future]
                try:
                    proc = future.result()
                    combo_results[idx] = proc
                    status = "ok" if proc.returncode == 0 else "failed"
                    print(f"[{idx}/{len(combos)}] {map_name}_{direction}: {status}")
                except Exception as e:
                    print(f"[{idx}/{len(combos)}] {map_name}_{direction}: exception: {e}")
                    combo_results[idx] = None
    else:
        # Sequential execution
        for idx, map_name, direction, combo_cfg_path, combo_output_dir in combo_tasks:
            print(
                f"[{idx}/{len(combos)}] map={map_name}, direction={direction}, "
                f"episodes={episodes_per_combo}, render={render}"
            )
            proc = run_combo(
                collector_script=collector_script,
                collector_cfg=base_collector_cfg,
                combo_cfg_path=combo_cfg_path,
                combo_id=idx,
                map_name=map_name,
                direction=direction,
                episodes_per_combo=episodes_per_combo,
                render=render,
                combo_output_dir=combo_output_dir,
            )
            combo_results[idx] = proc

    # Build manifest from results
    for idx, map_name, direction, combo_cfg_path, combo_output_dir in combo_tasks:
        proc = combo_results.get(idx)
        if proc is None:
            exit_code = -1
            status = "failed"
            end_reason = None
            collisions = None
            boundaries = None
        else:
            exit_code = int(proc.returncode)
            if proc.returncode == 0:
                status, meta = get_combo_quality_status(combo_cfg_path, combo_output_dir)
                end_reasons = (meta or {}).get("episode_end_reasons") or []
                end_reason = end_reasons[0] if end_reasons else None
                collisions = (meta or {}).get("num_collision_steps")
                boundaries = (meta or {}).get("num_boundary_steps")
            else:
                status = "failed"
                end_reason = None
                collisions = None
                boundaries = None

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
        }
        manifest["combos"].append(combo_status)

        if status == "failed" and stop_on_error and mode == "sequential":
            # Only stop on first error in sequential mode
            print("Stopping due to failure and run.stop_on_error=true")
            break

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    n_ok = sum(1 for c in manifest["combos"] if c["status"] == "ok")
    n_fail = sum(1 for c in manifest["combos"] if c["status"] != "ok")
    print(f"Finished multi-map run: ok={n_ok}, failed={n_fail}")
    print(f"Manifest: {manifest_path}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
