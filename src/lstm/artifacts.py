from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from LSTM_training.scripts.pytorch_to_onnx_converter import export_to_onnx


@dataclass(frozen=True)
class LSTMArtifactBundle:
    model_name: str
    config_path: Path
    onnx_path: Path
    input_scaler_path: Path
    target_scaler_path: Path
    model_config: dict


def _require_exists(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_lstm_artifacts(
    model_config_path: str | Path,
    onnx_model_path: str | Path | None = None,
    input_scaler_path: str | Path | None = None,
    target_scaler_path: str | Path | None = None,
    auto_export_if_missing: bool = True,
    export_device: str = "cpu",
    base_dir: str | Path | None = None,
) -> LSTMArtifactBundle:
    base_path = Path(base_dir).expanduser().resolve() if base_dir is not None else None

    def _resolve_candidate(candidate: str | Path) -> Path:
        path = Path(candidate).expanduser()
        if path.is_absolute():
            return path.resolve()
        if base_path is not None:
            return (base_path / path).resolve()
        return path.resolve()

    config_path = _require_exists(_resolve_candidate(model_config_path), "Model config")
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    model_name = config_path.stem.replace("_config", "")
    if onnx_model_path is not None:
        onnx_path = _resolve_candidate(onnx_model_path)
        scaler_in = _resolve_candidate(input_scaler_path) if input_scaler_path else None
        scaler_out = _resolve_candidate(target_scaler_path) if target_scaler_path else None
    else:
        onnx_dir = config_path.parent.parent / "onnx_models" / model_name
        onnx_path = onnx_dir / f"{model_name}.onnx"
        scaler_in = onnx_dir / f"{model_name}_input_scaler.pkl"
        scaler_out = onnx_dir / f"{model_name}_target_scaler.pkl"

    if auto_export_if_missing and (not onnx_path.is_file() or not scaler_in.is_file() or not scaler_out.is_file()):
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_onnx(str(config_path), str(onnx_path), device=export_device)

    return LSTMArtifactBundle(
        model_name=model_name,
        config_path=config_path,
        onnx_path=_require_exists(onnx_path, "ONNX model"),
        input_scaler_path=_require_exists(scaler_in, "Input scaler"),
        target_scaler_path=_require_exists(scaler_out, "Target scaler"),
        model_config=cfg.get("model_config", {}),
    )
