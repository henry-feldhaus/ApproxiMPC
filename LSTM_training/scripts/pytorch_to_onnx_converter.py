"""
Export a trained LSTM model and its scalers to ONNX for F1Tenth simulation.
"""
import argparse
import os
import torch
import numpy as np
import joblib
from LSTM_training.scripts.load_model import load_model_bundle_from_config

def export_to_onnx(config_path, output_path, opset=17, device="cpu"):
    # Load model and scalers
    model, input_scaler, target_scaler, cfg = load_model_bundle_from_config(config_path, device=device)
    model.eval()

    # Save the scalers for use in simulation
    base = os.path.splitext(output_path)[0]
    joblib.dump(input_scaler, base + "_input_scaler.pkl")
    joblib.dump(target_scaler, base + "_target_scaler.pkl")

    # Prepare dummy input for export
    seq_len = cfg["model_config"].get("seq_length", 10)
    input_dim = cfg["model_config"].get("input_dim", 63)
    dummy_input = torch.zeros(1, seq_len, input_dim, dtype=torch.float32)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size"}},
    )
    print(f"Exported ONNX model to {output_path}")
    print(f"Saved input scaler to {base}_input_scaler.pkl")
    print(f"Saved target scaler to {base}_target_scaler.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LSTM model to ONNX with scalers.")
    parser.add_argument("--config", type=str, required=True, help="Path to model config JSON.")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX file path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda).")
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="If provided, will auto-populate --config and --output based on a model directory containing <name>_config.json and <name>.pt. Example: --model_dir 'LSTM_training/models/4-16-25 Models/LSTM_1B_128D_Pred_1'"
    )

    args = parser.parse_args()

    # If model_dir is provided, auto-populate config/output
    if args.model_dir:
        # Find config and model name
        files = os.listdir(args.model_dir)
        config_file = next((f for f in files if f.endswith('_config.json')), None)
        if not config_file:
            raise FileNotFoundError("No *_config.json found in model_dir")
        model_name = config_file.replace('_config.json', '')
        config_path = os.path.join(args.model_dir, config_file)
        output_path = os.path.join(os.getcwd(), f"{model_name}.onnx")
    else:
        config_path = args.config
        output_path = args.output

    export_to_onnx(config_path, output_path, args.opset, args.device)
