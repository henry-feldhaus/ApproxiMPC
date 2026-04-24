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
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(output_path)[0]
    joblib.dump(input_scaler, base + "_input_scaler.pkl")
    joblib.dump(target_scaler, base + "_target_scaler.pkl")

    # Prepare dummy input for export
    # The trained artifact config does not currently persist seq_length, but the
    # training pipeline and deployed runtime both use 100 as the intended
    # baseline history length.
    seq_len = cfg["model_config"].get("seq_length", 100)
    input_dim = cfg["model_config"].get("input_dim", 63)
    dummy_input = torch.zeros(1, seq_len, input_dim, dtype=torch.float32)

    # Export to ONNX (only batch_size is dynamic, seq_len is fixed)
    export_kwargs = dict(
        model=model,
        args=(dummy_input,),
        f=output_path,
        export_params=True,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Some torch versions accept `use_external_data_format`; others do not.
    # Try with it first, then fall back to calling without it.
    try:
        torch.onnx.export(**{**export_kwargs, "use_external_data_format": False})
    except TypeError:
        # older torch versions may raise TypeError for unknown kwargs
        torch.onnx.export(**export_kwargs)

    # Post-process: attempt to re-embed any external data into the single .onnx file
    try:
        import onnx
        import glob

        # Load model including external data, then save back to single-file .onnx
        mdl = onnx.load_model(output_path, load_external_data=True)
        onnx.save_model(mdl, output_path)

        # Remove any external data shards created by the exporter
        for shard in glob.glob(output_path + ".data*"):
            try:
                os.remove(shard)
            except OSError:
                pass
    except Exception as e:
        print(f"Warning: could not re-embed external data for {output_path}: {e}")

    print(f"Exported ONNX model to {output_path}")
    print(f"Saved input scaler to {base}_input_scaler.pkl")
    print(f"Saved target scaler to {base}_target_scaler.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LSTM model to ONNX with scalers.")
    parser.add_argument("--config", type=str, required=False, help="Path to model config JSON.")
    parser.add_argument("--output", type=str, required=False, help="Output ONNX file path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda).")
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="If provided, will auto-populate --config and --output based on a model directory containing <name>_config.json and <name>.pt. Example: --model_dir 'LSTM_training/models/4-16-25 Models/LSTM_1B_128D_Pred_1'"
    )

    args = parser.parse_args()

    # If model_dir is provided, auto-populate config/output
    if args.model_dir:
        # Recursively search for *_config.json in model_dir and subdirs
        config_path = None
        model_name = None
        for root, dirs, files in os.walk(args.model_dir):
            for f in files:
                if f.endswith('_config.json'):
                    config_path = os.path.join(root, f)
                    model_name = f.replace('_config.json', '')
                    break
            if config_path:
                break
        if not config_path:
            raise FileNotFoundError(f"No *_config.json found in model_dir or its subdirectories: {args.model_dir}")
        onnx_dir = os.path.join(os.path.dirname(args.model_dir), "onnx_models")
        os.makedirs(onnx_dir, exist_ok=True)
        # Create a subdirectory per model to keep artifacts organized
        output_dir = os.path.join(onnx_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_name}.onnx")
    else:
        if not args.config or not args.output:
            parser.error("--config and --output are required if --model_dir is not provided.")
        config_path = args.config
        output_path = args.output

    export_to_onnx(config_path, output_path, args.opset, args.device)
