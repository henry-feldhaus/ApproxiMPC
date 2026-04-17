"""
Example: Inference with exported ONNX LSTM model and scalers
"""
import onnxruntime as ort
import numpy as np
import joblib

# Paths to exported artifacts (per-model subdirectory)
onnx_path = "LSTM_training/models/4-16-25 Models/onnx_models/LSTM_1B_128D/LSTM_1B_128D.onnx"
input_scaler_path = "LSTM_training/models/4-16-25 Models/onnx_models/LSTM_1B_128D/LSTM_1B_128D_input_scaler.pkl"
target_scaler_path = "LSTM_training/models/4-16-25 Models/onnx_models/LSTM_1B_128D/LSTM_1B_128D_target_scaler.pkl"

# Load scalers
input_scaler = joblib.load(input_scaler_path)
target_scaler = joblib.load(target_scaler_path)

# Example input (batch_size=1, seq_len=10, input_dim=63)
raw_input = np.random.rand(1, 10, 63).astype(np.float32)

# Preprocess input
scaled_input = input_scaler.transform(raw_input.reshape(-1, raw_input.shape[-1])).reshape(raw_input.shape)

# Ensure correct dtype for ONNX runtime
scaled_input = scaled_input.astype(np.float32)

# Run ONNX inference (use the actual input name from the model)
sess = ort.InferenceSession(onnx_path)
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: scaled_input})[0]

# Postprocess output; ensure 2D for scaler
if output.ndim == 3:
	# collapse timestep dim if present: assume final-timestep prediction shape (batch, seq, dim)
	output_to_scale = output.reshape(output.shape[0], -1)
else:
	output_to_scale = output

pred = target_scaler.inverse_transform(output_to_scale)
print("Predicted action(s):", pred)
