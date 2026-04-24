from __future__ import annotations

from collections import deque

import joblib
import numpy as np
import onnxruntime as ort

from .artifacts import LSTMArtifactBundle


class LSTMInferenceModel:
    def __init__(self, artifacts: LSTMArtifactBundle):
        self.artifacts = artifacts
        self.model_name = artifacts.model_name
        self.input_dim = int(artifacts.model_config.get("input_dim", 68))
        self.output_dim = int(artifacts.model_config.get("output_dim", 2))

        self.input_scaler = joblib.load(artifacts.input_scaler_path)
        self.target_scaler = joblib.load(artifacts.target_scaler_path)
        self.ort_session = ort.InferenceSession(str(artifacts.onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name
        self.seq_length = self._resolve_seq_length()

        self.sequence_buffer = deque(maxlen=self.seq_length)
        zero_step = np.zeros(self.input_dim, dtype=np.float32)
        for _ in range(self.seq_length):
            self.sequence_buffer.append(zero_step.copy())

    def _resolve_seq_length(self) -> int:
        configured = self.artifacts.model_config.get("seq_length")
        if configured is not None:
            return int(configured)

        input_meta = self.ort_session.get_inputs()[0]
        input_shape = input_meta.shape
        if len(input_shape) >= 2 and isinstance(input_shape[1], int):
            return int(input_shape[1])

        return 100

    def reset(self) -> None:
        self.sequence_buffer.clear()
        zero_step = np.zeros(self.input_dim, dtype=np.float32)
        for _ in range(self.seq_length):
            self.sequence_buffer.append(zero_step.copy())

    def predict(self, feature_step: np.ndarray) -> np.ndarray:
        feature_step = np.asarray(feature_step, dtype=np.float32)
        if feature_step.shape != (self.input_dim,):
            raise ValueError(f"Expected feature vector shape {(self.input_dim,)}, got {feature_step.shape}")

        self.sequence_buffer.append(feature_step)
        seq = np.stack(self.sequence_buffer, axis=0).astype(np.float32, copy=False)
        scaled = self.input_scaler.transform(seq).astype(np.float32, copy=False)
        model_input = scaled[None, :, :]
        output = self.ort_session.run(None, {self.input_name: model_input})[0]
        output = np.asarray(output, dtype=np.float32)
        if output.ndim == 3:
            output = output.reshape(output.shape[0], -1)
        pred = self.target_scaler.inverse_transform(output)
        if pred.shape[-1] != self.output_dim:
            raise ValueError(f"Expected output dim {self.output_dim}, got {pred.shape[-1]}")
        return pred[0].astype(np.float32, copy=False)
