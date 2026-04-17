"""
Dummy F1Tenth Gym simulation loop using ONNX model and scalers.
This demonstrates how to load the ONNX model and scalers, preprocess observations, and run inference.
"""
import os
import numpy as np
import onnxruntime as ort
import joblib
import gym
from collections import deque

ONNX_MODEL_PATH = "LSTM_1B_128D.onnx"
INPUT_SCALER_PATH = "LSTM_1B_128D_input_scaler.pkl"
TARGET_SCALER_PATH = "LSTM_1B_128D_target_scaler.pkl"
SEQ_LENGTH = 10
INPUT_DIM = 63

def process_raw_observation(obs):
    """
    Extracts and formats data to match the model's INPUT_FEATURES:
    ['velocity', 'steering_angle', 'lidar_scans']
    """
    velocity = obs['linear_vels_x'][0]
    steering = obs['steering_angles'][0] if 'steering_angles' in obs else 0.0
    raw_lidar = obs['scans'][0]
    indices = np.linspace(0, len(raw_lidar) - 1, 61, dtype=int)
    downsampled_lidar = raw_lidar[indices]
    feature_vector = np.concatenate([[velocity, steering], downsampled_lidar])
    return feature_vector

def run_simulation():
    print("Loading Scalers...")
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    print("Initializing ONNX Runtime Session...")
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    print("Starting F1Tenth Gym Environment...")
    env = gym.make('f110_gym:f110-v0', map=os.path.abspath('levine'), map_ext='.png', num_agents=1)
    obs, _, _, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
    sequence_buffer = deque(maxlen=SEQ_LENGTH)
    initial_features = process_raw_observation(obs)
    scaled_initial = input_scaler.transform(initial_features.reshape(1, -1))[0]
    for _ in range(SEQ_LENGTH):
        sequence_buffer.append(scaled_initial)
    print("🟢 Simulating... Press Ctrl+C to stop.")
    try:
        while True:
            model_input = np.array(sequence_buffer, dtype=np.float32).reshape(1, SEQ_LENGTH, INPUT_DIM)
            onnx_pred = ort_session.run(None, {input_name: model_input})[0]
            physical_commands = target_scaler.inverse_transform(onnx_pred)[0]
            pred_steering = physical_commands[0]
            pred_velocity = physical_commands[1]
            obs, _, done, _ = env.step(np.array([[pred_steering, pred_velocity]]))
            env.render(mode='human')
            if done:
                print("💥 CRASH! Resetting...")
                obs, _, _, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
                initial_features = process_raw_observation(obs)
                scaled_initial = input_scaler.transform(initial_features.reshape(1, -1))[0]
                for _ in range(SEQ_LENGTH):
                    sequence_buffer.append(scaled_initial)
                continue
            new_features = process_raw_observation(obs)
            scaled_new_features = input_scaler.transform(new_features.reshape(1, -1))[0]
            sequence_buffer.append(scaled_new_features)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == '__main__':
    run_simulation()
