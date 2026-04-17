import os
import numpy as np
import onnxruntime as ort
import joblib
import gym
from collections import deque

# ==========================================
# 1. CONFIGURATION & FILE PATHS
# ==========================================
ONNX_MODEL_PATH = "LSTM_2B_512D.onnx"
INPUT_SCALER_PATH = "LSTM_2B_512D_input_scaler.pkl"
TARGET_SCALER_PATH = "LSTM_2B_512D_target_scaler.pkl"

SEQ_LENGTH = 10
INPUT_DIM = 63

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def process_raw_observation(obs):
    """
    Extracts and formats data to match the model's INPUT_FEATURES:
    ['velocity', 'steering_angle', 'lidar_scans']
    """
    velocity = obs['linear_vels_x'][0]
    steering = obs['steering_angles'][0] if 'steering_angles' in obs else 0.0
    raw_lidar = obs['scans'][0]
    
    # Downsample LiDAR to exactly 61 points (1 vel + 1 steer + 61 lidar = 63 inputs)
    # Using linear spacing across the full scan array
    indices = np.linspace(0, len(raw_lidar) - 1, 61, dtype=int)
    downsampled_lidar = raw_lidar[indices]
    
    # Concatenate in the EXACT order defined in Devin's script
    feature_vector = np.concatenate([[velocity, steering], downsampled_lidar])
    
    return feature_vector

# ==========================================
# 3. MAIN SIMULATION LOOP
# ==========================================
def run_simulation():
    print("Loading Scalers...")
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    
    print("Initializing ONNX Runtime Session...")
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    
    print("Starting F1Tenth Gym Environment...")
    # Using the absolute path fix so it finds the levine map we downloaded!
    env = gym.make('f110_gym:f110-v0', map=os.path.abspath('levine'), map_ext='.png', num_agents=1)
    
    obs, step_reward, done, info = env.reset(np.array([[0.0, 0.0, 0.0]]))
    sequence_buffer = deque(maxlen=SEQ_LENGTH)
    
    # Pre-fill the sequence buffer
    initial_features = process_raw_observation(obs)
    scaled_initial = input_scaler.transform(initial_features.reshape(1, -1))[0]
    for _ in range(SEQ_LENGTH):
        sequence_buffer.append(scaled_initial)
        
    print("🟢 Simulating... Press Ctrl+C to stop.")
    
    try:
        while True:  # Run forever so we can watch it crash and restart!
            # 1. Format the sequence for ONNX
            model_input = np.array(sequence_buffer, dtype=np.float32).reshape(1, SEQ_LENGTH, INPUT_DIM)
            
            # 2. Run Inference
            onnx_pred = ort_session.run(None, {input_name: model_input})[0]
            
            # 3. Inverse Scale the Output
            physical_commands = target_scaler.inverse_transform(onnx_pred)[0]
            
            # TARGET_FEATURES = ['steering_angle', 'velocity']
            pred_steering = physical_commands[0]
            pred_velocity = physical_commands[1]
            
            # 4. Step the simulator
            obs, step_reward, done, info = env.step(np.array([[pred_steering, pred_velocity]]))
            env.render(mode='human')
            
            # 5. Handle crashes (Reset without closing the window)
            if done:
                print("💥 CRASH! Resetting...")
                obs, _, _, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))
                initial_features = process_raw_observation(obs)
                scaled_initial = input_scaler.transform(initial_features.reshape(1, -1))[0]
                for _ in range(SEQ_LENGTH):
                    sequence_buffer.append(scaled_initial)
                continue
            
            # 6. Process new observation for the next step
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