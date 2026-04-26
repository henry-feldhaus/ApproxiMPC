import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_npz_to_dataframe(filepath):
    """loads the datasets, change the filepath here."""
    data = np.load(filepath)
    
    # Extract the arrays into a dataframe, now including the flags!
    df = pd.DataFrame({
        'timestamps': data['timestamps'],
        'x': data['x'],
        'y': data['y'],
        'velocity': data['velocity'],
        'steering_angle': data['steering_angle'],
        'collision_flag': data['collision_flag'],
        'completion_flag': data['completion_flag']
    })
    
    # Standardize time to start at 0 seconds
    df['elapsed_time'] = df['timestamps'] - df['timestamps'].iloc[0]
    
    # Calculate cumulative distance to align paths spatially
    dx = np.diff(df['x'], prepend=df['x'].iloc[0])
    dy = np.diff(df['y'], prepend=df['y'].iloc[0])
    df['step_distance'] = np.sqrt(dx**2 + dy**2)
    df['cumulative_distance'] = df['step_distance'].cumsum()
    
    return df

def calculate_lap_metrics(df_mpc, df_lstm):
    mpc_time = df_mpc['elapsed_time'].iloc[-1]
    lstm_time = df_lstm['elapsed_time'].iloc[-1]
    
    print("\n=== 1. LAP TIME DIFFERENCE ===")
    print(f"Baseline (MPC) Lap Time: {mpc_time:.3f} seconds")
    print(f"Model (LSTM) Lap Time:   {lstm_time:.3f} seconds")
    print(f"Difference:              {abs(mpc_time - lstm_time):.3f} seconds")

    print("\n=== 2. LAP COMPLETION STATUS ===")
    # Check MPC
    if df_mpc['completion_flag'].max() > 0:
        print("MPC Baseline: ✅ Completed the lap successfully.")
    elif df_mpc['collision_flag'].max() > 0:
        print("MPC Baseline: 💥 CRASHED into a wall.")
    else:
        print("MPC Baseline: ⚠️ Did not finish (Time ran out).")

    # Check LSTM
    if df_lstm['completion_flag'].max() > 0:
        print("LSTM Model:   ✅ Completed the lap successfully.")
    elif df_lstm['collision_flag'].max() > 0:
        print("LSTM Model:   💥 CRASHED into a wall.")
    else:
        print("LSTM Model:   ⚠️ Did not finish (Time ran out).")

def calculate_spatial_similarity(df_mpc, df_lstm):
    # Find the shortest distance traveled between the two so we don't extrapolate
    max_dist = min(df_mpc['cumulative_distance'].max(), df_lstm['cumulative_distance'].max())
    shared_distances = np.linspace(0, max_dist, num=1000)
    
    # Create interpolation functions for X and Y
    mpc_x_func = interp1d(df_mpc['cumulative_distance'], df_mpc['x'], kind='linear')
    mpc_y_func = interp1d(df_mpc['cumulative_distance'], df_mpc['y'], kind='linear')
    
    lstm_x_func = interp1d(df_lstm['cumulative_distance'], df_lstm['x'], kind='linear')
    lstm_y_func = interp1d(df_lstm['cumulative_distance'], df_lstm['y'], kind='linear')
    
    # Calculate the physical distance between the two lines at every point
    spatial_errors = np.sqrt((mpc_x_func(shared_distances) - lstm_x_func(shared_distances))**2 + 
                             (mpc_y_func(shared_distances) - lstm_y_func(shared_distances))**2)
    
    print("\n=== 3. PATH SIMILARITY ===")
    print(f"Average Path Deviation: {np.mean(spatial_errors):.3f} meters")
    print(f"Maximum Path Deviation: {np.max(spatial_errors):.3f} meters")

def plot_comparisons(df_mpc, df_lstm):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    axs[0].plot(df_mpc['x'], df_mpc['y'], label='Baseline (MPC)', color='blue', linewidth=2)
    axs[0].plot(df_lstm['x'], df_lstm['y'], label='Model (LSTM)', color='red', linestyle='--', linewidth=2)
    axs[0].set_title('Racing Line Comparison')
    axs[0].set_xlabel('X Position')
    axs[0].set_ylabel('Y Position')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(df_mpc['elapsed_time'], df_mpc['velocity'], label='MPC', color='blue')
    axs[1].plot(df_lstm['elapsed_time'], df_lstm['velocity'], label='LSTM', color='red', linestyle='--')
    axs[1].set_title('Velocity Profile')
    axs[1].set_xlabel('Elapsed Time (s)')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[2].plot(df_mpc['elapsed_time'], df_mpc['steering_angle'], label='MPC', color='blue')
    axs