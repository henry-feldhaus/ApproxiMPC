"""Minimal test to verify the gym environment works.
Uses Pure Pursuit controller for 100 steps.
No train imports, no RL dependencies.
Headless with live telemetry output.
"""

import gymnasium as gym
import numpy as np
from waypoint_follow import PurePursuitPlanner
from gymkhana.envs.gymkhana_env import GKEnv


def main():
    print("\n" + "="*60)
    print("  GymKhana Pure Pursuit Controller Demo")
    print("  Track: Spielberg | Model: Single-Track | Steps: 100")
    print("="*60 + "\n")
    
    env = gym.make(
        "gymkhana:gymkhana-v0",
        config={
            "map": "Spielberg",
            "num_agents": 1,
            "timestep": 0.01,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "st",
            "observation_config": {"type": "kinematic_state"},
        },
        render_mode=None,
    )
    
    print("Setting up Pure Pursuit controller...")
    track = env.unwrapped.track
    plans = PurePursuitPlanner(track=track, wb=0.17145 + 0.15875)
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"\n{'Step':<6} {'X (m)':<10} {'Y (m)':<10} {'θ (rad)':<12} {'Speed':<8} {'Steer':<8}")
    print("-" * 60)
    
    for step in range(100):
        x = obs["agent_0"]["pose_x"]
        y = obs["agent_0"]["pose_y"]
        theta = obs["agent_0"]["pose_theta"]
        
        speed, steer = plans.plan(
            x, y, theta,
            lookahead_distance=0.8,
            vgain=1.0,
        )
        action = np.array([[steer, speed]])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print telemetry every 10 steps
        if step % 10 == 0:
            print(f"{step:<6} {x:<10.3f} {y:<10.3f} {theta:<12.3f} {speed:<8.3f} {steer:<8.3f}")
        
        if terminated or truncated:
            print(f"\n✓ Episode completed at step {step}")
            break
    
    env.close()
    print("✓ GUI test complete - gym environment works!")


if __name__ == "__main__":
    main()
