"""Runner script for the Kinematic MPC controller on the F1TENTH gym."""

import gymnasium as gym
import numpy as np

import gymkhana  # noqa: F401  # ensures gym env registration
from mpc.gym_bridge import KMPCGymBridge

from examples_utils import display_kinematic_state_obs

REF_SPEED = 4.0
MAX_STEPS = 2000


def get_kmpc_race_config() -> dict:
    return {
        "map": "Spielberg",
        "num_agents": 1,
        "timestep": 0.01,
        "integrator": "rk4",
        "model": "ks",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "kinematic_state"},
        "normalize_act": False,
        "normalize_obs": False,
        "training_mode": "race",
        "track_direction": "normal",
        "max_episode_steps": MAX_STEPS,
    }


def main():
    config = get_kmpc_race_config()

    env = gym.make(
        "gymkhana:gymkhana-v0",
        config=config,
        render_mode="human",
    )

    bridge = KMPCGymBridge(env, ref_speed=REF_SPEED)

    x0, y0, yaw0 = bridge.get_start_pose()
    obs, info = env.reset(options={"poses": np.array([[x0, y0, yaw0]])})

    step = 0
    total_reward = 0.0
    env.render()

    while step < MAX_STEPS:
        action = bridge.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        display_kinematic_state_obs(step, obs, reward, total_reward)
        step += 1
        env.render()

        if terminated or truncated:
            print(f"Episode ended at step {step} (terminated={terminated}, truncated={truncated})")
            break

    print(f"Done. steps={step}, total_reward={total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
