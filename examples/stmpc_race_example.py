"""Runner script for the Single Track MPC controller on the F1TENTH gym (race mode)."""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

import gymkhana  # noqa: F401  # ensures gym env registration
from gymkhana.envs import GKEnv

# Avoid importing through controllers package __init__, which currently pulls stale RL modules.
sys.path.insert(0, str(Path(__file__).resolve().parent / "controllers"))
from mpc.gym_bridge import STMPCGymBridge

from examples.examples_utils import display_frenet_dynamic_state_obs

REF_SPEED = 4.0
MAX_STEPS = 2000


def get_stmpc_race_config() -> dict:
    return {
        "map": "Spielberg",
        "num_agents": 1,
        "timestep": 0.01,
        "integrator": "rk4",
        "model": "std",
        "control_input": ["speed", "steering_angle"],
        "observation_config": {"type": "frenet_dynamic_state"},
        "normalize_act": False,
        "normalize_obs": False,
        "training_mode": "race",
        "track_direction": "normal",
        "max_episode_steps": MAX_STEPS,
        "params": GKEnv.f1tenth_std_vehicle_params(),
    }


def main():
    config = get_stmpc_race_config()

    env = gym.make(
        "gymkhana:gymkhana-v0",
        config=config,
        render_mode="human",
    )

    bridge = STMPCGymBridge(env, ref_speed=REF_SPEED)

    # Build initial state
    # State format: [x, y, delta, v, yaw, yaw_rate, slip_angle]
    track = env.unwrapped.track
    x, y, yaw = track.frenet_to_cartesian(0, ey=0, ephi=0)
    init_states = np.array([[x, y, 0.0, REF_SPEED, yaw, 0.0, 0.0]])

    obs, info = env.reset(options={"states": init_states})
    bridge.init_from_obs(obs)

    env.render()

    for step in range(MAX_STEPS):
        action = bridge.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        display_frenet_dynamic_state_obs(step, obs, reward)
        env.render()

        if done or truncated:
            print(f"Episode ended at step {step} (terminated={done}, truncated={truncated})")
            obs, info = env.reset(options={"states": init_states})
            bridge.init_from_obs(obs)
            break

    print("Done")
    env.close()


if __name__ == "__main__":
    main()
