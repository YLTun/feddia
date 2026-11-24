#!/usr/bin/env python3
# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Play/visualize trained navigation policies.

Supports two model formats:
- pt: PyTorch state dict (actor +/- critic) exported by federated aggregation
- zip: SB3 TD3 checkpoint (.zip), e.g. a client's local model

Usage (inside container):
    # Play a federated global model (e.g. FedDIA)
    /opt/IsaacLab/isaaclab.sh -p scripts/play.py \
        --model_path logs/feddia_m/<run_dir>/global_agent_states/latest_agent.pt \
        --obstacle_mode dense

    # Play an isolated (local-only) client model
    /opt/IsaacLab/isaaclab.sh -p scripts/play.py \
        --model_path logs/isolated/<run_dir>/clients/client_0/latest_model.zip
"""

from __future__ import annotations

import argparse
import os
import sys

# --- Parse arguments before importing Isaac Lab ---
parser = argparse.ArgumentParser(description="Play/visualize trained navigation agents.")
parser.add_argument(
    "--headless", action="store_true", help="Run in headless mode (no GUI)."
)
parser.add_argument(
    "--num_envs", type=int, default=4, help="Number of parallel environments."
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to trained model (.zip SB3 checkpoint or .pt FRL agent state).",
)
parser.add_argument(
    "--model_format",
    type=str,
    default="auto",
    choices=["auto", "zip", "pt"],
    help="Model format. 'auto' infers from file extension.",
)
parser.add_argument(
    "--debug_env",
    type=int,
    default=-1,
    help="Environment index to print debug info for (e.g., 0). -1 to disable.",
)
parser.add_argument(
    "--obstacle_mode", type=str, default="random",
    choices=["none", "random", "dense", "random_dense", "gate_palisade_multi", "factory", "living"],
    help="Obstacle distribution mode."
)
parser.add_argument(
    "--lidar_vis",
    action="store_true",
    help="Draw the LiDAR ray hits in the viewport. Shows what the sensor "
    "actually returns, which is the only way to see a collider the robot is "
    "blocked by but cannot sense.",
)
args = parser.parse_args()

# --- Launch Isaac Sim app ---
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# --- Now import Isaac Lab and env ---
import gymnasium as gym
import torch
import numpy as np

# Register our environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: F401 — triggers gym.register

from envs.navigation_env import NavigationEnvCfg
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from frl_utils import load_agent_state


def infer_model_format(path: str, explicit_format: str) -> str:
    """Infer model format from file extension or use explicit format."""
    if explicit_format != "auto":
        return explicit_format
    _, extension = os.path.splitext(path)
    extension = extension.lower()
    if extension == ".zip":
        return "zip"
    if extension == ".pt":
        return "pt"
    raise ValueError(f"Unable to infer model format from extension: {path}")


def build_agent_model(env) -> TD3:
    """Build a fresh model scaffold for agent state restoration."""
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.3 * np.ones(n_actions)
    )
    return TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=1000,
        batch_size=1024,
        tau=0.005,
        gamma=0.995,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def load_model(env, model_path: str, model_format: str) -> TD3:
    """Load model from either .zip (SB3) or .pt (FRL agent state)."""
    if model_format == "zip":
        print(f"[INFO] Loading SB3 model from: {model_path}")
        return TD3.load(model_path, env=env)

    # Format is .pt: build scaffold and load agent state
    print(f"[INFO] Building model scaffold for agent state: {model_path}")
    model = build_agent_model(env)
    agent_state = torch.load(model_path, map_location="cpu")
    load_agent_state(model, agent_state)
    print(f"[INFO] Loaded agent state successfully.")
    return model


def main():
    """Play/visualize the trained navigation agent."""
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file not found at: {args.model_path}")
        sys.exit(1)

    # --- Infer model format ---
    model_format = infer_model_format(args.model_path, args.model_format)
    print(f"[INFO] Model format: {model_format}")

    # --- Create environment ---
    # Hand-authored scenes (factory, living) are separate env classes selected
    # by gym id rather than by obstacle_mode. Their episode_length_s is left
    # alone rather than forced to 40 s: it derives from ARENA_SIZE, so
    # overriding it would decouple this view from what the evaluation ran.
    # (Both are 40 s today, since both scenes use the 5 m arena.)
    if args.obstacle_mode in envs.SCENE_MODES:
        gym_id, cfg_cls = envs.SCENE_MODES[args.obstacle_mode]
        env_cfg = cfg_cls()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.lidar.debug_vis = args.lidar_vis
        env = gym.make(gym_id, cfg=env_cfg)
    else:
        env_cfg = NavigationEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.obstacle_mode = args.obstacle_mode
        env_cfg.episode_length_s = 40.0
        env_cfg.lidar.debug_vis = args.lidar_vis
        env = gym.make("Isaac-Navigation-v0", cfg=env_cfg)

    # --- Wrap for SB3 ---
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    env = Sb3VecEnvWrapper(env)

    # --- Load Model ---
    model = load_model(env, args.model_path, model_format)

    # --- Evaluation Loop ---
    print("[INFO] Starting play loop...")
    obs = env.reset()
    step_count = 0
    try:
        while simulation_app.is_running():
            # Get action from the model (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)

            # Step the environment
            obs, rewards, dones, infos = env.step(action)
            
            # --- Debug Logging ---
            if args.debug_env >= 0 and args.debug_env < args.num_envs and step_count % 50 == 0:
                idx = args.debug_env
                print(f"\n--- Step {step_count} | Env {idx} ---")
                print(f"Action (Network Output raw): v_norm={action[idx][0]:.3f}, w_norm={action[idx][1]:.3f}")
                
                # Parse 124-dim observation: [lidar(120), dist(1), heading(1), vel_fwd(1), vel_ang(1)]
                obs_env = obs[idx]
                if len(obs_env) == 124:
                    lidar = obs_env[:120]
                    dist_n = obs_env[120]
                    heading_n = obs_env[121]
                    fwd_n = obs_env[122]
                    ang_n = obs_env[123]
                    print(f"Goal -> Dist: {dist_n:.3f}, Heading: {heading_n:.3f}")
                    print(f"Velocity -> Fwd: {fwd_n:.3f}, Ang: {ang_n:.3f}")
                    print(f"LiDAR Min -> {min(lidar):.3f}")
                print(f"Reward -> {rewards[idx]:.3f} (Done: {dones[idx]})")
                
            # Reset step count for the debug environment if it finished its episode
            if args.debug_env >= 0 and args.debug_env < args.num_envs and dones[args.debug_env]:
                step_count = 0
            else:
                step_count += 1
    except KeyboardInterrupt:
        print("\n[INFO] Play loop interrupted by user.")

    # --- Cleanup ---
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
