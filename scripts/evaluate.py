#!/usr/bin/env python3
# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Unified quantitative evaluator for navigation policies.

Supports two model formats:
- zip: SB3 TD3 checkpoint (.zip)
- pt: PyTorch state dict (actor +/- critic) exported by FRL global aggregation

Writes evaluation JSON reports with a consistent schema across both formats.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import json
import numpy as np

# --- Parse arguments before importing Isaac Lab ---
parser = argparse.ArgumentParser(description="Quantitatively Evaluate navigation agent.")
parser.add_argument(
    "--headless", action="store_true", default=True, help="Force headless mode for pure evaluation."
)
parser.add_argument(
    "--num_envs", type=int, default=256, help="Number of parallel environments."
)
parser.add_argument(
    "--eval_episodes", type=int, default=1000, help="Total number of episodes to record before exiting."
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to model (.zip SB3 checkpoint or .pt agent state).",
)
parser.add_argument(
    "--model_format",
    type=str,
    default="auto",
    choices=["auto", "zip", "pt"],
    help="Model format. 'auto' infers from file extension.",
)
parser.add_argument(
    "--report_path",
    type=str,
    default="",
    help="Optional explicit output path for evaluation JSON report.",
)
parser.add_argument(
    "--run_dir",
    type=str,
    default="",
    help="Optional run root. Used for default centralized eval output layout.",
)
parser.add_argument(
    "--eval_scope",
    type=str,
    default="auto",
    choices=["auto", "global", "clients", "isolated"],
    help="Evaluation scope namespace for default report path.",
)
parser.add_argument(
    "--eval_label",
    type=str,
    default="",
    help="Optional model label for default report path.",
)
parser.add_argument(
    "--obstacle_mode", type=str, default="random", choices=["none", "random", "dense", "random_dense", "gate_palisade_multi", "factory", "living"], help="Obstacle distribution mode."
)
parser.add_argument(
    "--gpu-index",
    type=int,
    default=None,
    help="Optional GPU index to expose via CUDA_VISIBLE_DEVICES.",
)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpu_index is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    print(
        f"[INFO] GPU selection: using GPU index {args.gpu_index} "
        "(CUDA_DEVICE_ORDER=PCI_BUS_ID)."
    )
else:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(
        f"[INFO] GPU selection: using default visibility "
        f"(CUDA_VISIBLE_DEVICES={visible_devices}, CUDA_DEVICE_ORDER=PCI_BUS_ID)."
    )

# --- Launch Isaac Sim app ---
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# --- Now import Isaac Lab and env ---
import gymnasium as gym
import torch

# Register our environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: F401 — triggers gym.register

from envs.navigation_env import NavigationEnvCfg
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from frl_utils import load_agent_state


def infer_model_format(path: str, explicit_format: str) -> str:
    if explicit_format != "auto":
        return explicit_format
    _, extension = os.path.splitext(path)
    extension = extension.lower()
    if extension == ".zip":
        return "zip"
    if extension == ".pt":
        return "pt"
    raise ValueError(f"Unable to infer model format from extension: {path}")


def infer_run_dir(model_path: str) -> str:
    normalized = os.path.normpath(model_path)
    marker = f"{os.sep}global_agent_states{os.sep}"
    if marker in normalized:
        return normalized.split(marker)[0]
    marker = f"{os.sep}clients{os.sep}"
    if marker in normalized:
        return normalized.split(marker)[0]
    return os.path.dirname(normalized)


def infer_eval_scope(model_path: str, model_format: str, explicit_scope: str) -> str:
    if explicit_scope != "auto":
        return explicit_scope
    if model_format == "pt":
        return "global"
    normalized = os.path.normpath(model_path)
    if f"{os.sep}isolated{os.sep}" in normalized:
        return "isolated"
    return "clients"


def resolve_eval_label(model_path: str, explicit_label: str) -> str:
    if explicit_label:
        return explicit_label
    filename = os.path.basename(model_path)
    stem, _ = os.path.splitext(filename)
    return stem or "model"


def resolve_report_path(model_path: str, model_format: str) -> tuple[str, str, str, str]:
    if args.report_path:
        report_path = args.report_path
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        scope = infer_eval_scope(model_path, model_format, args.eval_scope)
        label = resolve_eval_label(model_path, args.eval_label)
        run_dir = args.run_dir or infer_run_dir(model_path)
        return report_path, run_dir, scope, label

    run_dir = args.run_dir or infer_run_dir(model_path)
    scope = infer_eval_scope(model_path, model_format, args.eval_scope)
    label = resolve_eval_label(model_path, args.eval_label)
    output_dir = os.path.join(run_dir, "evaluations", scope, label)
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"evaluation_results_{args.obstacle_mode}.json")
    return report_path, run_dir, scope, label


def build_eval_model_for_agent_state(env) -> TD3:
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
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
        verbose=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def load_model(env, model_path: str, model_format: str) -> TD3:
    if model_format == "zip":
        print(f"[INFO] Loading SB3 model from: {model_path}")
        return TD3.load(model_path, env=env)

    print(f"[INFO] Building eval model scaffold for agent-state: {model_path}")
    model = build_eval_model_for_agent_state(env)
    agent_state = torch.load(model_path, map_location="cpu")
    load_agent_state(model, agent_state)
    return model


def evaluate_policy(model, env) -> dict[str, float]:
    total_episodes = 0
    episode_successes: list[bool] = []
    episode_collisions: list[bool] = []
    episode_timeouts: list[bool] = []
    efficiencies: list[float] = []
    steps_to_goal: list[float] = []

    print(f"[INFO] Starting quantitative evaluation... Goal: {args.eval_episodes} episodes.")
    obs = env.reset()

    try:
        while total_episodes < args.eval_episodes and simulation_app.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            del rewards, infos

            done_indices = np.where(dones)[0]
            if len(done_indices) == 0:
                continue

            ext = env.unwrapped.extras
            batch_success = ext["success"][done_indices].cpu().numpy()
            batch_coll = ext["collision"][done_indices].cpu().numpy()
            batch_to = ext["timeout"][done_indices].cpu().numpy()
            batch_travel = ext["travel_dist"][done_indices].cpu().numpy()
            batch_initial = ext["initial_dist"][done_indices].cpu().numpy()
            batch_steps = ext["episode_steps"][done_indices].cpu().numpy()

            episode_successes.extend(batch_success.tolist())
            episode_collisions.extend(batch_coll.tolist())
            episode_timeouts.extend(batch_to.tolist())

            for index in range(len(done_indices)):
                ep_global_idx = total_episodes + index
                if batch_success[index] and ep_global_idx < args.eval_episodes:
                    required_dist = max(batch_initial[index] - 0.2, 0.0)
                    dist_traveled = max(batch_travel[index], 0.01)
                    efficiency = min(required_dist / dist_traveled, 1.0)
                    efficiencies.append(float(efficiency))
                    steps_to_goal.append(float(batch_steps[index]))

            total_episodes += len(done_indices)
            if total_episodes % 100 < len(done_indices):
                print(f"[{total_episodes}/{args.eval_episodes}] Episodes evaluated...")
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted early by user.")

    if total_episodes == 0:
        return {
            "total_episodes": 0,
            "success_rate_percent": 0.0,
            "collision_rate_percent": 0.0,
            "timeout_rate_percent": 0.0,
            "trajectory_efficiency": 0.0,
            "avg_steps_to_goal": 0.0,
        }

    clipped_successes = episode_successes[:args.eval_episodes]
    clipped_collisions = episode_collisions[:args.eval_episodes]
    clipped_timeouts = episode_timeouts[:args.eval_episodes]
    n = len(clipped_successes)
    avg_efficiency = float(np.mean(efficiencies)) if efficiencies else 0.0
    avg_steps = float(np.mean(steps_to_goal)) if steps_to_goal else 0.0
    return {
        "total_episodes": n,
        "success_rate_percent": (sum(clipped_successes) / n) * 100.0,
        "collision_rate_percent": (sum(clipped_collisions) / n) * 100.0,
        "timeout_rate_percent": (sum(clipped_timeouts) / n) * 100.0,
        "trajectory_efficiency": float(avg_efficiency),
        "avg_steps_to_goal": float(avg_steps),
    }


def print_report(metrics: dict[str, float]) -> None:
    print("=========================================")
    print(f"      QUANTITATIVE EVAL RESULTS ({args.obstacle_mode.upper()}) ")
    print("=========================================")
    print(f"Total Episodes Executed : {metrics['total_episodes']}")
    print(f"Success Rate            : {metrics['success_rate_percent']:.2f}%")
    print(f"Collision/Crash Rate    : {metrics['collision_rate_percent']:.2f}%")
    print(f"Timeout/Stall Rate      : {metrics['timeout_rate_percent']:.2f}%")
    print(f"Trajectory Efficiency   : {metrics['trajectory_efficiency']:.3f} (1.0 = perfect straight line)")
    print(f"Avg Steps to Goal       : {metrics['avg_steps_to_goal']:.1f} steps (successful runs only)")
    print("=========================================")


def save_csv_sidecar(report_path: str, payload: dict[str, float | str]) -> None:
    csv_path = os.path.splitext(report_path)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)


def main():
    """Evaluate navigation model and save report."""
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file not found at: {args.model_path}")
        sys.exit(1)

    model_format = infer_model_format(args.model_path, args.model_format)
    report_path, run_dir, eval_scope, eval_label = resolve_report_path(args.model_path, model_format)

    # Hand-authored scenes (factory, living) are separate env classes selected
    # by gym id rather than by obstacle_mode.
    if args.obstacle_mode in envs.SCENE_MODES:
        gym_id, cfg_cls = envs.SCENE_MODES[args.obstacle_mode]
        env_cfg = cfg_cls()
        env_cfg.scene.num_envs = args.num_envs
        env = gym.make(gym_id, cfg=env_cfg)
    else:
        env_cfg = NavigationEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.obstacle_mode = args.obstacle_mode
        env = gym.make("Isaac-Navigation-v0", cfg=env_cfg)

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    env = Sb3VecEnvWrapper(env)
    model = load_model(env, args.model_path, model_format)
    metrics = evaluate_policy(model, env)
    print_report(metrics)

    report_dict = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_path": args.model_path,
        "model_format": model_format,
        "run_dir": run_dir,
        "eval_scope": eval_scope,
        "eval_label": eval_label,
        "obstacle_mode": args.obstacle_mode,
        **metrics,
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report_dict, file, indent=2)

    save_csv_sidecar(report_path, report_dict)
    print(f"[INFO] Report saved to: {report_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app.is_running():
            simulation_app.close()
