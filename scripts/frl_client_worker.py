#!/usr/bin/env python3
# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Per-client Isaac worker for FRL baseline training.

Runs one client action in a fresh Isaac process:
- train_eval: train local TD3, evaluate, save model/replay/td3 state/metrics.

Global agent states are applied before local training (if --agent_in_path provided).
After loading the global state, target networks are hard-synced from online networks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback

import numpy as np

# --- Parse arguments before importing Isaac Lab ---
parser = argparse.ArgumentParser(description="FRL isolated client worker.")
parser.add_argument("--mode", type=str, required=True, choices=["train_eval"])
parser.add_argument("--headless", action="store_true", help="Run without GUI.")
parser.add_argument("--client_id", type=str, required=True)
parser.add_argument("--round_index", type=int, required=True)
parser.add_argument("--obstacle_mode", type=str, required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--seed", type=int, required=True, help="Client environment seed.")
parser.add_argument("--model_init_seed", type=int, required=True, help="Common seed for fresh model init.")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--replay_buffer_path", type=str, required=True)
parser.add_argument("--model_out_path", type=str, default="")
parser.add_argument("--replay_buffer_out_path", type=str, default="")
parser.add_argument("--tensorboard_log", type=str, required=True)
parser.add_argument("--local_timesteps", type=int, default=0)
parser.add_argument("--eval_episodes", type=int, default=0)
parser.add_argument("--pre_eval_episodes", type=int, default=0, help="Episodes to evaluate global model before local training (difficulty signal).")
parser.add_argument("--checkpoint_freq", type=int, default=0)
parser.add_argument("--train_progress_interval", type=int, default=10_000)
parser.add_argument("--eval_progress_interval", type=int, default=10)
parser.add_argument("--max_eval_steps", type=int, default=0)
parser.add_argument("--agent_in_path", type=str, default="")
parser.add_argument("--agent_out_path", type=str, default="")
parser.add_argument("--metrics_out_path", type=str, default="")
parser.add_argument("--soft_update_lambda", type=float, default=0.0, help="Blend local model with global (lambda=local fraction). 0=pure global, 1=pure local.")
parser.add_argument("--aggregate_critic", action="store_true", help="Include critic in extracted agent state for server aggregation.")
parser.add_argument("--defective_mode", type=str, default="none", choices=["none", "random", "stuck"],
                    help="Override all env actions: 'random' samples uniform random, 'stuck' outputs zero velocity.")
parser.add_argument("--fedkl_lambda1", type=float, default=0.0,
                    help="FedKL global penalty coefficient (MSE to round-start global model). 0 disables.")
parser.add_argument("--fedkl_lambda2", type=float, default=0.0,
                    help="FedKL local penalty coefficient (MSE to recent chunk snapshot). 0 disables.")
parser.add_argument("--fedkl_n_chunks", type=int, default=1,
                    help="Number of training chunks per round for FedKL local iteration simulation. "
                         "1 disables chunking (no recent_actor updates). Default: 1.")
parser.add_argument("--fedprox_mu", type=float, default=0.0,
                    help="FedProx proximal coefficient μ (parameter-space penalty to round-start global model). "
                         "0 disables FedProx. Required when --method fedprox.")

args = parser.parse_args()

# --- Launch Isaac Sim app ---
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# --- Now import Isaac Lab and env ---
import gymnasium as gym
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: F401

from envs.navigation_env import NavigationEnvCfg
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecEnvWrapper

from frl_utils import blend_agent_states, extract_agent_state, load_agent_state, sync_agent_targets_from_online


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Seed process-level RNGs early so downstream randomness is controlled.
set_global_seeds(args.seed)


def log(message: str) -> None:
    print(message, flush=True)


def make_env(obstacle_mode: str, num_envs: int, seed: int):
    log(
        f"[INFO] Building env: obstacle_mode={obstacle_mode}, num_envs={num_envs}, "
        f"seed={seed}"
    )
    env_start = time.time()
    env_cfg = NavigationEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.obstacle_mode = obstacle_mode

    env = gym.make("Isaac-Navigation-v0", cfg=env_cfg)
    log(f"[INFO] Env created in {time.time() - env_start:.2f}s.")

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
    return Sb3VecEnvWrapper(env)


class DefectiveActionWrapper(VecEnvWrapper):
    """Overrides every env action before it reaches the simulator.

    mode='random': replaces agent actions with a uniform random sample each step.
    mode='stuck':  replaces agent actions with zero velocity every step.

    Both modes affect training rollouts and evaluation equally, ensuring
    r_pre ≈ r_post ≈ 0 and delta_i ≈ 0 regardless of internal model weights.
    """

    def __init__(self, env, mode: str):
        assert mode in ("random", "stuck"), f"Unknown defective mode: {mode}"
        super().__init__(env)
        self.mode = mode

    def __getattr__(self, name: str):
        # Forward any attribute not defined on this wrapper to the underlying env.
        # This covers env.unwrapped (needed by evaluate_model for Isaac Lab extras),
        # env.seed(), env.render(), env.get_attr(), etc.
        return getattr(self.venv, name)

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions: np.ndarray) -> None:
        if self.mode == "random":
            # Sample independently for each env; action_space.sample() returns a
            # single-env action of shape (action_dim,), so stack across all envs to
            # produce the required (num_envs, action_dim) batched action.
            override = np.stack([self.action_space.sample() for _ in range(self.num_envs)])
        else:  # stuck
            override = np.zeros_like(actions)
        self.venv.step_async(override)

    def step_wait(self):
        return self.venv.step_wait()


def build_model(env, seed: int, tensorboard_log: str) -> TD3:
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
    model_kwargs = dict(
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
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if args.fedkl_lambda1 > 0 or args.fedkl_lambda2 > 0:
        from models.td3_fedkl import TD3FedKL
        return TD3FedKL(
            "MlpPolicy",
            env,
            fedkl_lambda1=args.fedkl_lambda1,
            fedkl_lambda2=args.fedkl_lambda2,
            **model_kwargs,
        )
    if args.fedprox_mu > 0:
        from models.td3_fedprox import TD3FedProx
        return TD3FedProx(
            "MlpPolicy",
            env,
            fedprox_mu=args.fedprox_mu,
            **model_kwargs,
        )
    return TD3("MlpPolicy", env, **model_kwargs)


def maybe_save_replay_buffer(model: TD3, path: str) -> None:
    if hasattr(model, "save_replay_buffer"):
        model.save_replay_buffer(path)


def maybe_load_replay_buffer(model: TD3, path: str) -> None:
    if hasattr(model, "load_replay_buffer") and os.path.exists(path):
        model.load_replay_buffer(path)


def load_or_create_client_model():
    env = make_env(args.obstacle_mode, args.num_envs, args.seed)
    if args.defective_mode != "none":
        env = DefectiveActionWrapper(env, mode=args.defective_mode)
        log(f"[INFO] Defective mode active: {args.defective_mode}")
    if os.path.exists(args.model_path):
        log(f"[INFO] Loading existing client model: {args.model_path}")
        if args.fedkl_lambda1 > 0 or args.fedkl_lambda2 > 0:
            from models.td3_fedkl import TD3FedKL
            model = TD3FedKL.load(args.model_path, env=env)
            model.fedkl_lambda1 = args.fedkl_lambda1
            model.fedkl_lambda2 = args.fedkl_lambda2
        elif args.fedprox_mu > 0:
            from models.td3_fedprox import TD3FedProx
            model = TD3FedProx.load(args.model_path, env=env)
            model.fedprox_mu = args.fedprox_mu
        else:
            model = TD3.load(args.model_path, env=env)
        model.tensorboard_log = args.tensorboard_log
        replay_load_start = time.time()
        maybe_load_replay_buffer(model, args.replay_buffer_path)
        log(f"[INFO] Replay buffer loaded, elapsed={time.time() - replay_load_start:.1f}s")
        return model, env

    # Keep fresh model initialization identical across clients.
    # Save and restore global RNG state so that model init does not clobber
    # the env/training RNG stream that was seeded from args.seed above.
    log(f"[INFO] Creating fresh client model with model_init_seed={args.model_init_seed}")
    _rng_py = random.getstate()
    _rng_np = np.random.get_state()
    _rng_torch_cpu = torch.get_rng_state()
    _rng_torch_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        set_global_seeds(args.model_init_seed)
        model = build_model(env, args.model_init_seed, args.tensorboard_log)
    finally:
        random.setstate(_rng_py)
        np.random.set_state(_rng_np)
        torch.set_rng_state(_rng_torch_cpu)
        if _rng_torch_cuda is not None:
            torch.cuda.set_rng_state_all(_rng_torch_cuda)
    model.tensorboard_log = args.tensorboard_log
    return model, env


def apply_agent_state_if_requested(model: TD3, env) -> None:
    if args.agent_in_path:
        agent_state = torch.load(args.agent_in_path, map_location="cpu")

        lam = getattr(args, "soft_update_lambda", 0.0)
        if lam > 0.0:
            local_state = extract_agent_state(model, include_actor=True, include_critic=True)
            agent_state = blend_agent_states(local_state, agent_state, lam)
            log(f"[INFO] Soft local update (lambda={lam}): {lam:.0%} local + {1.0-lam:.0%} global.")

        load_agent_state(model, agent_state)
        sync_agent_targets_from_online(model)
        log(f"[INFO] Applied agent state: {args.agent_in_path}")
        log("[INFO] Synced target networks from online networks.")

        # FedKL: snapshot global_actor and recent_actor from the freshly loaded global model.
        if hasattr(model, "set_fedkl_references"):
            model.set_fedkl_references()
            log("[INFO] FedKL: global_actor and recent_actor set from global model.")

        # FedProx: snapshot global_actor from the freshly loaded global model.
        if hasattr(model, "set_global_reference"):
            model.set_global_reference()
            log("[INFO] FedProx: global_actor set from global model.")


def evaluate_model(model: TD3, env, target_episodes: int) -> dict[str, float]:
    eval_start = time.time()
    log(f"[INFO] Evaluation start: target_episodes={target_episodes}")
    obs = env.reset()
    total_episodes = 0
    episode_successes: list[bool] = []
    episode_collisions: list[bool] = []
    episode_timeouts: list[bool] = []
    efficiencies: list[float] = []
    steps_to_goal: list[float] = []
    episode_returns: list[float] = []
    running_returns = np.zeros(env.num_envs, dtype=np.float64)
    env_steps = 0
    last_progress_print = 0

    while total_episodes < target_episodes and simulation_app.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        del infos
        env_steps += 1
        running_returns += rewards

        if args.max_eval_steps > 0 and env_steps >= args.max_eval_steps:
            log(
                f"[WARNING] Evaluation reached max_eval_steps={args.max_eval_steps}. "
                f"episodes_completed={total_episodes}/{target_episodes}"
            )
            break

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

        for local_index, env_index in enumerate(done_indices.tolist()):
            ep_global_idx = total_episodes + local_index
            episode_returns.append(float(running_returns[env_index]))
            running_returns[env_index] = 0.0
            if batch_success[local_index] and ep_global_idx < target_episodes:
                required_dist = max(batch_initial[local_index] - 0.2, 0.0)
                dist_traveled = max(batch_travel[local_index], 0.01)
                efficiencies.append(min(required_dist / dist_traveled, 1.0))
                steps_to_goal.append(float(batch_steps[local_index]))

        total_episodes += len(done_indices)
        if total_episodes - last_progress_print >= max(args.eval_progress_interval, 1):
            elapsed = time.time() - eval_start
            log(
                f"[INFO] Evaluation progress: episodes={total_episodes}/{target_episodes}, "
                f"env_steps={env_steps}, elapsed={elapsed:.1f}s"
            )
            last_progress_print = total_episodes

    if total_episodes == 0:
        return {
            "episodes": 0,
            "success_rate_percent": 0.0,
            "collision_rate_percent": 0.0,
            "timeout_rate_percent": 0.0,
            "avg_efficiency": 0.0,
            "avg_steps_to_goal": 0.0,
            "avg_return": 0.0,
        }

    clipped_returns = episode_returns[:target_episodes]
    clipped_successes = episode_successes[:target_episodes]
    clipped_collisions = episode_collisions[:target_episodes]
    clipped_timeouts = episode_timeouts[:target_episodes]
    n = len(clipped_returns)
    summary = {
        "episodes": n,
        "success_rate_percent": (sum(clipped_successes) / n) * 100.0,
        "collision_rate_percent": (sum(clipped_collisions) / n) * 100.0,
        "timeout_rate_percent": (sum(clipped_timeouts) / n) * 100.0,
        "avg_efficiency": float(np.mean(efficiencies)) if efficiencies else 0.0,
        "avg_steps_to_goal": float(np.mean(steps_to_goal)) if steps_to_goal else 0.0,
        "avg_return": float(np.mean(clipped_returns)) if clipped_returns else 0.0,
    }
    elapsed = time.time() - eval_start
    log(
        f"[INFO] Evaluation done: episodes={summary['episodes']}, "
        f"success={summary['success_rate_percent']:.2f}%, "
        f"avg_return={summary['avg_return']:.3f}, elapsed={elapsed:.1f}s"
    )
    return summary


def train_one_round(model: TD3) -> float:
    class ProgressCallback(BaseCallback):
        def __init__(self, interval: int, local_budget: int, client_name: str, round_id: int):
            super().__init__()
            self.interval = max(interval, 1)
            self.local_budget = max(local_budget, 1)
            self.client_name = client_name
            self.round_id = round_id
            self.start_steps = 0
            self.last_print = 0
            self.start_time = 0.0

        def _on_training_start(self) -> None:
            self.start_steps = int(self.model.num_timesteps)
            self.start_time = time.time()

        def _on_step(self) -> bool:
            local_steps = int(self.model.num_timesteps) - self.start_steps
            if local_steps - self.last_print >= self.interval:
                elapsed = time.time() - self.start_time
                log(
                    f"[INFO] Train progress ({self.client_name}, round {self.round_id}): "
                    f"local_steps={local_steps}/{self.local_budget}, elapsed={elapsed:.1f}s"
                )
                self.last_print = local_steps
            return True

    callbacks: list[BaseCallback] = [
        ProgressCallback(
            interval=args.train_progress_interval,
            local_budget=args.local_timesteps,
            client_name=args.client_id,
            round_id=args.round_index,
        )
    ]
    
    if args.checkpoint_freq > 0:
        checkpoint_dir = os.path.join(
            os.path.dirname(args.model_path), "checkpoints", f"round_{args.round_index:03d}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // max(1, model.get_env().num_envs), 1),
                save_path=checkpoint_dir,
                name_prefix="checkpoint",
            )
        )

    n_chunks = getattr(args, "fedkl_n_chunks", 1)
    train_start = time.time()

    if hasattr(model, "update_recent_actor") and n_chunks > 1:
        # FedKL chunked training: simulate local iterations by snapshotting
        # recent_actor between chunks, making the local penalty reference drift
        # away from the global model after the first chunk.
        chunk_size = args.local_timesteps // n_chunks
        remainder = args.local_timesteps % n_chunks
        if remainder != 0:
            log(
                f"[WARNING] FedKL ({args.client_id}, round {args.round_index}): "
                f"local_timesteps={args.local_timesteps} is not evenly divisible by "
                f"n_chunks={n_chunks}. Remainder {remainder} steps will be added to the "
                f"last chunk."
            )
        log(
            f"[INFO] FedKL train start ({args.client_id}, round {args.round_index}): "
            f"local_timesteps={args.local_timesteps}, n_chunks={n_chunks}, chunk_size={chunk_size}"
        )
        for chunk_idx in range(n_chunks):
            effective_size = chunk_size + (remainder if chunk_idx == n_chunks - 1 else 0)
            chunk_callbacks: list[BaseCallback] = [
                ProgressCallback(
                    interval=args.train_progress_interval,
                    local_budget=effective_size,
                    client_name=args.client_id,
                    round_id=args.round_index,
                )
            ]
            if args.checkpoint_freq > 0:
                checkpoint_dir = os.path.join(
                    os.path.dirname(args.model_path), "checkpoints", f"round_{args.round_index:03d}"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                chunk_callbacks.append(
                    CheckpointCallback(
                        save_freq=max(args.checkpoint_freq // max(1, model.get_env().num_envs), 1),
                        save_path=checkpoint_dir,
                        name_prefix=f"checkpoint_chunk{chunk_idx}",
                    )
                )
            model.learn(
                total_timesteps=effective_size,
                callback=chunk_callbacks,
                log_interval=10,
                reset_num_timesteps=False,
            )
            model.update_recent_actor()
            log(
                f"[INFO] FedKL chunk {chunk_idx + 1}/{n_chunks} done "
                f"({args.client_id}, round {args.round_index}): recent_actor updated."
            )
    else:
        log(
            f"[INFO] Train start ({args.client_id}, round {args.round_index}): "
            f"local_timesteps={args.local_timesteps}"
        )
        model.learn(
            total_timesteps=args.local_timesteps,
            callback=callbacks,
            log_interval=10,
            reset_num_timesteps=False,
        )

    elapsed = time.time() - train_start
    log(f"[INFO] Train done ({args.client_id}, round {args.round_index}), elapsed={elapsed:.1f}s")



def write_json(path: str, payload: dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def run_train_eval() -> None:
    log(
        f"[INFO] Worker start: client={args.client_id}, round={args.round_index}, "
        f"local_timesteps={args.local_timesteps}, eval_episodes={args.eval_episodes}"
    )
    model, env = load_or_create_client_model()

    # ── All Isaac-Sim-dependent work ─────────────────────────────────────────
    # env.close() is guaranteed by the finally block even if an exception occurs.
    # File I/O is intentionally deferred to after env.close() so that no live
    # Isaac Sim / CUDA resources are held during disk operations.
    agent_state = None
    metrics: dict[str, float | None] = {
        "episodes": None,
        "success_rate_percent": None,
        "collision_rate_percent": None,
        "timeout_rate_percent": None,
        "avg_efficiency": None,
        "avg_steps_to_goal": None,
        "avg_return": None,
    }
    try:
        apply_agent_state_if_requested(model, env)

        # Phase wall-clock, persisted so the pre-evaluation overhead can be
        # measured after the fact instead of re-read from terminal scrollback.
        # Denominator is the server's round_elapsed_seconds, not the sum of
        # these three -- a round also pays env creation, replay-buffer I/O and
        # model saves, none of which are FedDIA-specific.
        if args.pre_eval_episodes > 0:
            log(f"[INFO] Pre-training evaluation: target_episodes={args.pre_eval_episodes}")
            phase_start = time.time()
            pre_metrics = evaluate_model(model, env, args.pre_eval_episodes)
            metrics["pre_eval_seconds"] = time.time() - phase_start
            metrics["pre_avg_return"] = pre_metrics["avg_return"]
            metrics["pre_success_rate_percent"] = pre_metrics["success_rate_percent"]

        if args.local_timesteps > 0:
            phase_start = time.time()
            train_one_round(model)
            metrics["train_seconds"] = time.time() - phase_start

        if args.eval_episodes > 0:
            phase_start = time.time()
            eval_metrics = evaluate_model(model, env, args.eval_episodes)
            metrics["post_eval_seconds"] = time.time() - phase_start
            metrics.update(eval_metrics)

        # Extract agent online state while the sim is still alive (model may use CUDA).
        if args.agent_out_path:
            agent_state = extract_agent_state(
                model, 
                include_critic=args.aggregate_critic,
            )
    finally:
        log(f"[INFO] Closing environment for {args.client_id}...")
        try:
            env.close()
        except Exception as exc:
            log(f"[WARN] env.close() raised: {exc}")
        log(f"[INFO] Environment closed for {args.client_id}.")

    # ── File I/O — env is closed; no live Isaac Sim resources held ───────────
    model_out_path = args.model_out_path or args.model_path
    replay_out_path = args.replay_buffer_out_path or args.replay_buffer_path

    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    t0 = time.time()
    model.save(model_out_path)
    log(f"[INFO] Saved model ({time.time() - t0:.1f}s): {model_out_path}")

    os.makedirs(os.path.dirname(replay_out_path), exist_ok=True)
    t0 = time.time()
    maybe_save_replay_buffer(model, replay_out_path)
    log(f"[INFO] Saved replay buffer ({time.time() - t0:.1f}s): {replay_out_path}")

    if agent_state is not None:
        os.makedirs(os.path.dirname(args.agent_out_path), exist_ok=True)
        t0 = time.time()
        torch.save(agent_state, args.agent_out_path)
        log(f"[INFO] Saved agent state ({time.time() - t0:.1f}s): {args.agent_out_path}")

    if args.metrics_out_path:
        # Attach seed provenance so logs can confirm per-client/per-round uniqueness.
        # returns_hash fingerprints the key eval metrics — if two clients produce the
        # same hash it means their evaluation sequences were identical (a sign of RNG
        # collision). With per-client, per-round worker seeds this should never happen.
        import hashlib
        metrics["worker_seed"] = args.seed
        metrics["model_init_seed"] = args.model_init_seed
        metrics["round_index"] = args.round_index
        _fingerprint = json.dumps({
            "avg_return": metrics.get("avg_return"),
            "success_rate_percent": metrics.get("success_rate_percent"),
            "episodes": metrics.get("episodes"),
        }, sort_keys=True)
        metrics["returns_hash"] = hashlib.sha256(_fingerprint.encode()).hexdigest()[:16]
        t0 = time.time()
        write_json(args.metrics_out_path, metrics)
        log(f"[INFO] Saved metrics ({time.time() - t0:.1f}s): {args.metrics_out_path}")


def main() -> None:
    if args.mode != "train_eval":
        raise ValueError(f"Unsupported mode: {args.mode}")
    run_train_eval()


if __name__ == "__main__":
    _exit_code = 0
    try:
        main()
    except Exception as exc:
        _exit_code = 1
        log(f"[ERROR] Worker terminated with exception: {exc}")
        traceback.print_exc()
    finally:
        if simulation_app.is_running():
            log("[INFO] Closing simulation app...")
            simulation_app.close()
            log("[INFO] Simulation app closed.")
        # Isaac Sim can leave background threads that prevent normal Python exit.
        # This is a subprocess; os._exit() guarantees the process terminates.
        os._exit(_exit_code)
