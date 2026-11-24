#!/usr/bin/env python3
# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Process-isolated FRL baseline runner for TD3 navigation.

Implements the following methods (--method):
1. isolated         -- local training only, no aggregation
2. fedavg           -- uniform averaging
3. reward_weighted  -- softmax over client returns
4. feddia_m         -- FedDIA (proposed): improvement-gated, difficulty-amplified
5. difficulty_only  -- FedDIA ablation: difficulty signal alone
6. fgrl             -- growth-value weighting (Tian et al., 2024)
7. fedkl            -- KL-style global/local penalties during local training
8. fedprox          -- proximal term during local training

This script is a pure controller process. It launches one Isaac worker process
per client per round (train_eval mode), aggregates agent states on the server,
and writes per-round logs for analysis. The canonical federated agent state is always
saved as global_agent_states/latest_agent.pt, with optional per-round history files.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass

def build_arg_parser() -> argparse.ArgumentParser:
    # Parse arguments before importing Isaac Lab / torch-heavy modules.
    parser = argparse.ArgumentParser(description="Train FRL baselines (process-isolated clients).")
    parser.add_argument("--headless", action="store_true", help="Run worker clients without GUI.")
    parser.add_argument(
        "--method",
        type=str,
        default="reward_weighted",
        choices=["isolated", "fedavg", "reward_weighted", "feddia_m", "difficulty_only", "fgrl", "fedkl", "fedprox"],
        help="Federated baseline to run.",
    )
    parser.add_argument(
        "--client_modes",
        type=str,
        default="none,random,dense,gate_palisade_multi",
        help="Comma-separated obstacle modes, one per client.",
    )
    parser.add_argument(
        "--clients_per_round",
        type=int,
        default=0,
        help=(
            "Partial participation: train and aggregate this many clients each round, "
            "sampled uniformly without replacement from the fleet. 0 (default) means "
            "full participation. An absolute count rather than a fraction, so the "
            "per-round cost is explicit and there is no rounding rule."
        ),
    )
    parser.add_argument("--num_envs", type=int, default=64, help="Parallel envs per client.")
    parser.add_argument("--rounds", type=int, default=3, help="Federated rounds.")
    parser.add_argument(
        "--local_timesteps",
        type=int,
        default=50_000,
        help="Local RL training timesteps per client per round.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=50,
        help="Evaluation episodes per client after each round.",
    )
    parser.add_argument(
        "--pre_eval_episodes",
        type=int,
        default=0,
        help="Episodes to evaluate global model before local training; required by feddia_m and difficulty_only for weighting.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Base directory for FRL outputs.",
    )
    parser.add_argument(
        "--resume_run_dir",
        type=str,
        default="",
        help=(
            "Existing run directory to resume (e.g., logs/fedavg/run_YYYYMMDD_HHMMSS_seed_42). "
            "When set, round execution continues from the next unfinished round."
        ),
    )
    parser.add_argument(
        "--weight_metric",
        type=str,
        default="avg_return",
        choices=["avg_return", "success_rate_percent"],
        help="Metric used as extrinsic client utility.",
    )
    parser.add_argument(
        "--weight_normalization",
        type=str,
        default="rank",
        choices=["raw", "rank", "zscore", "minmax"],
        help="Server-side normalization mode before softmax weighting.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature for every score-based aggregation method "
            "(reward_weighted, feddia_m, difficulty_only). "
            "Lower concentrates weight on the top-scoring client; higher "
            "approaches uniform. Must be > 0."
        ),
    )
    parser.add_argument(
        "--aggregate_critic",
        action="store_true",
        help="Include critic in agent state aggregation across clients. Off by default (actor-only).",
    )
    parser.add_argument(
        "--soft_update_lambda",
        type=float,
        default=0.0,
        help=(
            "FGRL soft local update coefficient \u03bb [0.0, 1.0). "
            "0.0 = full global replacement (default, backward compatible). "
            "0.9 = FGRL paper value: clients retain 90%% of their local weights per round."
        ),
    )
    parser.add_argument(
        "--fgrl_psi",
        type=float,
        default=0.5,
        help="FGRL growth value balance coefficient \u03c8 \u2208 (0,1). Weights success-rate (\u03b6) vs steps (\u03c6) terms in g_m. Paper value: 0.5.",
    )
    parser.add_argument(
        "--fgrl_omega",
        type=float,
        default=0.1,
        help="FGRL growth threshold \u03a9. Agents with \u03b7 < \u03a9 are excluded from aggregation this round. Paper value: 0.1.",
    )
    parser.add_argument(
        "--feddia_m_alpha",
        type=float,
        default=1.0,
        help=(
            "Amplification strength for difficulty bonus in feddia_m (>= 0). "
            "score_i = s_improv * (1 + alpha * s_diff), "
            "where s_improv = minmax(max(0, r_post - r_pre)) and s_diff = 1 - minmax(r_pre). "
            "alpha=0 -> purely improvement-driven (no difficulty bonus); "
            "alpha=1.0 -> hard env gets 2x bonus over easy env."
        ),
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=0,
        help="Optional checkpoint frequency inside local training; 0 disables it.",
    )
    parser.add_argument(
        "--train_progress_interval",
        type=int,
        default=10_000,
        help="Print training heartbeat every N local timesteps.",
    )
    parser.add_argument(
        "--eval_progress_interval",
        type=int,
        default=10,
        help="Print evaluation heartbeat every N completed episodes.",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=0,
        help="Optional hard cap on evaluation env-steps per client (0 disables cap).",
    )
    parser.add_argument(
        "--isaaclab_script",
        type=str,
        default="/opt/IsaacLab/isaaclab.sh",
        help="Path to Isaac Lab launcher script used for worker processes.",
    )
    parser.add_argument(
        "--keep_global_history",
        action="store_true",
        help="Keep per-round global agent snapshots (round_XXX.pt).",
    )
    parser.add_argument(
        "--no_cleanup_client_agent_states",
        "--no_cleanup_client_td3_states",
        dest="cleanup_client_agent_states",
        action="store_false",
        default=True,
        help="Preserve per-round client agent_state.pt files (default behavior is cleanup).",
    )
    parser.add_argument(
        "--no_cleanup_replay_buffers",
        dest="cleanup_replay_buffers",
        action="store_false",
        default=True,
        help=(
            "Preserve each client's latest_replay_buffer.pkl after the run finishes. "
            "By default, replay buffers are deleted once training completes, since they "
            "are only needed to resume an in-progress run (resuming a finished run simply "
            "starts with an empty buffer). Pass this flag to keep them."
        ),
    )

    parser.add_argument(
        "--defective_client_idx",
        type=int,
        default=-1,
        help="Index of the client to simulate as defective (-1 disables). Used with --defective_mode.",
    )
    parser.add_argument(
        "--defective_mode",
        type=str,
        default="none",
        choices=["none", "random", "stuck"],
        help="Action override for the defective client: 'random' samples uniform random, 'stuck' outputs zero velocity.",
    )
    parser.add_argument(
        "--fedkl_lambda1",
        type=float,
        default=0.0,
        help="FedKL global penalty coefficient (MSE to round-start global model). Required when --method fedkl.",
    )
    parser.add_argument(
        "--fedkl_lambda2",
        type=float,
        default=0.0,
        help="FedKL local penalty coefficient (MSE to recent chunk snapshot). Required when --method fedkl.",
    )
    parser.add_argument(
        "--fedkl_n_chunks",
        type=int,
        default=6,
        help="Number of training chunks per round for FedKL local iteration simulation (default: 6 × 50k = 300k).",
    )
    parser.add_argument(
        "--fedprox_mu",
        type=float,
        default=0.0,
        help="FedProx proximal coefficient μ (parameter-space penalty to round-start global model). Required when --method fedprox.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="Optional GPU index to expose via CUDA_VISIBLE_DEVICES for controller and workers.",
    )
    return parser


def parse_client_modes(spec: str) -> list[str]:
    """Expand a --client_modes spec into one mode per client.

    Accepts the plain form, "none,none,random,dense", and a repeat form where a
    mode is followed by *count: "none*20,random*10,dense*10". The two may be
    mixed. The repeat form exists so a large fleet does not depend on the caller
    building a 40-entry string in the shell -- an unset variable silently becomes
    an empty fleet, and a miscounted one silently becomes the wrong fleet.
    """
    modes: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "*" not in token:
            modes.append(token)
            continue
        mode, _, count_text = token.partition("*")
        mode, count_text = mode.strip(), count_text.strip()
        if not mode or not count_text.isdigit() or int(count_text) < 1:
            raise ValueError(
                f"Malformed --client_modes entry {token!r}. The repeat form is "
                "mode*count with a positive integer count, e.g. 'none*20'."
            )
        modes.extend([mode] * int(count_text))
    if not modes:
        raise ValueError(
            f"--client_modes expanded to zero clients from {spec!r}. If this came "
            "from a shell variable, it was unset or empty -- the repeat form "
            "'none*20,random*10,dense*10' avoids the shell entirely."
        )
    return modes


args = build_arg_parser().parse_args()

# Methods that derive aggregation weights from the pre-training evaluation
# (r_i^pre). Used by both the hard requirement below and the "set but unused"
# warning further down -- keep them reading one list so a new method cannot be
# added to one check and missed in the other.
PRE_EVAL_METHODS = ("feddia_m", "difficulty_only")

if args.feddia_m_alpha < 0.0:
    raise ValueError("--feddia_m_alpha must be >= 0.0.")
if not (0.0 <= args.soft_update_lambda < 1.0):
    raise ValueError("--soft_update_lambda must be in [0.0, 1.0).")
if args.method in PRE_EVAL_METHODS and args.pre_eval_episodes <= 0:
    raise ValueError(
        f"--method {args.method} requires --pre_eval_episodes > 0. "
        "Without pre-evaluation, all difficulty scores are 0.0 and the method silently "
        "degrades to reward_weighted."
    )
if args.method == "difficulty_only" and args.feddia_m_alpha != 1.0:
    print(
        "[WARNING] --feddia_m_alpha is set but method is difficulty_only, which has no "
        "improvement term for alpha to amplify. The value is ignored.",
        flush=True,
    )
if args.method == "fedkl" and args.fedkl_lambda1 <= 0 and args.fedkl_lambda2 <= 0:
    raise ValueError(
        "--method fedkl requires at least one of --fedkl_lambda1 or --fedkl_lambda2 to be > 0."
    )
if args.method == "fedkl" and args.fedkl_lambda2 > 0 and args.fedkl_n_chunks <= 1:
    print(
        "[WARNING] --fedkl_lambda2 > 0 but --fedkl_n_chunks <= 1: recent_actor will never "
        "be updated during local training, making the local penalty identical to the global "
        "penalty (both anchor to the round-start model). Set --fedkl_n_chunks >= 2 to use "
        "the local penalty meaningfully."
    )
if args.method == "fedprox" and args.fedprox_mu <= 0:
    raise ValueError("--method fedprox requires --fedprox_mu > 0.")
if args.eval_episodes <= 0 and args.method != "isolated":
    raise ValueError(
        "--eval_episodes must be > 0 for non-isolated methods. "
        "Post-training evaluation is required to compute aggregation metrics "
        "(avg_return, success_rate_percent, etc.) and to populate round_metrics.jsonl."
    )
if args.clients_per_round < 0:
    raise ValueError(
        f"--clients_per_round must be >= 0, got {args.clients_per_round}. "
        "Use 0 for full participation."
    )
_num_clients = len(parse_client_modes(args.client_modes))
if args.clients_per_round > _num_clients:
    raise ValueError(
        f"--clients_per_round {args.clients_per_round} exceeds the fleet size "
        f"({_num_clients} clients) from --client_modes={args.client_modes!r}."
    )
if args.clients_per_round and args.method == "isolated":
    raise ValueError(
        "--clients_per_round is meaningless for --method isolated: there is no "
        "aggregation, so every client must train every round."
    )
if args.defective_mode != "none" and args.defective_client_idx < 0:
    raise ValueError(
        "--defective_mode requires --defective_client_idx >= 0. "
        "Specify which client index should be defective."
    )
if args.defective_client_idx >= 0 and args.defective_mode == "none":
    raise ValueError(
        "--defective_client_idx is set but --defective_mode is 'none'. "
        "Specify --defective_mode random or --defective_mode stuck."
    )
if args.pre_eval_episodes > 0 and args.method not in PRE_EVAL_METHODS:
    print(
        f"[WARNING] --pre_eval_episodes is set but method is not one of {', '.join(PRE_EVAL_METHODS)}. "
        "Pre-evaluation episodes will run, but they are not used for aggregation weights.",
        flush=True,
    )

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpu_index is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    print(
        f"[INFO] GPU selection: using GPU index {args.gpu_index} "
        "(CUDA_DEVICE_ORDER=PCI_BUS_ID).",
        flush=True,
    )
else:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(
        f"[INFO] GPU selection: using default visibility "
        f"(CUDA_VISIBLE_DEVICES={visible_devices}, CUDA_DEVICE_ORDER=PCI_BUS_ID).",
        flush=True,
    )

import torch

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"run_{timestamp}_seed_{args.seed}"
if args.resume_run_dir:
    args.log_dir = os.path.abspath(args.resume_run_dir)
else:
    args.log_dir = os.path.join(args.log_dir, args.method, run_name)

from frl_utils import (
    append_jsonl,
    aggregate_agent_states,
    build_weight_report,
    ensure_dir,
    normalize_utilities,
    sanitize_for_json,
    save_json,
    softmax_weights,
    uniform_weights,
)


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class ClientSpec:
    client_id: str
    obstacle_mode: str
    seed: int
    directory: str
    model_path: str
    replay_buffer_path: str
    model_init_seed: int
    agent_state_path: str
    metrics_path: str
    current_weight: float


def initialize_clients(client_modes: list[str], run_dir: str, seed: int) -> list[ClientSpec]:
    clients: list[ClientSpec] = []
    for index, mode in enumerate(client_modes):
        client_id = f"client_{index}"
        client_dir = ensure_dir(os.path.join(run_dir, "clients", client_id))
        clients.append(
            ClientSpec(
                client_id=client_id,
                obstacle_mode=mode,
                seed=seed + index,
                directory=client_dir,
                model_path=os.path.join(client_dir, "latest_model.zip"),
                replay_buffer_path=os.path.join(client_dir, "latest_replay_buffer.pkl"),
                model_init_seed=seed,
                agent_state_path=os.path.join(client_dir, "latest_agent.pt"),
                metrics_path=os.path.join(client_dir, "latest_metrics.json"),
                current_weight=1.0,
            )
        )
    return clients


def build_round_artifacts(client: ClientSpec, round_index: int) -> dict[str, str]:
    round_dir = ensure_dir(os.path.join(client.directory, "round_artifacts", f"round_{round_index:03d}"))
    return {
        "round_dir": round_dir,
        "metrics_path": os.path.join(round_dir, "metrics.json"),
        "agent_state_path": os.path.join(round_dir, "agent_state.pt"),
        "model_path": os.path.join(round_dir, "model.zip"),
        "replay_buffer_path": os.path.join(round_dir, "replay_buffer.pkl"),
    }


def load_round_agent_state(path: str, cleanup: bool) -> dict[str, dict[str, torch.Tensor]]:
    agent_state = torch.load(path, map_location="cpu")
    if cleanup and os.path.exists(path):
        os.remove(path)
    return agent_state


def discover_completed_rounds(run_dir: str) -> list[int]:
    completed: list[int] = []
    for file_name in os.listdir(run_dir):
        if not file_name.startswith("round_") or not file_name.endswith("_summary.json"):
            continue
        middle = file_name[len("round_") : -len("_summary.json")]
        if middle.isdigit():
            completed.append(int(middle))
    return sorted(set(completed))


def load_existing_config_if_present(run_dir: str) -> dict | None:
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_resume_compatibility(run_dir: str, client_modes: list[str]) -> None:
    existing_config = load_existing_config_if_present(run_dir)
    if existing_config is None:
        return

    if existing_config.get("method") != args.method:
        raise ValueError(
            "Resume method mismatch: "
            f"existing='{existing_config.get('method')}', requested='{args.method}'."
        )

    existing_modes = existing_config.get("client_modes")
    if isinstance(existing_modes, list) and existing_modes != client_modes:
        raise ValueError(
            "Resume client_modes mismatch between existing run and requested args."
        )

    numeric_keys = ["num_envs", "local_timesteps", "eval_episodes", "pre_eval_episodes", "feddia_m_alpha", "seed",
                    "fedkl_lambda1", "fedkl_lambda2", "fedkl_n_chunks", "fedprox_mu", "defective_client_idx",
                    "fgrl_psi", "fgrl_omega", "soft_update_lambda", "clients_per_round"]
    for key in numeric_keys:
        existing_value = existing_config.get(key)
        requested_value = getattr(args, key)
        if existing_value is not None and existing_value != requested_value:
            raise ValueError(
                f"Resume {key} mismatch: existing={existing_value}, requested={requested_value}."
            )
    # Validate additional important flags that affect training behavior
    bool_keys = ["aggregate_critic", "keep_global_history"]
    for key in bool_keys:
        existing_value = existing_config.get(key)
        requested_value = getattr(args, key, False)
        if existing_value is not None and bool(existing_value) != bool(requested_value):
            raise ValueError(
                f"Resume {key} mismatch: existing={existing_value}, requested={requested_value}."
            )

    # Validate server-side weighting parameters
    for key in ["weight_metric", "weight_normalization", "defective_mode"]:
        existing_value = existing_config.get(key)
        requested_value = getattr(args, key)
        if existing_value is not None and existing_value != requested_value:
            raise ValueError(
                f"Resume {key} mismatch: existing={existing_value}, requested={requested_value}."
            )
    existing_temp = existing_config.get("temperature")
    if existing_temp is not None:
        if abs(float(existing_temp) - float(args.temperature)) > 1e-9:
            raise ValueError(
                f"Resume temperature mismatch: existing={existing_temp}, requested={args.temperature}."
            )

    # Checkpoint frequency affects saved artifacts
    existing_cp = existing_config.get("checkpoint_freq")
    if existing_cp is not None and int(existing_cp) != int(args.checkpoint_freq):
        raise ValueError(
            f"Resume checkpoint_freq mismatch: existing={existing_cp}, requested={args.checkpoint_freq}."
        )


def run_worker(
    client: ClientSpec,
    round_index: int,
    mode: str,
    agent_in_path: str | None,
    metrics_out_path: str | None,
    agent_out_path: str | None,
    model_out_path: str | None,
    replay_buffer_out_path: str | None,
    defective_mode: str = "none",
) -> None:
    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frl_client_worker.py")

    # Derive a per-round worker seed so each round sees a different (but deterministic)
    # sequence of env resets / obstacle & goal placements.
    # Formula: worker_seed = client.seed + round_index * 10000
    #   - client.seed   = base_seed + client_index  → distinct per client
    #   - round_index * 10000                        → distinct per round
    # The large multiplier (10000) avoids overlap between client offsets (max ~N clients)
    # and round offsets. model_init_seed is kept fixed so fresh model weights are
    # identical across rounds and methods for the same replicate.
    worker_seed = client.seed + round_index * 10000

    cmd = [
        args.isaaclab_script,
        "-p",
        worker_script,
        "--mode",
        mode,
        "--client_id",
        client.client_id,
        "--round_index",
        str(round_index),
        "--obstacle_mode",
        client.obstacle_mode,
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(worker_seed),
        "--model_init_seed",
        str(client.model_init_seed),
        "--model_path",
        client.model_path,
        "--replay_buffer_path",
        client.replay_buffer_path,
        "--tensorboard_log",
        client.directory,
        "--local_timesteps",
        str(args.local_timesteps),
        "--eval_episodes",
        str(args.eval_episodes),
        "--pre_eval_episodes",
        str(args.pre_eval_episodes),
        "--checkpoint_freq",
        str(args.checkpoint_freq),
        "--train_progress_interval",
        str(args.train_progress_interval),
        "--eval_progress_interval",
        str(args.eval_progress_interval),
        "--max_eval_steps",
        str(args.max_eval_steps),
    ]

    if args.headless:
        cmd.append("--headless")

    if args.aggregate_critic:
        cmd.append("--aggregate_critic")
    if getattr(args, "soft_update_lambda", 0.0) > 0.0:
        cmd.extend(["--soft_update_lambda", str(args.soft_update_lambda)])
    if agent_in_path:
        cmd.extend(["--agent_in_path", agent_in_path])
    if metrics_out_path:
        cmd.extend(["--metrics_out_path", metrics_out_path])
    if agent_out_path:
        cmd.extend(["--agent_out_path", agent_out_path])
    if model_out_path:
        cmd.extend(["--model_out_path", model_out_path])
    if replay_buffer_out_path:
        cmd.extend(["--replay_buffer_out_path", replay_buffer_out_path])
    if defective_mode != "none":
        cmd.extend(["--defective_mode", defective_mode])
    if args.method == "fedkl":
        cmd.extend(["--fedkl_lambda1", str(args.fedkl_lambda1)])
        cmd.extend(["--fedkl_lambda2", str(args.fedkl_lambda2)])
        cmd.extend(["--fedkl_n_chunks", str(args.fedkl_n_chunks)])
    if args.method == "fedprox":
        cmd.extend(["--fedprox_mu", str(args.fedprox_mu)])

    log(
        f"[INFO] Launching worker ({mode}) for {client.client_id} "
        f"round={round_index}, obstacle={client.obstacle_mode}"
    )
    try:
        worker_env = dict(os.environ)
        worker_env['DISPLAY'] = ''  # Prevent X11 race condition crash in headless Isaac Sim
        subprocess.run(cmd, check=True, env=worker_env)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Worker failed for {client.client_id} round={round_index} mode={mode} "
            f"exit_code={error.returncode}"
        ) from error


def promote_round_artifact(src: str, dst: str) -> None:
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing round artifact: {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.replace(src, dst)


def prune_empty_round_artifact_dirs(clients: list[ClientSpec]) -> None:
    """Remove empty round_artifacts/round_XXX directories left after artifact promotion."""
    for client in clients:
        base = os.path.join(client.directory, "round_artifacts")
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            round_dir = os.path.join(base, name)
            if not os.path.isdir(round_dir):
                continue
            # Only prune leaf round directories (e.g., round_001).
            if os.listdir(round_dir):
                continue
            os.rmdir(round_dir)


def load_prev_metrics(log_dir: str) -> dict:
    """Load the last completed round's per-client metrics from round_metrics.jsonl.

    Used by FGRL to reconstruct prev_metrics after a crash and resume.
    Stores all clients unconditionally so resume behaviour matches the live loop.
    has_valid_data in resolve_weights guards against zero-zeta or zero-phi baselines.
    Returns {} for fresh runs or when the JSONL does not exist yet.

    LIMITATION under --clients_per_round: this reads only the highest round present,
    which under partial participation holds just that round's cohort. A resumed FGRL
    run therefore recovers baselines for the cohort alone, and non-participants fall
    back to the has_valid_data guard on their next selection rather than to the
    baseline they actually last recorded. Correct fix is to take each client's most
    recent entry across all rounds. Not done: FGRL is not run with partial
    participation, and changing it would alter existing full-participation resumes.
    """
    path = os.path.join(log_dir, "round_metrics.jsonl")
    if not os.path.exists(path):
        return {}
    rounds: dict[int, dict[str, dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed line (e.g. crash mid-write)
            rounds.setdefault(entry["round"], {})[entry["client_id"]] = entry
    if not rounds:
        return {}
    last = rounds[max(rounds)]
    result = {}
    for cid, entry in last.items():
        zeta = entry["metrics"]["success_rate_percent"] / 100.0
        phi = entry["metrics"].get("avg_steps_to_goal", 0.0)
        result[cid] = {"zeta": zeta, "phi": phi}
    return result


def resolve_weights(
    method: str,
    client_metrics: dict[str, dict[str, float]],
    round_index: int = 1,
    prev_metrics: dict | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    client_ids = list(client_metrics.keys())
    raw_utilities = {client_id: float(client_metrics[client_id][args.weight_metric]) for client_id in client_ids}

    if method == "isolated":
        weights = {client_id: 0.0 for client_id in client_ids}
        return raw_utilities, raw_utilities, weights
    if method == "fedavg" or method == "fedkl" or method == "fedprox":
        weights = uniform_weights(client_ids)
        return raw_utilities, raw_utilities, weights

    if method == "feddia_m":
        # FedDIA: Difficulty- and Improvement-Aware aggregation.
        #
        # Two signals, both measured against the same received global model so
        # they are comparable across clients:
        #   s_improv_i = minmax(max(0, post_i - pre_i))   [improvement this round]
        #   s_diff_i   = 1 - minmax(pre_i)                [environment difficulty]
        #
        # Combined multiplicatively:
        #   score_i = s_improv_i * (1 + alpha * s_diff_i)
        #   weight_i = softmax(score, temperature)
        #
        # Properties:
        #   - Difficulty acts as a bonus multiplier, not an independent signal.
        #   - Improvement gates the difficulty bonus: if s_improv_i = 0 (client
        #     did not improve, e.g. a defective or stuck client), score_i = 0
        #     provided at least one other client improved this round (see
        #     all-zero guard below for the all-plateau case).
        #   - At all-plateau (all s_improv -> 0): an explicit guard detects this
        #     before normalization and returns uniform FedAvg weights directly.
        #     This enforces the stated property rather than relying on softmax
        #     over near-zero scores, which would be corrupted by the minmax
        #     tie-breaking fallback (1/N) in normalize_utilities.
        #   - The hard-to-easy score ratio is constant at (1 + alpha) throughout
        #     training, independent of the absolute improvement magnitude.
        #   - alpha is amplification strength, not a blend fraction:
        #       alpha=0   -> score_i = s_improv_i (no difficulty bonus)
        #       alpha=1.0 -> hard env gets 2x bonus over easy env
        #       alpha=2.0 -> hard env gets 3x bonus over easy env
        pre_returns_raw = {cid: client_metrics[cid].get("pre_avg_return") for cid in client_ids}
        missing = [cid for cid, v in pre_returns_raw.items() if v is None]
        if missing:
            raise ValueError(
                f"feddia_m: pre_avg_return is missing for clients {missing}. "
                "Ensure --pre_eval_episodes > 0 and all workers completed pre-evaluation."
            )
        pre_returns = {cid: float(v) for cid, v in pre_returns_raw.items()}
        post_returns = {cid: float(client_metrics[cid].get("avg_return") or 0.0) for cid in client_ids}

        deltas = {cid: max(0.0, post_returns[cid] - pre_returns[cid]) for cid in client_ids}

        # All-zero delta guard: if no client improved this round, the multiplicative
        # gate produces score=0 for all clients. Rather than passing all-zero scores
        # through softmax (uniform) or risking the minmax 1/N fallback corrupting
        # norm_delta, return uniform FedAvg directly. Enforces the stated property:
        # difficulty can only amplify improvement, never substitute for it.
        if all(v < 1e-8 for v in deltas.values()):
            weights = uniform_weights(client_ids)
            return post_returns, {cid: 1.0 / len(client_ids) for cid in client_ids}, weights

        norm_delta = normalize_utilities(deltas, "minmax")

        norm_pre = normalize_utilities(pre_returns, "minmax")
        difficulty = {cid: 1.0 - norm_pre[cid] for cid in client_ids}

        alpha = args.feddia_m_alpha
        combined_utilities = {
            cid: norm_delta[cid] * (1.0 + alpha * difficulty[cid])
            for cid in client_ids
        }
        weights = softmax_weights(combined_utilities, args.temperature)
        return post_returns, combined_utilities, weights

    if method == "difficulty_only":
        # Ablation of FedDIA: the improvement gate removed.
        #
        #   score_i = c_i^D = 1 - minmax(pre_i)
        #
        # Same difficulty signal and the same softmax as feddia_m above, but no
        # delta term, so weight follows environment difficulty alone.
        #
        # This is NOT a proposed method and not a value of alpha -- alpha
        # parametrizes c^I * (1 + alpha * c^D), which is gated by c^I at every
        # alpha, so difficulty-only lies outside that family and needs its own
        # branch. It exists to answer "could you get FedDIA's gains by always
        # favouring hard clients?".
        #
        # NO all-zero guard, deliberately. feddia_m falls back to uniform when
        # no client improves, because its gate zeroes every score in that case.
        # Here there is no gate, so that condition cannot arise, and adding a
        # guard would reintroduce part of the very mechanism this ablation
        # removes. A plateaued -- or defective -- client in a hard environment
        # therefore keeps its weight, which is exactly the behaviour under test.
        #
        # Edge case: if every client reports the same pre_avg_return, minmax
        # has no spread and normalize_utilities returns its 1/N tie-break, so
        # weights come out uniform. Harmless, and correct.
        pre_returns_raw = {cid: client_metrics[cid].get("pre_avg_return") for cid in client_ids}
        missing = [cid for cid, v in pre_returns_raw.items() if v is None]
        if missing:
            raise ValueError(
                f"difficulty_only: pre_avg_return is missing for clients {missing}. "
                "Ensure --pre_eval_episodes > 0 and all workers completed pre-evaluation."
            )
        pre_returns = {cid: float(v) for cid, v in pre_returns_raw.items()}
        post_returns = {cid: float(client_metrics[cid].get("avg_return") or 0.0) for cid in client_ids}

        norm_pre = normalize_utilities(pre_returns, "minmax")
        difficulty = {cid: 1.0 - norm_pre[cid] for cid in client_ids}
        weights = softmax_weights(difficulty, args.temperature)
        return post_returns, difficulty, weights

    if method == "fgrl":
        # Growth-value weighted aggregation (FGRL, Tian et al., Appl. Sci. 2024).
        #
        # Each round, compute per-client growth value g_m^k from cross-round deltas
        # in success rate (ζ) and avg steps to goal (φ). Map through sigmoid → η.
        # Clients with η < Ω are excluded from aggregation (K' ≤ K).
        # Weights are proportional to η over passing clients (not softmax).
        # Soft local update (λ) is handled in the worker before load_agent_state.
        #
        # g_m^k = ψ*(Δζ/ζ_{m-1}) + (1-ψ)*(Δφ/φ_{m-1})   [Eq. 10 as written]
        # η_m^k = σ(g_m^k)
        # exclude if η_m^k < Ω; fallback to uniform if K'=0
        #
        # Unified η=1 fallback rule (no reliable growth signal):
        #   - round 1 or client not in prev_metrics
        #   - this round had no successes (ζ_m = 0, φ_m meaningless)
        #   - prev round had no successes (ζ_{m-1} = 0, relative Δ undefined)
        if prev_metrics is None:
            prev_metrics = {}
        psi = getattr(args, "fgrl_psi", 0.5)
        omega = getattr(args, "fgrl_omega", 0.1)

        eta: dict[str, float | None] = {}  # None = excluded from aggregation
        for cid in client_ids:
            zeta_m = client_metrics[cid]["success_rate_percent"] / 100.0
            phi_m = client_metrics[cid].get("avg_steps_to_goal", 0.0)
            prev = prev_metrics.get(cid)

            has_valid_data = (
                round_index > 1
                and prev is not None
                and zeta_m > 0.0            # this round had successes
                and prev["zeta"] > 1e-8     # prev round had successes (no div-by-zero)
                and prev.get("phi", 0.0) > 0.0  # prev phi valid
            )

            if not has_valid_data:
                eta[cid] = 1.0  # neutral — round-1 or no reliable signal
            else:
                zeta_prev = prev["zeta"]
                phi_prev = prev["phi"]
                d_zeta = (zeta_m - zeta_prev) / zeta_prev
                d_phi = (phi_m - phi_prev) / phi_prev  # Eq. 10 as written (sign inconsistency preserved)
                g = psi * d_zeta + (1.0 - psi) * d_phi
                eta_raw = 1.0 / (1.0 + math.exp(-g))
                eta[cid] = eta_raw if eta_raw >= omega else None  # None = excluded

        # K'=0 fallback: if all agents excluded, revert to uniform FedAvg
        passing = {cid: v for cid, v in eta.items() if v is not None}
        if not passing:
            log("[INFO] FGRL: all agents excluded (K'=0); falling back to uniform FedAvg.")
            passing = {cid: 1.0 for cid in client_ids}

        total_eta = sum(passing.values())
        weights = {cid: (passing[cid] / total_eta if cid in passing else 0.0) for cid in client_ids}
        # Use eta values as normalized utilities for logging (0.0 for excluded)
        normalized_utilities = {cid: (eta[cid] if eta[cid] is not None else 0.0) for cid in client_ids}
        return raw_utilities, normalized_utilities, weights

    normalized = normalize_utilities(raw_utilities, args.weight_normalization)
    weights = softmax_weights(normalized, args.temperature)
    return raw_utilities, normalized, weights


def main() -> None:
    run_dir = ensure_dir(args.log_dir)
    client_modes = parse_client_modes(args.client_modes)
    if args.resume_run_dir:
        validate_resume_compatibility(run_dir, client_modes)

    clients = initialize_clients(client_modes, run_dir, args.seed)
    if args.defective_client_idx >= len(clients):
        raise ValueError(
            f"--defective_client_idx {args.defective_client_idx} is out of range: "
            f"only {len(clients)} clients (0–{len(clients)-1})."
        )
    # Base seeds must be spaced further apart than the fleet size. client.seed is
    # base_seed + index, so runs at base seeds 32/42/52 with 40 clients produce
    # overlapping ranges (32-71, 42-81, 52-91) and therefore identical worker seeds
    # -- and identical env reset/obstacle/goal streams -- across supposedly
    # independent replicates. Warn rather than raise: only the caller knows which
    # other base seeds this run will be compared against.
    if len(clients) > 10:
        log(
            f"[INFO] Fleet size {len(clients)}: client seeds span "
            f"{args.seed}-{args.seed + len(clients) - 1}. Replicate base seeds closer "
            f"together than {len(clients)} share worker seeds on the overlap."
        )
    prune_empty_round_artifact_dirs(clients)
    global_agent_in_path: str | None = None

    jsonl_path = os.path.join(run_dir, "round_metrics.jsonl")
    aggregated_agent_dir = ensure_dir(os.path.join(run_dir, "global_agent_states"))
    latest_global_agent_path = os.path.join(aggregated_agent_dir, "latest_agent.pt")
    prev_metrics: dict = load_prev_metrics(run_dir) if args.method == "fgrl" else {}

    completed_rounds = discover_completed_rounds(run_dir) if args.resume_run_dir else []
    last_completed_round = completed_rounds[-1] if completed_rounds else 0
    start_round = last_completed_round + 1 if args.resume_run_dir else 1

    if args.resume_run_dir:
        log(
            f"[INFO] Resume mode enabled: run_dir={run_dir}, "
            f"completed_rounds={len(completed_rounds)}, last_completed_round={last_completed_round}, "
            f"target_rounds={args.rounds}"
        )

        if args.method != "isolated" and last_completed_round > 0:
            if os.path.exists(latest_global_agent_path):
                global_agent_in_path = latest_global_agent_path
            else:
                fallback_snapshot = os.path.join(
                    aggregated_agent_dir, f"round_{last_completed_round:03d}.pt"
                )
                if os.path.exists(fallback_snapshot):
                    global_agent_in_path = fallback_snapshot
                    log(
                        "[WARNING] latest_agent.pt missing in resumed run; "
                        f"using snapshot fallback: {fallback_snapshot}"
                    )
                else:
                    raise FileNotFoundError(
                        "Cannot resume federated global state: missing latest_agent.pt and round snapshot "
                        f"for round {last_completed_round}."
                    )

        if start_round > args.rounds:
            log(
                f"[INFO] Nothing to do: completed rounds already cover target rounds "
                f"(completed={last_completed_round}, target={args.rounds})."
            )
            return

    save_json(
        os.path.join(run_dir, "config.json"),
        {
            "method": args.method,
            "client_modes": client_modes,
            "clients_per_round": args.clients_per_round,
            "num_envs": args.num_envs,
            "rounds": args.rounds,
            "local_timesteps": args.local_timesteps,
            "eval_episodes": args.eval_episodes,
            "pre_eval_episodes": args.pre_eval_episodes,
            "feddia_m_alpha": args.feddia_m_alpha,
            "seed": args.seed,
            "weight_metric": args.weight_metric,
            "weight_normalization": args.weight_normalization,
            "temperature": args.temperature,
            "isaaclab_script": args.isaaclab_script,
            "keep_global_history": args.keep_global_history,
            "cleanup_client_agent_states": args.cleanup_client_agent_states,
            "cleanup_replay_buffers": args.cleanup_replay_buffers,
            "gpu_index": args.gpu_index,
            "aggregate_critic": getattr(args, "aggregate_critic", False),
            "headless": args.headless,
            "checkpoint_freq": args.checkpoint_freq,
            "train_progress_interval": args.train_progress_interval,
            "eval_progress_interval": args.eval_progress_interval,
            "max_eval_steps": args.max_eval_steps,
            "soft_update_lambda": getattr(args, "soft_update_lambda", 0.0),
            "fgrl_psi": getattr(args, "fgrl_psi", 0.5),
            "fgrl_omega": getattr(args, "fgrl_omega", 0.1),
            "fedkl_lambda1": getattr(args, "fedkl_lambda1", 0.0),
            "fedkl_lambda2": getattr(args, "fedkl_lambda2", 0.0),
            "fedkl_n_chunks": getattr(args, "fedkl_n_chunks", 6),
            "fedprox_mu": getattr(args, "fedprox_mu", 0.0),
            "defective_client_idx": getattr(args, "defective_client_idx", -1),
            "defective_mode": getattr(args, "defective_mode", "none"),
            "resume_run_dir": args.resume_run_dir,
            "resume_last_completed_round": last_completed_round,
            "resume_start_round": start_round,
        },
    )

    for round_index in range(start_round, args.rounds + 1):
        round_start = time.time()
        log(f"[INFO] Starting round {round_index}/{args.rounds}")
        client_agent_states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        client_metrics: dict[str, dict[str, float]] = {}
        round_artifacts_by_client: dict[str, dict[str, str]] = {}

        # Partial participation: sample this round's cohort. The RNG is seeded from
        # (base_seed, round_index) alone -- never from the method -- so FedAvg and
        # FedDIA at the same seed see an identical cohort sequence and any difference
        # between them cannot be sampling luck. It is also stateless per round rather
        # than one generator advanced across rounds, so a resumed run draws the
        # cohorts it would have drawn had it never stopped. Seeded from a string so
        # it cannot collide with worker_seed = client.seed + round_index * 10000.
        # Indices, not client_ids: sorting client_ids puts client_10 before client_2.
        if args.clients_per_round and args.clients_per_round < len(clients):
            cohort_rng = random.Random(f"cohort|{args.seed}|{round_index}")
            picked = sorted(cohort_rng.sample(range(len(clients)), args.clients_per_round))
        else:
            picked = list(range(len(clients)))
        cohort = [(idx, clients[idx]) for idx in picked]
        if args.clients_per_round:
            log(
                f"[INFO] Round {round_index} cohort ({len(cohort)}/{len(clients)}): "
                + ", ".join(f"{c.client_id}({c.obstacle_mode})" for _, c in cohort)
            )

        for client_idx, client in cohort:
            log(f"[INFO] Preparing {client.client_id} ({client.obstacle_mode})")
            artifacts = build_round_artifacts(client, round_index)
            agent_in_path = global_agent_in_path if args.method != "isolated" else None
            client_defective_mode = (
                args.defective_mode if client_idx == args.defective_client_idx else "none"
            )
            run_worker(
                client=client,
                round_index=round_index,
                mode="train_eval",
                agent_in_path=agent_in_path,
                metrics_out_path=artifacts["metrics_path"],
                agent_out_path=artifacts["agent_state_path"],
                model_out_path=artifacts["model_path"],
                replay_buffer_out_path=artifacts["replay_buffer_path"],
                defective_mode=client_defective_mode,
            )
            round_artifacts_by_client[client.client_id] = artifacts
            with open(artifacts["metrics_path"], "r", encoding="utf-8") as metrics_file:
                metrics = sanitize_for_json(json.load(metrics_file))
            client_metrics[client.client_id] = metrics
            client_agent_states[client.client_id] = load_round_agent_state(
                path=artifacts["agent_state_path"],
                cleanup=args.cleanup_client_agent_states,
            )

        raw_utilities, normalized_utilities, weights = resolve_weights(
            args.method, client_metrics, round_index, prev_metrics
        )

        if args.method == "fgrl":
            for _, client in cohort:
                cid = client.client_id
                zeta_m = client_metrics[cid]["success_rate_percent"] / 100.0
                phi_m = client_metrics[cid].get("avg_steps_to_goal", 0.0)
                # Always overwrite with current round's data so prev_metrics never
                # goes stale. has_valid_data in resolve_weights already guards against
                # using a zero-zeta or zero-phi baseline (prev["zeta"] > 1e-8 check).
                prev_metrics[cid] = {"zeta": zeta_m, "phi": phi_m}

        if args.method != "isolated":
            aggregated_agent_state = aggregate_agent_states(client_agent_states, weights)
            torch.save(aggregated_agent_state, latest_global_agent_path)
            log(f"[INFO] Saved canonical global agent state: {latest_global_agent_path}")
            global_agent_in_path = latest_global_agent_path

            if args.keep_global_history:
                agent_snapshot_path = os.path.join(aggregated_agent_dir, f"round_{round_index:03d}.pt")
                torch.save(aggregated_agent_state, agent_snapshot_path)
                log(f"[INFO] Saved global agent state history snapshot: {agent_snapshot_path}")

        # Commit client round artifacts atomically after the round fully succeeds.
        # Cohort only -- non-participants produced no artifacts this round, and their
        # committed model/buffer/metrics from their last participation stay untouched.
        for _, client in cohort:
            artifacts = round_artifacts_by_client[client.client_id]
            promote_round_artifact(artifacts["model_path"], client.model_path)
            promote_round_artifact(artifacts["replay_buffer_path"], client.replay_buffer_path)
            promote_round_artifact(artifacts["metrics_path"], client.metrics_path)
        prune_empty_round_artifact_dirs(clients)

        weight_report = build_weight_report(raw_utilities, normalized_utilities, weights)
        round_summary = {
            "round": round_index,
            "method": args.method,
            # Non-participants are absent from "clients" rather than present with
            # nulls, so downstream code can distinguish "did not train this round"
            # from "trained and scored zero".
            "participants": [client.client_id for _, client in cohort],
            "num_clients_total": len(clients),
            "weights": weight_report,
            "clients": {},
        }

        for _, client in cohort:
            client_payload = {
                "round": round_index,
                "method": args.method,
                "client_id": client.client_id,
                "obstacle_mode": client.obstacle_mode,
                "local_timesteps": args.local_timesteps,
                "metrics": client_metrics[client.client_id],
                "weight_info": weight_report[client.client_id],
                "model_path": client.model_path,
            }
            append_jsonl(jsonl_path, sanitize_for_json(client_payload))
            round_summary["clients"][client.client_id] = client_payload

        # Persisted so per-round overhead is recoverable afterwards. This is the
        # denominator for the pre-evaluation overhead: it covers the whole round,
        # including env creation, replay-buffer I/O and model saves, not just the
        # three phases the worker times. Measured before writing the summary --
        # that write is milliseconds and is server bookkeeping, not round work.
        round_elapsed = time.time() - round_start
        round_summary["round_elapsed_seconds"] = round_elapsed

        save_json(
            os.path.join(run_dir, f"round_{round_index:03d}_summary.json"),
            sanitize_for_json(round_summary),
        )

        log(f"[INFO] Round {round_index} complete, elapsed={round_elapsed:.1f}s")

    # Training finished successfully (all rounds complete). A client's replay buffer is
    # only needed to resume an in-progress run, so delete them here unless asked to keep
    # them. This runs only on normal completion; a crash before this point leaves the
    # buffers intact for resuming. Resuming a finished run simply starts with an empty
    # buffer (a graceful, supported path). See --no_cleanup_replay_buffers.
    if args.cleanup_replay_buffers:
        removed = 0
        for client in clients:
            buf_path = client.replay_buffer_path
            if os.path.exists(buf_path):
                try:
                    os.remove(buf_path)
                    removed += 1
                except OSError as exc:
                    log(f"[WARNING] Could not remove replay buffer {buf_path}: {exc}")
        if removed:
            log(
                f"[INFO] Removed {removed} replay buffer(s) on completion "
                f"(pass --no_cleanup_replay_buffers to keep them)."
            )

    log(f"[INFO] Federated baseline run complete. Logs saved to: {run_dir}")


if __name__ == "__main__":
    main()