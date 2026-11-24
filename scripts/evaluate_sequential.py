#!/usr/bin/env python3
# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run multi-environment evaluation for a single model and write summary artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential evaluator for .zip and .pt navigation models.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model (.zip or .pt).")
    parser.add_argument(
        "--model_format",
        type=str,
        default="auto",
        choices=["auto", "zip", "pt"],
        help="Model format. 'auto' infers from extension.",
    )
    parser.add_argument(
        "--eval_modes",
        type=str,
        default="none,random,dense,gate_palisade_multi",
        help="Comma-separated obstacle modes to evaluate.",
    )
    parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments per evaluation run.")
    parser.add_argument("--eval_episodes", type=int, default=1000, help="Episodes per obstacle mode.")
    parser.add_argument("--headless", action="store_true", help="Run evaluations in headless mode.")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="Optional GPU index forwarded to evaluate.py.",
    )
    parser.add_argument(
        "--isaaclab_script",
        type=str,
        default="/opt/IsaacLab/isaaclab.sh",
        help="Isaac Lab launcher script path.",
    )
    parser.add_argument(
        "--evaluate_script",
        type=str,
        default="",
        help="Path to evaluate.py. Defaults to scripts/evaluate.py next to this file.",
    )
    parser.add_argument("--run_dir", type=str, default="", help="Optional explicit run root.")
    parser.add_argument(
        "--eval_scope",
        type=str,
        default="auto",
        choices=["auto", "global", "clients", "isolated"],
        help="Namespace under run_dir/evaluations.",
    )
    parser.add_argument(
        "--eval_label",
        type=str,
        default="",
        help="Model label under run_dir/evaluations/<scope>/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional explicit output directory for reports and summary.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="",
        help="Optional explicit summary JSON path.",
    )
    return parser


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
    stem, _ = os.path.splitext(os.path.basename(model_path))
    return stem or "model"


def resolve_evaluate_script_path(explicit_path: str) -> str:
    if explicit_path:
        return explicit_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py")


def resolve_output_directory(args: argparse.Namespace, model_format: str) -> tuple[str, str, str, str]:
    run_dir = args.run_dir or infer_run_dir(args.model_path)
    eval_scope = infer_eval_scope(args.model_path, model_format, args.eval_scope)
    eval_label = resolve_eval_label(args.model_path, args.eval_label)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(run_dir, "evaluations", eval_scope, eval_label)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, run_dir, eval_scope, eval_label


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_mode_csv(path: str, reports: list[dict[str, Any]]) -> None:
    fieldnames = [
        "obstacle_mode",
        "total_episodes",
        "success_rate_percent",
        "collision_rate_percent",
        "timeout_rate_percent",
        "trajectory_efficiency",
        "avg_steps_to_goal",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow({key: report.get(key) for key in fieldnames})


def aggregate_summary(per_mode_reports: list[dict[str, Any]]) -> dict[str, float]:
    if not per_mode_reports:
        return {
            "mean_success_rate_percent": 0.0,
            "mean_collision_rate_percent": 0.0,
            "mean_timeout_rate_percent": 0.0,
            "mean_trajectory_efficiency": 0.0,
            "mean_avg_steps_to_goal": 0.0,
            "worst_mode_success_rate_percent": 0.0,
            "best_mode_success_rate_percent": 0.0,
        }

    def mean(field: str) -> float:
        values = [float(report[field]) for report in per_mode_reports]
        return float(sum(values) / len(values))

    success_values = [float(report["success_rate_percent"]) for report in per_mode_reports]
    return {
        "mean_success_rate_percent": mean("success_rate_percent"),
        "mean_collision_rate_percent": mean("collision_rate_percent"),
        "mean_timeout_rate_percent": mean("timeout_rate_percent"),
        "mean_trajectory_efficiency": mean("trajectory_efficiency"),
        "mean_avg_steps_to_goal": mean("avg_steps_to_goal"),
        "worst_mode_success_rate_percent": float(min(success_values)),
        "best_mode_success_rate_percent": float(max(success_values)),
    }


def main() -> None:
    args = build_arg_parser().parse_args()

    model_format = infer_model_format(args.model_path, args.model_format)
    evaluate_script = resolve_evaluate_script_path(args.evaluate_script)
    output_dir, run_dir, eval_scope, eval_label = resolve_output_directory(args, model_format)
    eval_modes = [mode.strip() for mode in args.eval_modes.split(",") if mode.strip()]

    if not eval_modes:
        raise ValueError("No valid modes provided via --eval_modes")

    per_mode_reports: list[dict[str, Any]] = []
    for mode in eval_modes:
        report_path = os.path.join(output_dir, f"evaluation_results_{mode}.json")
        cmd = [
            args.isaaclab_script,
            "-p",
            evaluate_script,
            "--model_path",
            args.model_path,
            "--model_format",
            model_format,
            "--num_envs",
            str(args.num_envs),
            "--eval_episodes",
            str(args.eval_episodes),
            "--obstacle_mode",
            mode,
            "--report_path",
            report_path,
            "--run_dir",
            run_dir,
            "--eval_scope",
            eval_scope,
            "--eval_label",
            eval_label,
        ]
        if args.headless:
            cmd.append("--headless")
        if args.gpu_index is not None:
            cmd.extend(["--gpu-index", str(args.gpu_index)])

        print(f"[INFO] Evaluating mode={mode}")
        subprocess.run(cmd, check=True)
        per_mode_reports.append(read_json(report_path))

    aggregate = aggregate_summary(per_mode_reports)
    summary = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_path": args.model_path,
        "model_format": model_format,
        "run_dir": run_dir,
        "eval_scope": eval_scope,
        "eval_label": eval_label,
        "eval_modes": eval_modes,
        "num_envs": args.num_envs,
        "eval_episodes": args.eval_episodes,
        "aggregate": aggregate,
        "per_mode_reports": per_mode_reports,
    }

    summary_path = args.summary_path or os.path.join(output_dir, "summary.json")
    write_json(summary_path, summary)
    write_mode_csv(os.path.join(output_dir, "summary.csv"), per_mode_reports)
    print(f"[INFO] Sequential evaluation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
