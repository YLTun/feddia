#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
from typing import Any

import torch


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def append_jsonl(path: str, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


def _get_agent_module(model, name: str) -> torch.nn.Module:
    module = getattr(model, name, None)
    if module is None and hasattr(model, "policy"):
        module = getattr(model.policy, name, None)
    if module is None:
        raise AttributeError(f"Agent module '{name}' not found on model.")
    return module


def extract_agent_state(
    model,
    include_actor: bool = True,
    include_critic: bool = True,
    include_targets: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    state: dict[str, dict[str, torch.Tensor]] = {}
    if include_actor:
        actor = _get_agent_module(model, "actor")
        state["actor"] = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    if include_critic:
        critic = _get_agent_module(model, "critic")
        state["critic"] = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
    if include_targets:
        if include_actor:
            actor_target = _get_agent_module(model, "actor_target")
            state["actor_target"] = {k: v.detach().cpu().clone() for k, v in actor_target.state_dict().items()}
        if include_critic:
            critic_target = _get_agent_module(model, "critic_target")
            state["critic_target"] = {k: v.detach().cpu().clone() for k, v in critic_target.state_dict().items()}
    return state


def load_agent_state(model, state: dict[str, dict[str, torch.Tensor]]) -> None:
    module_map = {
        "actor": "actor",
        "critic": "critic",
    }
    for key, attr in module_map.items():
        if key in state:
            _get_agent_module(model, attr).load_state_dict(state[key])


def blend_agent_states(
    local_state: dict[str, dict[str, torch.Tensor]],
    global_state: dict[str, dict[str, torch.Tensor]],
    lam: float,
) -> dict[str, dict[str, torch.Tensor]]:
    """Soft local update: blended = lam * local + (1 - lam) * global.

    Used by FGRL workers before loading the global model state. lam=0.9 means
    the client retains 90% of its local weights per round. Only modules present
    in global_state are blended; the result has the same keys as global_state.
    """
    blended: dict[str, dict[str, torch.Tensor]] = {}
    for module_name, global_module in global_state.items():
        blended[module_name] = {}
        local_module = local_state.get(module_name, {})
        for tensor_name, global_tensor in global_module.items():
            local_tensor = local_module.get(tensor_name)
            if local_tensor is not None:
                blended[module_name][tensor_name] = (
                    lam * local_tensor.float() + (1.0 - lam) * global_tensor.float()
                )
            else:
                blended[module_name][tensor_name] = global_tensor.clone()
    return blended


def sync_agent_targets_from_online(model) -> None:
    actor = _get_agent_module(model, "actor")
    actor_target = _get_agent_module(model, "actor_target")
    critic = _get_agent_module(model, "critic")
    critic_target = _get_agent_module(model, "critic_target")
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())


def aggregate_agent_states(
    client_states: dict[str, dict[str, dict[str, torch.Tensor]]],
    weights: dict[str, float],
) -> dict[str, dict[str, torch.Tensor]]:
    if not client_states:
        raise ValueError("No client states provided for aggregation.")

    reference_client = next(iter(client_states))
    aggregated: dict[str, dict[str, torch.Tensor]] = {}
    for module_name, module_state in client_states[reference_client].items():
        aggregated[module_name] = {}
        for tensor_name in module_state:
            combined = None
            for client_id, state in client_states.items():
                contribution = state[module_name][tensor_name].float() * float(weights[client_id])
                combined = contribution if combined is None else combined + contribution
            aggregated[module_name][tensor_name] = combined
    return aggregated


def normalize_utilities(
    utilities: dict[str, float],
    mode: str,
) -> dict[str, float]:
    if mode == "raw":
        return dict(utilities)

    ordered_ids = list(utilities.keys())
    values = torch.tensor([utilities[client_id] for client_id in ordered_ids], dtype=torch.float32)

    if mode == "rank":
        sort_indices = torch.argsort(values)
        normalized = torch.zeros_like(values)
        if len(values) == 1:
            normalized[0] = 0.0
        else:
            for rank, idx in enumerate(sort_indices.tolist()):
                normalized[idx] = rank / (len(values) - 1)
        return {client_id: float(normalized[i].item()) for i, client_id in enumerate(ordered_ids)}

    if mode == "zscore":
        mean = values.mean()
        std = values.std(unbiased=False)
        if float(std.item()) < 1e-8:
            normalized = torch.zeros_like(values)
        else:
            normalized = (values - mean) / std
        return {client_id: float(normalized[i].item()) for i, client_id in enumerate(ordered_ids)}

    if mode == "minmax":
        min_val = values.min()
        max_val = values.max()
        range_val = max_val - min_val
        if float(range_val.item()) < 1e-8:
            # All clients identical — no information to differentiate; fall back to uniform
            normalized = torch.full_like(values, 1.0 / len(values))
        else:
            normalized = (values - min_val) / range_val
        return {client_id: float(normalized[i].item()) for i, client_id in enumerate(ordered_ids)}

    raise ValueError(f"Unsupported normalization mode: {mode}")


def softmax_weights(
    utilities: dict[str, float],
    temperature: float,
) -> dict[str, float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    ordered_ids = list(utilities.keys())
    values = torch.tensor([utilities[client_id] for client_id in ordered_ids], dtype=torch.float32)
    scaled = values / temperature
    scaled = scaled - torch.max(scaled)
    probs = torch.softmax(scaled, dim=0)
    return {client_id: float(probs[i].item()) for i, client_id in enumerate(ordered_ids)}


def uniform_weights(client_ids: list[str]) -> dict[str, float]:
    if not client_ids:
        raise ValueError("client_ids must not be empty")
    weight = 1.0 / len(client_ids)
    return {client_id: weight for client_id in client_ids}


def build_weight_report(
    raw_utilities: dict[str, float],
    normalized_utilities: dict[str, float],
    weights: dict[str, float],
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for client_id in raw_utilities:
        report[client_id] = {
            "raw_utility": float(raw_utilities[client_id]),
            "normalized_utility": float(normalized_utilities[client_id]),
            "aggregation_weight": float(weights[client_id]),
        }
    return report


def sanitize_for_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: sanitize_for_json(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [sanitize_for_json(value) for value in payload]
    if isinstance(payload, float):
        if math.isnan(payload) or math.isinf(payload):
            return None
    return payload