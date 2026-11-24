#!/usr/bin/env python3
# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
FedProx-adapted TD3 for off-policy federated RL.

FedProx (Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020)
adds a proximal term to each client's local objective to limit how far local
parameters drift from the round-start global model:

    F_i(w) + (μ/2) · ‖w − w^t‖²

where w^t is the round-start global model and μ ≥ 0 is the proximal coefficient.

**Adaptation for TD3** (actor only — because only the actor is aggregated and shared):

    L_actor  = −Q(s, μ_θ(s)) + (μ/2) · ProxLoss(actor, global_actor)

where ProxLoss is the mean squared L2 distance in *parameter space*:

    ProxLoss(net, ref) = Σ_k ‖θ_k − θ^t_k‖² / N_params

Normalising by parameter count makes μ scale-independent (same μ works regardless
of network width), analogous to how FedKL uses MSE (not sum) on actor outputs.

**Why actor-only (not actor + critic)**:
In our FRL setup the default aggregation sends only actor weights to all clients.
The critic is never aggregated — each client keeps its own local critic across
rounds. There is therefore no "global critic" at θ^t to anchor to. Applying a
proximal term to the critic would compare the client's critic against its own
previous-round state (a local trust-region), which is not FedProx. Restricting
the proximal term to the actor ensures the penalty is semantically correct and
keeps the comparison with FedKL (actor-only output-space penalty) on equal footing.

**Contrast with FedKL (td3_fedkl.py)**:
    FedKL   — output space:     MSE(μ_θ(s), μ_global(s))    actor only, per batch
    FedProx — parameter space:  ‖θ − θ^t‖² / N              actor only, per update

No chunking is needed: the proximal term always references the round-start global
model θ^t (fixed), so a single continuous model.learn(local_timesteps) call is
sufficient. This also means no recent_actor equivalent is required.

References:
    FedProx: https://arxiv.org/abs/1812.06127 (MLSys 2020)
    SB3 TD3:  stable_baselines3.td3.td3.TD3
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.utils import polyak_update


def _proximal_loss(net: th.nn.Module, ref: th.nn.Module) -> th.Tensor:
    """Mean squared parameter distance between net and a frozen reference network.

    ProxLoss = Σ_k ‖θ_k − θ^t_k‖² / N_total_params

    Normalising by total parameter count keeps the loss magnitude independent of
    network width so μ is interpretable across different architectures.
    """
    total_sq: th.Tensor = th.tensor(0.0, device=next(net.parameters()).device)
    n_params = 0
    for param, ref_param in zip(net.parameters(), ref.parameters()):
        total_sq = total_sq + (param - ref_param).pow(2).sum()
        n_params += param.numel()
    return total_sq / max(n_params, 1)


class TD3FedProx(TD3):
    """TD3 subclass with FedProx proximal penalty applied to actor and critic.

    A frozen copy of the round-start global actor is maintained.
    Each delayed actor update, the proximal term (μ/2)·‖θ − θ^t‖²/N is added
    to the actor loss, constraining local actor drift in parameter space.
    The critic is never penalised (see module docstring for rationale).

    Usage per round (in frl_client_worker.py):
        1. Load global model weights via load_agent_state() as usual.
        2. Call model.set_global_reference() — snapshots actor to θ^t.
        3. model.learn(local_timesteps, reset_num_timesteps=False)  ← single call, no chunking.

    Unlike FedKL there is no update_recent_reference() step — the reference is
    fixed at θ^t for the entire round.
    """

    def __init__(
        self,
        *args: Any,
        fedprox_mu: float = 0.0,
        **kwargs: Any,
    ) -> None:
        # Set all custom attributes BEFORE super().__init__() so they exist if
        # train() is ever dispatched during _setup_model() (defensive pattern).
        self.fedprox_mu = fedprox_mu
        self.global_actor: th.nn.Module | None = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Reference management (called externally once per round)
    # ------------------------------------------------------------------

    def set_global_reference(self) -> None:
        """Snapshot the current actor into a frozen global reference.

        Call once per round immediately after loading the round-start global model.
        The reference is frozen (no gradients, eval mode) and held for the entire
        local training phase.

        Only the actor is snapshotted because only the actor is aggregated and
        shared across clients. The critic is never part of the global model in
        our default FRL setup, so there is no meaningful global critic to anchor
        to (see module docstring).
        """
        device = next(self.actor.parameters()).device

        self.global_actor = copy.deepcopy(self.actor).to(device)
        self.global_actor.set_training_mode(False)
        for param in self.global_actor.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Overridden train() with FedProx proximal penalties
    # ------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # type: ignore[override]
        """TD3 train loop with optional FedProx proximal penalty on the actor.

        Faithfully copied from stable_baselines3.td3.td3.TD3.train() with the
        following addition in the delayed actor update block:
            - actor update: actor_loss += (μ/2) · ProxLoss(actor, global_actor)

        The reference network is frozen; no gradients flow through it.

        Additional TensorBoard scalar logged:
            train/fedprox_actor_prox  — raw ProxLoss on actor params (before ×μ/2)
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        actor_prox_vals: list[float] = []

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # n-step discount support (falls back to gamma for standard replay).
            discounts = getattr(replay_data, "discounts", None)
            if discounts is None:
                discounts = self.gamma

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                    self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)

            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed actor update.
            if self._n_updates % self.policy_delay == 0:
                obs = replay_data.observations
                current_actions = self.actor(obs)
                actor_loss = -self.critic.q1_forward(obs, current_actions).mean()

                # ── FedProx actor proximal term ───────────────────────────────
                if self.global_actor is not None and self.fedprox_mu > 0:
                    actor_prox = _proximal_loss(self.actor, self.global_actor)
                    actor_loss = actor_loss + (self.fedprox_mu / 2.0) * actor_prox
                    actor_prox_vals.append(actor_prox.item())

                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.actor.parameters(), self.actor_target.parameters(), self.tau
                )
                # Copy running batch-norm stats (see SB3 GH issue #996).
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_batch_norm_stats_target,
                    1.0,
                )
                polyak_update(
                    self.actor_batch_norm_stats,
                    self.actor_batch_norm_stats_target,
                    1.0,
                )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if actor_prox_vals:
            self.logger.record("train/fedprox_actor_prox", np.mean(actor_prox_vals))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
