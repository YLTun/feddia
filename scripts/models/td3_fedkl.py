#!/usr/bin/env python3
# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
FedKL-adapted TD3 for deterministic off-policy federated RL.

FedKL (Xie & Song, IEEE JSAC 2023) adds two KL-divergence penalties to the local
training objective to constrain how far the local policy drifts from:
  (a) the round-start global model (global/anchor penalty), and
  (b) the policy at the start of the current local iteration (local/recent penalty).

Since TD3 uses a deterministic actor, KL divergence is undefined (Dirac delta).
This module replaces KL with MSE on actor outputs — the L2 distance in action space:

    L_actor = -Q(s, μ_θ(s))
            + λ1 · MSE(μ_θ(s), μ_global(s))   ← global penalty (anchor)
            + λ2 · MSE(μ_θ(s), μ_recent(s))   ← local penalty (recent snapshot)

Local iterations are simulated by splitting 300k local timesteps into N chunks
(default: 6 × 50k) and calling update_recent_actor() between chunks. This makes
μ_recent diverge from μ_global after the first chunk, matching FedKL's structure
where the two references are identical in iter 1 but diverge in iter 2+.

References:
    FedKL: https://arxiv.org/abs/2204.10442 (IEEE JSAC 2023)
    SB3 TD3: stable_baselines3.td3.td3.TD3
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.utils import polyak_update


class TD3FedKL(TD3):
    """TD3 subclass with FedKL-style actor penalties adapted for deterministic policies.

    Two frozen reference networks are maintained:
        global_actor  — set once per round from the received global model (θ^t).
                        Never updated during local training.
        recent_actor  — starts equal to global_actor at round start.
                        Updated between training chunks via update_recent_actor().
                        After chunk 1, it tracks the locally-trained policy, making
                        the local penalty genuinely distinct from the global penalty.

    Usage per round (in frl_client_worker.py):
        1. Load global model weights via load_agent_state() as usual.
        2. Call model.set_fedkl_references() — snapshots both references to θ^t.
        3. For each chunk:
               model.learn(chunk_size, reset_num_timesteps=False)
               model.update_recent_actor()
    """

    def __init__(
        self,
        *args: Any,
        fedkl_lambda1: float = 0.0,
        fedkl_lambda2: float = 0.0,
        **kwargs: Any,
    ) -> None:
        # Set all custom attributes before super().__init__() so they exist if
        # train() is ever dispatched during _setup_model() (shouldn't happen, but
        # defensive — AttributeError inside an overridden train() is hard to debug).
        self.fedkl_lambda1 = fedkl_lambda1
        self.fedkl_lambda2 = fedkl_lambda2
        # Reference networks — populated by set_fedkl_references().
        self.global_actor: th.nn.Module | None = None
        self.recent_actor: th.nn.Module | None = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Reference management (called externally per round / per chunk)
    # ------------------------------------------------------------------

    def set_fedkl_references(self) -> None:
        """Snapshot current actor into both global_actor and recent_actor.

        Call once per round immediately after loading the global model weights.
        Both references start at θ^t (identical), simulating FedKL iteration 0
        where old == anchor == θ^t.
        """
        device = next(self.actor.parameters()).device

        self.global_actor = copy.deepcopy(self.actor).to(device)
        self.global_actor.set_training_mode(False)
        for param in self.global_actor.parameters():
            param.requires_grad_(False)

        self.recent_actor = copy.deepcopy(self.actor).to(device)
        self.recent_actor.set_training_mode(False)
        for param in self.recent_actor.parameters():
            param.requires_grad_(False)

    def update_recent_actor(self) -> None:
        """Sync recent_actor weights to the current (trained) actor.

        Call between training chunks to simulate FedKL's per-iteration 'old' policy
        snapshot. After chunk k, recent_actor = policy trained through k chunks,
        not the global model — making the two penalties genuinely distinct from chunk 2 on.
        """
        if self.recent_actor is not None:
            self.recent_actor.load_state_dict(self.actor.state_dict())

    # ------------------------------------------------------------------
    # Overridden train() with FedKL actor penalties
    # ------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # type: ignore[override]
        """TD3 train loop with optional FedKL MSE penalties on the actor loss.

        Faithfully copied from stable_baselines3.td3.td3.TD3.train() with the
        following additions in the delayed actor update block:
            - global penalty: λ1 · MSE(current_actions, global_actor(obs))
            - local penalty:  λ2 · MSE(current_actions, recent_actor(obs))
        Both reference networks are frozen (no gradients flow through them).

        Additional TensorBoard scalars logged:
            train/fedkl_global_mse  — raw MSE to global reference (before × λ1)
            train/fedkl_recent_mse  — raw MSE to recent reference (before × λ2)
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        global_mse_vals: list[float] = []
        recent_mse_vals: list[float] = []

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

                # ── FedKL global penalty ──────────────────────────────────────
                if self.global_actor is not None and self.fedkl_lambda1 > 0:
                    with th.no_grad():
                        global_actions = self.global_actor(obs)
                    global_mse = F.mse_loss(current_actions, global_actions)
                    actor_loss = actor_loss + self.fedkl_lambda1 * global_mse
                    global_mse_vals.append(global_mse.item())

                # ── FedKL local / recent penalty ──────────────────────────────
                if self.recent_actor is not None and self.fedkl_lambda2 > 0:
                    with th.no_grad():
                        recent_actions = self.recent_actor(obs)
                    recent_mse = F.mse_loss(current_actions, recent_actions)
                    actor_loss = actor_loss + self.fedkl_lambda2 * recent_mse
                    recent_mse_vals.append(recent_mse.item())

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
        if global_mse_vals:
            self.logger.record("train/fedkl_global_mse", np.mean(global_mse_vals))
        if recent_mse_vals:
            self.logger.record("train/fedkl_recent_mse", np.mean(recent_mse_vals))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
