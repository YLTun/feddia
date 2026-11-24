# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hand-authored layout for the factory (warehouse) environment.

Read by :class:`envs.scene_env.SceneNavigationEnv`, which takes ARENA_SIZE,
SPAWN_REGIONS, MIN_SPAWN_SEPARATION and PROPS from this module.
"""

from __future__ import annotations

from .scene_assets import (  # noqa: F401
    ARCHVIS,
    ISAAC_NUCLEUS_DIR,
    PROPS_DIR,
    WAREHOUSE,
)

# --------------------------------------------------------------------------
# Prop assets
# --------------------------------------------------------------------------
# Short names for the props this scene uses most, so the PROPS table below
# reads as objects rather than as URLs. Asset roots come from scene_assets.
BARREL_A = f"{WAREHOUSE}/SM_BarelPlastic_A_01.usd"
BARREL_B = f"{WAREHOUSE}/SM_BarelPlastic_B_01.usd"
BARREL_C = f"{WAREHOUSE}/SM_BarelPlastic_C_01.usd"
CARDBOX_A = f"{WAREHOUSE}/SM_CardBoxA_01.usd"
CARDBOX_B = f"{WAREHOUSE}/SM_CardBoxB_01.usd"
CARDBOX_C = f"{WAREHOUSE}/SM_CardBoxC_01.usd"
PALLET = f"{PROPS_DIR}/Pallet/pallet.usd"
KLT_BIN = f"{PROPS_DIR}/KLT_Bin/small_KLT_visual_collision.usd"
FORKLIFT = f"{PROPS_DIR}/Forklift/forklift.usd"

# --------------------------------------------------------------------------
# Arena
# --------------------------------------------------------------------------
# Matches the arena the policies trained in, and that matters for two reasons
# beyond fitting the props:
#
#   * episode_length_s is 8 * ARENA_SIZE, so 5 m gives the same 40 s / 1000-step
#     budget as every other obstacle mode.
#   * the goal-distance observation is normalised by the arena diagonal while
#     goal_reach_threshold is raw metres, so a different arena size changes
#     what "arrived" looks like to the policy.
ARENA_SIZE = 5.0

# --------------------------------------------------------------------------
# Spawn regions
# --------------------------------------------------------------------------
# (name, x0, x1, y0, y1) in env-local metres.
#
# Regions are symmetric endpoints, not fixed start/goal roles: each episode
# picks two distinct regions at random, putting the robot in one and the goal
# in the other. Four regions -- one inset from each wall -- give both straight
# crossings (W<->E, N<->S) and diagonal ones.
#
# Both edges are bounded by collision_threshold (0.12 m): a robot spawning
# closer than that to any surface is already colliding, so the episode ends on
# step one and it teleports to a new spawn. The wall inner face sits at
# 2.50 - 0.10 = 2.40, so keep outer edges at or inside 2.20 -- the same 0.20 m
# buffer the base environment allows.
SPAWN_REGIONS: list[tuple[str, float, float, float, float]] = [
    # (name, x0, x1, y0, y1) -- robot in one, goal in another
    ("E", 1.829, 2.129, -1.106, 1.454),
    ("N", -1.25, 1.31, 1.805, 2.105),
    ("S", -0.443, 1.357, -2.107, -1.807),
    ("W", -2.184, -1.884, -1.489, 1.711),
]

# Reject a sampled pair whose region centres are closer than this, so an
# unlucky draw cannot produce a near-trivial episode. With the four regions
# above the closest pair is ~4.1 m apart, so every pair is allowed.
MIN_SPAWN_SEPARATION = 2.5

# --------------------------------------------------------------------------
# Props
# --------------------------------------------------------------------------
# (usd_path, position, quaternion_wxyz, scale, footprint_radius)
#
# Positions are env-local metres. A full quaternion and per-prop scale, rather
# than a yaw angle, are what the USD spawner takes directly.
#
# The radius is an approximate footprint, kept for reference; physics and the
# LiDAR use the asset's real collision geometry.
PROPS: list[
    tuple[str, tuple[float, float, float], tuple[float, float, float, float],
          tuple[float, float, float], float]
] = [
    # (usd_path, position, quaternion_wxyz, scale, footprint_radius)
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/Props/SM_WaterDispenser_01a.usd", (-2.1164, 2.0648, -0.0), (-0.707107, 0.0, 0.0, 0.707107), (0.7, 0.7, 0.7), 0.14),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/Props/SM_WaterDispenser_01a.usd", (-1.5474, 2.0648, -0.0), (-0.707107, 0.0, 0.0, 0.707107), (0.7, 0.7, 0.7), 0.14),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd", (-1.5418, 0.0678, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.177),
    (f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/o3dyn_pallet.usd", (-1.5011, -0.7784, -0.0), (0.707107, 0.0, 0.0, 0.707107), (0.5, 0.5, 0.5), 0.301),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd", (-1.4856, 1.2236, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.182),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd", (-1.4449, 0.4152, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.177),
    (FORKLIFT, (-1.3953, -1.84, -0.0), (-0.707107, 0.0, 0.0, 0.707107), (0.5, 0.5, 0.5), 0.874),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd", (-0.9189, 0.1841, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.177),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxB_02.usd", (-0.8739, 1.0213, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.125),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd", (-0.7829, -0.9473, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.177),
    (CARDBOX_A, (-0.6901, 1.3298, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.175),
    (CARDBOX_C, (-0.5742, 1.0051, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.125),
    (CARDBOX_C, (-0.2711, -0.4355, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.125),
    (BARREL_A, (-0.1357, 0.2376, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.177),
    (CARDBOX_A, (-0.0564, -1.2383, -0.0), (0.833886, 0.0, 0.0, 0.551937), (0.5, 0.5, 0.5), 0.21),
    (BARREL_A, (0.2026, 0.3387, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.177),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxA_02.usd", (0.272, -1.4354, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.199),
    (CARDBOX_A, (0.3993, -0.5219, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.175),
    (f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/o3dyn_pallet.usd", (0.415, 1.2879, -0.0), (0.948324, 0.0, 0.0, 0.317305), (0.5, 0.5, 0.5), 0.361),
    (CARDBOX_B, (0.7509, -0.6464, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.125),
    (BARREL_A, (1.1409, 0.38, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.177),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_03.usd", (1.1769, -1.4376, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.15),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_03.usd", (1.388, -1.1729, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.15),
    (BARREL_A, (1.3952, 0.1785, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.177),
    (BARREL_A, (1.4436, -0.4848, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.177),
    (BARREL_A, (1.453, 0.5855, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.177),
    (BARREL_A, (1.5531, 1.2883, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.177),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_03.usd", (1.6373, -2.038, 0.0), (1.0, 0.0, 0.0, 0.0), (0.6, 0.6, 0.6), 0.18),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd", (1.6801, 1.8991, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.182),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_03.usd", (2.0534, -2.0218, 0.0), (1.0, 0.0, 0.0, 0.0), (0.6, 0.6, 0.6), 0.18),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_03.usd", (2.0556, -1.6085, 0.0), (1.0, 0.0, 0.0, 0.0), (0.6, 0.6, 0.6), 0.18),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd", (2.0697, 1.7081, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.182),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd", (2.0697, 2.1187, -0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.182),
]
