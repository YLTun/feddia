# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hand-authored layout for the living-space environment.

Same shape as :mod:`envs.factory_layout` -- the environment reads ARENA_SIZE,
SPAWN_REGIONS, MIN_SPAWN_SEPARATION and PROPS from whichever module its config
names, so a second scene needs only a second module like this one.
"""

from __future__ import annotations

from .scene_assets import (  # noqa: F401
    ARCHVIS,
    ISAAC_NUCLEUS_DIR,
    PROPS_DIR,
    WAREHOUSE,
)

# --------------------------------------------------------------------------
# Arena
# --------------------------------------------------------------------------
# 5 m, matching the arena the policies trained in. episode_length_s is
# 8 * ARENA_SIZE (40 s / 1000 steps), and the goal-distance observation is
# normalised by the same 7.07 m diagonal. See envs/factory_layout.py for why
# this is not a free choice.
ARENA_SIZE = 5.0

# --------------------------------------------------------------------------
# Spawn regions
# --------------------------------------------------------------------------
# (name, x0, x1, y0, y1). Symmetric endpoints: each episode draws two distinct
# regions, robot in one and goal in the other.
#
# Outer edges stay 0.20 m clear of the wall inner face (2.50 - 0.10 = 2.40): a
# robot spawning within collision_threshold (0.12 m) of a surface is already
# colliding, so the episode ends on step one and it teleports.
SPAWN_REGIONS: list[tuple[str, float, float, float, float]] = [
# (name, x0, x1, y0, y1) -- robot in one, goal in another
    ("E", 1.862, 2.222, -1.863, 2.233),
    ("N", -0.324, 1.519, 1.615, 2.275),
    ("S", -1.474, 1.262, -2.225, -1.535),
    ("W", -2.231, -1.721, -1.842, 0.462),
]

# Reject a sampled pair whose region centres are closer than this, so an
# unlucky draw cannot produce a near-trivial episode.
MIN_SPAWN_SEPARATION = 2.5

# --------------------------------------------------------------------------
# Props
# --------------------------------------------------------------------------
# (usd_path, position, quaternion_wxyz, scale, footprint_radius)
#
# ArchVis assets are authored in centimetres; a scale of 0.005 places the
# furniture at about half life size.
PROPS: list[
    tuple[str, tuple[float, float, float], tuple[float, float, float, float],
          tuple[float, float, float], float]
] = [
    # (usd_path, position, quaternion_wxyz, scale, footprint_radius)
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/MediaTables/Manchester.usd", (-2.219, 1.1771, 0.0), (0.707107, 0.0, 0.0, -0.707107), (0.005, 0.005, 0.005), 0.483),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/Props/SM_WaterDispenser_01a.usd", (-2.1092, -2.0914, 0.0), (0.707107, 0.0, 0.0, 0.707107), (0.7, 0.7, 0.7), 0.14),
    (f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/Props/SM_WaterDispenser_01a.usd", (-1.752, -2.0914, 0.0), (0.707107, 0.0, 0.0, 0.707107), (0.7, 0.7, 0.7), 0.14),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/FurnitureSets/Appleseed/Appleseed_Sofa.usd", (-1.3565, 0.0575, 0.0), (0.707107, 0.0, 0.0, 0.707107), (0.005, 0.005, 0.005), 0.54),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/Bookshelves/Costello.usd", (-1.3409, 2.1578, 0.0), (1.0, 0.0, 0.0, 0.0), (0.005, 0.005, 0.005), 0.875),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/FurnitureSets/Appleseed/Appleseed_Chair.usd", (-0.2796, 1.2067, 0.0), (1.0, 0.0, 0.0, 0.0), (0.005, 0.005, 0.005), 0.219),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/MediaTables/Soulard.usd", (-0.038, 0.3309, 0.0), (-0.0, 0.0, 0.0, 1.0), (0.005, 0.005, 0.005), 0.381),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/MediaTables/Soulard.usd", (-0.0337, -0.0052, 0.0), (-0.0, 0.0, 0.0, 1.0), (0.005, 0.005, 0.005), 0.381),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/FurnitureSets/Appleseed/Appleseed_Loveseat.usd", (-0.0168, -1.0246, 0.0), (0.0, 0.0, 0.0, 1.0), (0.005, 0.005, 0.005), 0.356),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/FurnitureSets/Appleseed/Appleseed_Chair.usd", (0.2113, 1.2067, 0.0), (1.0, 0.0, 0.0, 0.0), (0.005, 0.005, 0.005), 0.219),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/FurnitureSets/Appleseed/Appleseed_Loveseat.usd", (1.2851, -1.0246, 0.0), (0.0, 0.0, 0.0, 1.0), (0.005, 0.005, 0.005), 0.356),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/FurnitureSets/Appleseed/Appleseed_Loveseat.usd", (1.2853, 1.2065, 0.0), (1.0, 0.0, 0.0, 0.0), (0.005, 0.005, 0.005), 0.356),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/MediaTables/Soulard.usd", (1.3696, -0.0052, 0.0), (-0.0, 0.0, 0.0, 1.0), (0.005, 0.005, 0.005), 0.381),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/MediaTables/Soulard.usd", (1.3696, 0.3309, 0.0), (-0.0, 0.0, 0.0, 1.0), (0.005, 0.005, 0.005), 0.381),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/ArchVis/Residential/Furniture/MediaTables/Soulard.usd", (1.8776, -2.1395, 0.0), (-0.0, 0.0, 0.0, 1.0), (0.005, 0.005, 0.005), 0.381),
]
