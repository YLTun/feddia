# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Asset-server roots shared by every hand-authored scene.

Constants only. Each scene module (``factory_layout``, ``living_layout``, ...)
owns its own arena, spawn regions and props, and depends on this module rather
than on another scene -- so removing or editing one scene can never disturb
another.

Two libraries are in play:

* ``ISAAC_NUCLEUS_DIR`` -- the Isaac Sim asset set. Flat folders, robotics
  oriented: warehouse props, pallets, bins.
* ``NVIDIA_NUCLEUS_DIR`` -- the ArchVis library, a sibling of Isaac on the same
  server. Architectural visualisation: furniture, lighting, racks, plants. The
  Content browser cannot list it (it shows only registered mounts), but assets
  resolve normally by URL.

ArchVis nests by sub-type *and* design family, e.g.
``Furniture/FurnitureSets/Appleseed/Appleseed_Sofa.usd`` and
``Lighting/Floor Lamps/ArcFloorLamp.usd`` (note the space). Guessing a flat
path silently fails to spawn.
"""

from __future__ import annotations

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

#: Isaac warehouse props (barrels, card boxes, buckets).
WAREHOUSE = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props"

#: Isaac general props (pallets, KLT bins, forklift, packing table).
PROPS_DIR = f"{ISAAC_NUCLEUS_DIR}/Props"

#: NVIDIA ArchVis: Residential, Commercial and Industrial furniture.
ARCHVIS = f"{NVIDIA_NUCLEUS_DIR}/Assets/ArchVis"

# --------------------------------------------------------------------------
# Choosing props -- applies to every scene
# --------------------------------------------------------------------------
# Prefer solid shapes over anything with legs or openings at robot height. A
# prop whose collider does not match its visible shape produces an invisible
# barrier: physics blocks the robot in a gap the LiDAR reports as open, so the
# episode neither ends nor progresses. SceneNavigationEnv repairs the common
# case (colliders derived from visual meshes, see _use_convex_decomposition)
# but cannot repair every asset.
#
# The LiDAR sits ~0.11 m above the floor. Anything shorter than that -- bolts,
# books, cutlery, wall art -- is invisible to the robot and pointless as an
# obstacle, however good it looks.
#
# PROPS ONLY, NEVER SCENES. ArchVis mixes single objects with assembled rooms
# and the names do not reliably distinguish them. Anything like
# `Appleseed_LivingRoom_Var01` carries its own lighting, floor and boundary,
# which fight the arena's, and spans most of a 5 m room on its own (footprint
# radius 1.90 m when extracted). Compose from the individual pieces instead.
