# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Navigation in a hand-authored scene.

One environment class serving every hand-authored layout -- the factory and the
living space today -- each a fixed arrangement of USD props in the same 5 x 5 m
arena as every other obstacle mode. The scene is a *config field*
(``layout_module``), not a subclass, so adding one is a new layout module and a
config, with no new environment code.

Design notes
------------
* The layout is **fixed** and hand-authored: prop poses live in the layout
  module (:mod:`envs.factory_layout`, :mod:`envs.living_layout`) rather than in
  a saved USD, so they stay readable, diffable and tunable.
* Props are **static scenery**, spawned once before the environments are cloned
  and never moved. ``obstacle_mode`` is therefore ``"none"`` -- the per-episode
  obstacle machinery of :class:`NavigationEnv` is unused.
* Reward, observation and termination logic are inherited unchanged, so results
  here are directly comparable with the ``empty`` / ``random`` / ``grid`` modes.
* The arena size is **not** a free choice. ``episode_length_s`` derives from it,
  and the goal-distance observation is normalised by the arena diagonal while
  the goal and collision thresholds are raw metres.

Prop height matters
-------------------
The LiDAR sits ~0.11 m above the ground (base_link at z=0.04 plus a 0.07 joint
offset). A bare pallet is only ~0.14 m tall and is mostly fork openings at that
height, so rays pass straight through it. The sensed obstacles are therefore
barrels and boxes, which are solid at ray height. Verify prop scale visually in
the GUI before trusting a training run.
"""

from __future__ import annotations

import importlib
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from . import factory_layout, living_layout
from .navigation_env import NavigationEnv, NavigationEnvCfg, _yaw_to_quat


@configclass
class SceneNavigationEnvCfg(NavigationEnvCfg):
    """Everything a hand-authored scene shares, whichever scene it is.

    The per-scene configs below differ only in which layout module they name
    and the arena that module was authored at. Nothing here, and nothing in
    :class:`SceneNavigationEnv`, knows what is in the scene.
    """

    # Import path of the module supplying ARENA_SIZE, SPAWN_REGIONS,
    # MIN_SPAWN_SEPARATION and PROPS. A string rather than the module itself:
    # @configclass deep-copies every field default, and modules cannot be
    # deep-copied.
    #
    # Empty, not a default scene: this base is not usable on its own, and a
    # forgotten override should say so rather than silently build the factory.
    # Deliberately "" rather than dataclasses.MISSING -- MISSING means "no
    # default" to the dataclass machinery, which would make this a required
    # argument declared after inherited fields that have defaults, and that is
    # a TypeError at class-definition time.
    layout_module: str = ""

    # Props are static scenery, not per-episode obstacles.
    obstacle_mode: str = "none"

    # Uniform scale applied to every prop. Adjust if assets look wrong in the
    # viewport; per-prop scaling belongs in the layout table.
    prop_scale: float = 1.0

    # Spawned USD meshes default to a *convex hull* collider, which fills in
    # every opening: the convex hull of a table is a solid box, so the robot is
    # stopped between the legs by geometry the LiDAR cannot see (it raycasts the
    # real mesh, whose rays pass through the gap). The result is an invisible
    # wall that blocks movement without ending the episode. A convex
    # decomposition follows the true shape, so what stops the robot is also what
    # the sensor reports. Set False to keep the cheaper hull.
    prop_convex_decomposition: bool = True


# Every scene config repeats the same four lines, and that repetition is the
# point: arena_size, episode_length_s and env_spacing all derive from the size
# its layout was authored at, and reading them together is what makes a
# mismatch visible.
#
#   * `episode_length_s = 8 * arena_size` keeps the travel budget proportional:
#     at 5 m that is 40 s / 1000 steps, 8.8 m of travel against a 7.07 m
#     diagonal -- the same 1.24x margin as every other mode. Change one without
#     the other and the goal drifts out of reach.
#   * `env_spacing` must exceed arena_size, or neighbouring arenas overlap and
#     a robot near a wall senses (and collides with) the next environment's
#     props. Invisible in single-env testing; only bites during training.


@configclass
class FactoryNavigationEnvCfg(SceneNavigationEnvCfg):
    """Fixed warehouse layout, authored in :mod:`envs.factory_layout`."""

    layout_module: str = "envs.factory_layout"
    arena_size: float = factory_layout.ARENA_SIZE
    episode_length_s: float = 8.0 * factory_layout.ARENA_SIZE
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=factory_layout.ARENA_SIZE + 2.0,
        replicate_physics=True,
    )


@configclass
class LivingNavigationEnvCfg(SceneNavigationEnvCfg):
    """Hand-authored living space, from :mod:`envs.living_layout`.

    Identical machinery to the factory: only the layout differs, so the two
    scenes stay comparable with each other and with the toy modes.
    """

    layout_module: str = "envs.living_layout"
    arena_size: float = living_layout.ARENA_SIZE
    episode_length_s: float = 8.0 * living_layout.ARENA_SIZE
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=living_layout.ARENA_SIZE + 2.0,
        replicate_physics=True,
    )


class SceneNavigationEnv(NavigationEnv):
    """Navigation in a fixed, hand-authored layout.

    Only scene construction and reset placement differ from
    :class:`NavigationEnv`; rewards, observations and dones are inherited, so
    results stay comparable with the procedural modes.
    """

    cfg: SceneNavigationEnvCfg

    def __init__(self, cfg: SceneNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        # Static props must be a raycast target, registered before the sensor is
        # built in _setup_scene. Skipped when the layout is empty: a target whose
        # prim_expr matches nothing crashes the raycaster, which is why the base
        # class prunes its own Obstacle target when num_obstacles == 0.
        # Resolved and stored *before* super().__init__, for two reasons: the
        # base class has not run yet so `self.cfg` does not exist, and its
        # __init__ calls _setup_scene() -> _spawn_extra_props(), which reads
        # self._layout. Assigning it afterwards is too late.
        if not isinstance(cfg.layout_module, str) or not cfg.layout_module:
            raise ValueError(
                "SceneNavigationEnvCfg.layout_module is unset. Use one of the "
                "per-scene configs (FactoryNavigationEnvCfg, "
                "LivingNavigationEnvCfg) rather than the base class."
            )
        self._layout = importlib.import_module(cfg.layout_module)
        if self._layout.PROPS:
            cfg.lidar.mesh_prim_paths = list(cfg.lidar.mesh_prim_paths) + [
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="/World/envs/env_.*/Prop_.*",
                    track_mesh_transforms=False,  # static, never moved
                )
            ]
        super().__init__(cfg, render_mode, **kwargs)
        # Needs self.device, which only exists once the base class has run.
        self._build_region_tables()

    def _build_region_tables(self) -> None:
        """Precompute spawn-region bounds and the legal region pairs.

        Regions are symmetric endpoints: each episode draws one pair, putting
        the robot in the first and the goal in the second. Pairs whose centres
        are closer than MIN_SPAWN_SEPARATION are excluded up front, so an
        unlucky draw cannot produce a near-trivial episode.
        """
        regions = self._layout.SPAWN_REGIONS
        if len(regions) < 2:
            raise ValueError(
                f"{self._layout.__name__}.SPAWN_REGIONS needs at least 2 regions, "
                f"got {len(regions)}"
            )

        self._region_bounds = torch.tensor(
            [[x0, x1, y0, y1] for _n, x0, x1, y0, y1 in regions],
            dtype=torch.float32,
            device=self.device,
        )

        centres = [
            ((x0 + x1) / 2.0, (y0 + y1) / 2.0) for _n, x0, x1, y0, y1 in regions
        ]
        min_sep = self._layout.MIN_SPAWN_SEPARATION
        pairs = [
            (a, b)
            for a in range(len(regions))
            for b in range(len(regions))
            if a != b
            and math.dist(centres[a], centres[b]) >= min_sep
        ]
        if not pairs:
            raise ValueError(
                "no spawn-region pair satisfies "
                f"MIN_SPAWN_SEPARATION={min_sep}; regions are too close together"
            )
        self._region_pairs = torch.tensor(pairs, dtype=torch.long, device=self.device)

    # -----------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------

    def _spawn_extra_props(self) -> None:
        """Spawn the authored layout.

        Called before ``clone_environments``, so this runs once and Isaac
        replicates the props into every env.
        """
        for i, (usd, pos, quat, scale, _radius) in enumerate(self._layout.PROPS):
            spawn = sim_utils.UsdFileCfg(
                usd_path=usd,
                # Per-prop scale from the layout, times a global factor that is
                # only there for debugging odd-looking assets.
                scale=tuple(s * self.cfg.prop_scale for s in scale),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            spawn.func(
                f"/World/envs/env_.*/Prop_{i}",
                spawn,
                translation=tuple(pos),
                orientation=tuple(quat),
            )
            if self.cfg.prop_convex_decomposition:
                # Spawning happens on env_0 before cloning, so fixing the
                # collider here propagates to every environment.
                self._use_convex_decomposition(f"/World/envs/env_0/Prop_{i}")

    # Approximations that fill in a prop's openings. Anything else the asset
    # already declares (an exact "none" mesh, or a decomposition someone
    # authored deliberately) is left alone.
    _BAD_APPROXIMATIONS = frozenset({"boundingCube", "boundingSphere", "convexHull"})

    def _use_convex_decomposition(self, prim_path: str) -> None:
        """Give every mesh under ``prim_path`` a collider that follows its shape.

        This fixes props whose collider is derived from their visual meshes --
        the forklift and pallet among them. It does **not** fix every asset:
        geometry hidden inside instanceable prims is a read-only view of a
        shared prototype, and some assets ship dedicated collision proxies that
        are not meshes at all. The packing table is both, and no approximation
        setting reaches it; the fix there is to use a different prop. Prefer
        solid shapes (barrels, boxes, crates) over anything with legs or
        openings at robot height.
        """
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            return

        collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        mesh_cfg = sim_utils.ConvexDecompositionPropertiesCfg()

        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())

            api = UsdPhysics.MeshCollisionAPI(prim)
            attr = api.GetApproximationAttr() if api else None
            current = str(attr.Get()) if attr and attr.Get() else None
            if current is not None and current not in self._BAD_APPROXIMATIONS:
                continue

            sim_utils.define_collision_properties(path, collision_cfg)
            sim_utils.define_mesh_collision_properties(path, mesh_cfg)

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Place robot and goal in two different spawn regions.

        The base implementation samples anywhere in the arena, which would drop
        the robot inside a prop. Here both are confined to the authored regions,
        which are kept clear of props by construction.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # Skip NavigationEnv._reset_idx (its placement assumes movable
        # obstacles); go to the DirectRLEnv bookkeeping underneath.
        super(NavigationEnv, self)._reset_idx(env_ids)

        n = len(env_ids)

        # One region pair per env: robot in the first, goal in the second.
        pair = self._region_pairs[
            torch.randint(0, self._region_pairs.shape[0], (n,), device=self.device)
        ]
        rb = self._region_bounds[pair[:, 0]]  # (n, 4) robot region bounds
        gb = self._region_bounds[pair[:, 1]]  # (n, 4) goal region bounds

        def _sample_in(bounds: torch.Tensor) -> torch.Tensor:
            u = sample_uniform(0.0, 1.0, (n, 2), self.device)
            return torch.stack(
                [
                    bounds[:, 0] + u[:, 0] * (bounds[:, 1] - bounds[:, 0]),
                    bounds[:, 2] + u[:, 1] * (bounds[:, 3] - bounds[:, 2]),
                ],
                dim=1,
            )

        rxy = _sample_in(rb)
        gxy = _sample_in(gb)

        # Uniform heading, matching every other obstacle mode. Facing the goal
        # would make this environment measurably easier and confound any
        # comparison against them.
        ryaw = sample_uniform(-math.pi, math.pi, (n,), self.device)

        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, 0] = rxy[:, 0] + self.scene.env_origins[env_ids, 0]
        root[:, 1] = rxy[:, 1] + self.scene.env_origins[env_ids, 1]
        root[:, 2] = 0.04 + self.scene.env_origins[env_ids, 2]
        root[:, 3:7] = _yaw_to_quat(ryaw)
        root[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)

        jp = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), None, env_ids)

        self._goal_positions[env_ids] = gxy
        self._prev_goal_dist[env_ids] = torch.norm(gxy - rxy, dim=1)
        self._prev_angular_vel[env_ids] = 0.0

        self._initial_dist[env_ids] = self._prev_goal_dist[env_ids].clone()
        self._travel_dist[env_ids] = 0.0
        self._prev_pos_for_dist[env_ids] = rxy.clone()

        goal_root = self.goal.data.default_root_state[env_ids].clone()
        goal_root[:, 0] = gxy[:, 0] + self.scene.env_origins[env_ids, 0]
        goal_root[:, 1] = gxy[:, 1] + self.scene.env_origins[env_ids, 1]
        goal_root[:, 2] = 0.1
        self.goal.write_root_pose_to_sim(goal_root[:, :7], env_ids)

        # Walls are spawned stacked at the arena centre and only moved into
        # place here; skipping this leaves them piled at the origin.
        self._place_walls(env_ids)
