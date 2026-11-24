# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Mapless Navigation Environment using DirectRLEnv.

A differential-drive robot navigates to a goal position in a cluttered
2D arena using LiDAR observations and TD3.

The robot is loaded from a local URDF file (no Nucleus dependency).
"""

from __future__ import annotations

import copy
import math
import os
import sys
import torch
from collections.abc import Sequence
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import MultiMeshRayCaster, MultiMeshRayCasterCfg, patterns
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

# Path to local URDF (relative to workspace root)
_URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "diff_drive_robot.urdf",
)


# =============================================================================
# Environment Configuration
# =============================================================================


@configclass
class NavigationEnvCfg(DirectRLEnvCfg):
    """Configuration for the mapless navigation environment."""

    # --- Simulation ---
    decimation: int = 4
    episode_length_s: float = 40.0  # 1000 steps max
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,  # 100Hz physics
        render_interval=4,
    )

    # --- Spaces ---
    observation_space: int = 64  # auto-computed in __post_init__
    action_space: int = 2  # (linear_vel, angular_vel) in [-1, 1]
    state_space: int = 0

    # --- Sensors (defaults overwritten by __post_init__ from scalar params) ---
    lidar: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/lidar_link",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-90.0, 90.0),
            horizontal_res=3.0,  # placeholder, set by __post_init__
        ),
        debug_vis=False,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Obstacle_.*",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Wall_.*",
                track_mesh_transforms=True,
            ),
        ],
        max_distance=5.0,
    )

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=6.0,
        replicate_physics=True,
    )

    # --- Robot (loaded from local URDF) ---
    robot_cfg: ArticulationCfg = ArticulationCfg(
        spawn=UrdfFileCfg(
            asset_path=_URDF_PATH,
            fix_base=False,
            merge_fixed_joints=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="velocity",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=10.0,
                ),
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.04),
            joint_pos={"wheel_left_joint": 0.0, "wheel_right_joint": 0.0},
            joint_vel={"wheel_left_joint": 0.0, "wheel_right_joint": 0.0},
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
                effort_limit_sim=100.0,
                stiffness=0.0,  # velocity control
                damping=100.0,
            ),
        },
        prim_path="/World/envs/env_.*/Robot",
    )

    # --- Robot physical params ---
    wheel_radius: float = 0.033
    wheel_separation: float = 0.158  # 2 * 0.079m offset
    max_linear_vel: float = 0.22
    max_angular_vel: float = 2.0

    # --- LiDAR ---
    num_lidar_rays: int = 120
    lidar_fov_deg: float = 360.0

    # --- Goal / Collision ---
    goal_reach_threshold: float = 0.2
    collision_threshold: float = 0.12  # ~robot diagonal radius

    def __post_init__(self):
        """Auto-compute dependent config values from scalar parameters."""
        super().__post_init__()

        if self.num_gate_palisade_hole_rows * self.num_gate_palisade_holes_per_layer >= self.num_gate_rows:
            raise ValueError(
                "num_gate_palisade_holes_per_layer * num_gate_palisade_hole_rows must be smaller than num_gate_rows"
            )
        self.num_gate_palisade_obstacles = self.num_gate_palisade_layers * (
            self.num_gate_rows - self.num_gate_palisade_hole_rows * self.num_gate_palisade_holes_per_layer
        )
        
        # Intercept requested obstacle_mode before memory allocation
        if self.obstacle_mode == "none":
            self.num_obstacles = 0
            
        # Update LiDAR sensor config from scalar params
        half_fov = self.lidar_fov_deg / 2.0
        self.lidar.pattern_cfg.horizontal_fov_range = (-half_fov, half_fov)
        self.lidar.pattern_cfg.horizontal_res = self.lidar_fov_deg / self.num_lidar_rays
        # Update observation space: LiDAR rays + 2 goal + 2 velocity
        self.observation_space = self.num_lidar_rays + 4

    # --- Environment constraints ---
    arena_size: float = 5.0
    num_obstacles: int = 20  # Number of random obstacles
    num_dense_obstacles: int = 25  # Number of densely packed obstacles for the pure dense mode
    dense_grid_half_extent: float = 1.7  # Dense grid spans [-extent, extent] in both x/y
    num_random_dense_dense_obstacles: int = 16  # Dense-grid obstacle count used in random_dense mode
    num_random_dense_random_obstacles: int = 12  # Random overlay obstacle count used in random_dense mode
    num_gate_rows: int = 13  # Number of candidate pillar slots per wall layer
    num_gate_palisade_layers: int = 3  # Number of wall layers in gate_palisade_multi mode
    num_gate_palisade_holes_per_layer: int = 2  # Number of independent openings per layer
    num_gate_palisade_hole_rows: int = 3  # Height of each opening in row slots
    num_gate_palisade_obstacles: int = 22  # Auto-updated in __post_init__
    gate_wall_thickness: float = 0.2  # X thickness of each solid wall segment
    gate_x_span: float = 2.4  # Horizontal span covered by gate columns
    gate_gap_half_height: float = 0.5  # Half of vertical opening size for each gate
    gate_jitter: float = 0.1  # Per-obstacle random jitter to avoid overfitting
    obstacle_mode: str = "random"  # "none", "random", "dense", "random_dense", "gate_palisade_multi"

    # Obstacle properties
    obstacle_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_.*",
        spawn=sim_utils.CylinderCfg(
            radius=0.15,
            height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )

    # Goal properties (kinematic rigid object so we can move it via code)
    goal_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 0.5, 0.0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )

    # Wall properties (Spawn 4 walls per env to form a boundary)
    wall_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Wall_.*",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )

    # --- Rewards ---
    reward_goal: float = 200.0
    reward_collision: float = -40.0
    reward_progress_scale: float = 10.0
    reward_step_penalty: float = -0.05
    reward_smoothness_scale: float = -0.1
    reward_forward_velocity_scale: float = (
        0.25  # Increased bonus to powerfully overcome step penalty
    )



# =============================================================================
# Environment
# =============================================================================


class NavigationEnv(DirectRLEnv):
    """Mapless navigation environment for differential-drive robot."""

    cfg: NavigationEnvCfg

    def __init__(self, cfg: NavigationEnvCfg, render_mode: str | None = None, **kwargs):
        # Determine internal obstacle counts correctly based on mode
        self._num_random_obstacles = 0
        self._num_dense_obstacles = 0
        self._num_gate_palisade_obstacles = 0

        if cfg.obstacle_mode == "none":
            pass
        elif cfg.obstacle_mode == "random":
            self._num_random_obstacles = cfg.num_obstacles
        elif cfg.obstacle_mode == "dense":
            self._num_dense_obstacles = cfg.num_dense_obstacles
        elif cfg.obstacle_mode == "random_dense":
            self._num_dense_obstacles = cfg.num_random_dense_dense_obstacles
            self._num_random_obstacles = cfg.num_random_dense_random_obstacles
        elif cfg.obstacle_mode == "gate_palisade_multi":
            self._num_gate_palisade_obstacles = cfg.num_gate_palisade_obstacles
            safe_h = cfg.arena_size / 2.0 - 0.35
            segment_len = (2.0 * safe_h) / cfg.num_gate_rows
            cfg.obstacle_cfg.spawn = sim_utils.CuboidCfg(
                size=(cfg.gate_wall_thickness, segment_len, 0.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            )
        else:
            raise ValueError(f"Unsupported obstacle_mode: {cfg.obstacle_mode}")
            
        # Override the Isaac Lab total config dynamically
        cfg.num_obstacles = (
            self._num_random_obstacles
            + self._num_dense_obstacles
            + self._num_gate_palisade_obstacles
        )

        if cfg.num_obstacles == 0:
            # If 0 obstacles, LiDAR cannot search for obstacle meshes or it will crash.
            cfg.lidar.mesh_prim_paths = [
                target for target in cfg.lidar.mesh_prim_paths
                if "Obstacle" not in target.prim_expr
            ]
            
        super().__init__(cfg, render_mode, **kwargs)

        # --- Joint indices ---
        self._left_wheel_idx, _ = self.robot.find_joints("wheel_left_joint")
        self._right_wheel_idx, _ = self.robot.find_joints("wheel_right_joint")

        # --- Buffers ---
        self._goal_positions = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_angular_vel = torch.zeros(self.num_envs, device=self.device)
        self._actions = torch.zeros(self.num_envs, 2, device=self.device)

        # --- Evaluation Buffers ---
        self._travel_dist = torch.zeros(self.num_envs, device=self.device)
        self._initial_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_pos_for_dist = torch.zeros(self.num_envs, 2, device=self.device)

        # --- Action Space definition ---
        # NOTE: We must define BOTH single_action_space and action_space for SB3 compatibility.
        # 1. single_action_space: Used by Stable Baselines 3 to properly initialize the neural network
        #    (e.g., knowing to apply a Tanh squashing layer to bound outputs to [-1, 1] for a single env).
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_space,)
        )
        # 2. action_space: Defines the explicitly batched tensor shape expected by Isaac Lab's parallel physics engine.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_envs, self.cfg.action_space)
        )

# -----------------------------------------------------------------
    # Scene
    # -----------------------------------------------------------------

    def _spawn_extra_props(self) -> None:
        """Hook for subclasses to spawn extra static prims. No-op by default."""
        return

    def _setup_scene(self):
        """Ground plane + robot + obstacles + goal marker + light."""
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground plane with friction so wheels can grip during turns
        ground_cfg = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,  # Rubber-on-concrete grip
                dynamic_friction=0.8,  # Grip even while sliding
            ),
        )
        ground_cfg.func("/World/ground", ground_cfg)

        # Obstacles (spawned in a loop, then wrapped with RigidObject)
        if self.cfg.num_obstacles > 0:
            for i in range(self.cfg.num_obstacles):
                self.cfg.obstacle_cfg.spawn.func(
                    f"/World/envs/env_.*/Obstacle_{i}", self.cfg.obstacle_cfg.spawn
                )

            # Create RigidObject handle for obstacles (disable automatic spawning)
            obs_cfg = copy.deepcopy(self.cfg.obstacle_cfg)
            obs_cfg.spawn = None
            self.obstacles = RigidObject(obs_cfg)

        # Goal marker
        self.cfg.goal_cfg.spawn.func(
            "/World/envs/env_.*/GoalMarker", self.cfg.goal_cfg.spawn
        )
        goal_cfg = copy.deepcopy(self.cfg.goal_cfg)
        goal_cfg.spawn = None
        self.goal = RigidObject(goal_cfg)

        # Boundary Walls (spawned manually, wrapped)
        # Order is alphabetical: Bottom, Left, Right, Top
        thickness = 0.2
        height = 1.0
        L = self.cfg.arena_size
        wall_params = [
            ("Bottom", (L + thickness, thickness, height)),
            ("Left", (thickness, L + thickness, height)),
            ("Right", (thickness, L + thickness, height)),
            ("Top", (L + thickness, thickness, height)),
        ]
        for name, size in wall_params:
            self.cfg.wall_cfg.spawn.size = size
            self.cfg.wall_cfg.spawn.func(
                f"/World/envs/env_.*/Wall_{name}", self.cfg.wall_cfg.spawn
            )

        wall_cfg_obs = copy.deepcopy(self.cfg.wall_cfg)
        wall_cfg_obs.spawn = None
        self.walls = RigidObject(wall_cfg_obs)

        # Light
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

        # Apply high-friction physics material to wheel collision shapes
        # (UrdfFileCfg doesn't support physics_material, so we do it via USD API)
        import omni.usd
        from pxr import UsdPhysics, UsdShade

        stage = omni.usd.get_context().get_stage()

        # Create a physics material for wheels
        mat_path = "/World/PhysicsMaterials/WheelFriction"
        UsdShade.Material.Define(stage, mat_path)
        mat_prim = stage.GetPrimAtPath(mat_path)
        mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
        mat_api.CreateStaticFrictionAttr(1.5)
        mat_api.CreateDynamicFrictionAttr(1.0)
        mat_api.CreateRestitutionAttr(0.0)

        # Bind to wheel collision shapes in env_0 (replicated to all envs by clone)
        mat = UsdShade.Material(mat_prim)
        for wheel_name in ["wheel_left_link", "wheel_right_link"]:
            wheel_path = f"/World/envs/env_0/Robot/{wheel_name}"
            wheel_prim = stage.GetPrimAtPath(wheel_path)
            if wheel_prim.IsValid():
                binding_api = UsdShade.MaterialBindingAPI.Apply(wheel_prim)
                binding_api.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")

        # Hook for subclasses to add static scenery. Must run before cloning so
        # the extra prims are replicated into every env, and before the LiDAR is
        # constructed below so its raycast targets can resolve them.
        self._spawn_extra_props()

        # Clone + register
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        if hasattr(self, "obstacles"):
            self.scene.rigid_objects["obstacles"] = self.obstacles
        self.scene.rigid_objects["goal"] = self.goal
        self.scene.rigid_objects["walls"] = self.walls

        # Instantiate Sensors AFTER cloning
        self.lidar = MultiMeshRayCaster(self.cfg.lidar)
        self.scene.sensors["lidar"] = self.lidar

    # -----------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """(v_norm, w_norm) → wheel velocities via diff-drive kinematics."""
        # Secure actions from neural network against NaN or Inf
        # A NaN action means the neural network broke. We want the robot to safely stop:
        # v = 0 requires action[:, 0] = -1.0
        # w = 0 requires action[:, 1] = 0.0
        clamped_actions = torch.nan_to_num(self._actions, posinf=1.0, neginf=-1.0)

        # If any action dimension is NaN for an environment, force it to stop completely
        nan_envs = torch.isnan(self._actions).any(dim=1)
        clamped_actions[nan_envs, 0] = -1.0  # For v
        clamped_actions[nan_envs, 1] = 0.0  # For w

        v = (clamped_actions[:, 0] + 1.0) * 0.5 * self.cfg.max_linear_vel  # [0, max]
        w = clamped_actions[:, 1] * self.cfg.max_angular_vel  # [-max, max]

        r = self.cfg.wheel_radius
        L = self.cfg.wheel_separation
        wl = (v - w * L / 2.0) / r
        wr = (v + w * L / 2.0) / r

        # Instead of directly setting targets on the Articulation,
        # we need to set them on the actuator so the PD loop actually applies effort
        targets = torch.zeros(self.num_envs, 2, device=self.device)
        targets[:, 0] = wl
        targets[:, 1] = wr
        # Apply safety clamp to velocity targets
        targets = torch.clamp(targets, -100.0, 100.0)
        self.robot.set_joint_velocity_target(
            targets, joint_ids=[self._left_wheel_idx[0], self._right_wheel_idx[0]]
        )

    # -----------------------------------------------------------------
    # Observations (124-dim: 120 LiDAR + goal distance, goal heading,
    # forward and angular velocity)
    # -----------------------------------------------------------------

    def _get_observations(self) -> dict:
        pos_w = self.robot.data.root_pos_w
        quat_w = self.robot.data.root_quat_w
        lin_vel = self.robot.data.root_lin_vel_w
        ang_vel = self.robot.data.root_ang_vel_w

        xy = pos_w[:, :2] - self.scene.env_origins[:, :2]
        yaw = _quat_to_yaw(quat_w)

        # LiDAR data
        # Calculate distance from sensor origin to hit points
        pos_w_sensor = self.lidar.data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
        hits_w = self.lidar.data.ray_hits_w  # (num_envs, num_rays, 3)
        lidar_dist = torch.norm(hits_w - pos_w_sensor, dim=-1)  # (num_envs, num_rays)

        # Normalize LiDAR to [0, 1] relative to max distance
        lidar_n = lidar_dist / self.cfg.lidar.max_distance

        # Rays that hit nothing return NaN positions, which makes lidar_dist NaN
        # We replace NaNs with 1.0 (indicating the space is clear up to max_distance)
        lidar_n = torch.nan_to_num(lidar_n, nan=1.0)

        # Ensure it has exactly cfg.num_lidar_rays dimensions
        lidar_n = lidar_n[:, : self.cfg.num_lidar_rays]

        # Goal
        delta = self._goal_positions - xy
        dist = torch.norm(delta, dim=1, keepdim=True)
        dist_n = dist / (self.cfg.arena_size * math.sqrt(2))
        angle = torch.atan2(delta[:, 1], delta[:, 0])
        heading_n = (_wrap_angle(angle - yaw) / math.pi).unsqueeze(1)

        # Velocity in robot frame
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        fwd = lin_vel[:, 0] * cos_y + lin_vel[:, 1] * sin_y
        fwd_n = (fwd / self.cfg.max_linear_vel).unsqueeze(1)
        ang_n = (ang_vel[:, 2] / self.cfg.max_angular_vel).unsqueeze(1)

        obs = torch.cat([lidar_n, dist_n, heading_n, fwd_n, ang_n], dim=1)
        # Protect neural network inputs from explosion (e.g. huge velocity from collision physics,
        # or huge distance from LiDAR misses)
        obs = torch.clamp(obs, -5.0, 5.0)
        obs = torch.nan_to_num(obs, nan=0.0)

        return {"policy": obs}

    # -----------------------------------------------------------------
    # Rewards
    # -----------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        ang_vel_z = self.robot.data.root_ang_vel_w[:, 2]
        dist = torch.norm(self._goal_positions - xy, dim=1)

        # Goal reward
        r = (dist < self.cfg.goal_reach_threshold).float() * self.cfg.reward_goal

        # Collision penalty (LiDAR-based)
        pos_w_sensor = self.lidar.data.pos_w.unsqueeze(1)
        hits_w = self.lidar.data.ray_hits_w
        lidar_dist = torch.norm(hits_w - pos_w_sensor, dim=-1)
        lidar_dist = torch.nan_to_num(lidar_dist, nan=self.cfg.lidar.max_distance)
        min_dist, _ = torch.min(lidar_dist, dim=1)
        r += (
            min_dist < self.cfg.collision_threshold
        ).float() * self.cfg.reward_collision

        # Progress, step penalty, smoothness
        r += (self._prev_goal_dist - dist) * self.cfg.reward_progress_scale
        r += self.cfg.reward_step_penalty
        r += (
            torch.abs(ang_vel_z - self._prev_angular_vel)
            * self.cfg.reward_smoothness_scale
        )

        # Forward velocity bonus (keeps the robot moving to offset step penalty)
        yaw = _quat_to_yaw(self.robot.data.root_quat_w)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        lin_vel = self.robot.data.root_lin_vel_w
        fwd_vel = lin_vel[:, 0] * cos_y + lin_vel[:, 1] * sin_y
        r += fwd_vel * self.cfg.reward_forward_velocity_scale

        self._prev_goal_dist = dist.clone()
        self._prev_angular_vel = ang_vel_z.clone()
        # Protect network from explosive rewards
        r = torch.clamp(r, -500.0, 500.0)
        r = torch.nan_to_num(r, nan=0.0)
        return r

    # -----------------------------------------------------------------
    # Dones
    # -----------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Travel Distance Tracking ---
        xy = self.robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        step_dist = torch.norm(xy - self._prev_pos_for_dist, dim=1)
        self._travel_dist += step_dist
        self._prev_pos_for_dist = xy.clone()

        # --- 1. Goal Reached ---
        dist = torch.norm(self._goal_positions - xy, dim=1)
        goal_reached = dist < self.cfg.goal_reach_threshold

        # --- 2. Collision Detected ---
        # Get raw LiDAR distances (we recalculate here or pull from sensor data)
        pos_w_sensor = self.lidar.data.pos_w.unsqueeze(1)
        hits_w = self.lidar.data.ray_hits_w
        lidar_dist = torch.norm(hits_w - pos_w_sensor, dim=-1)

        # Replace NaNs with max distance (safe space)
        lidar_dist = torch.nan_to_num(lidar_dist, nan=self.cfg.lidar.max_distance)

        # If any LiDAR ray is closer than the robot's collision radius, we've collided
        min_dist, _ = torch.min(lidar_dist, dim=1)
        collided = min_dist < self.cfg.collision_threshold

        # --- 3. Time Out ---
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # A run is terminated (failed/succeeded) if it reaches the goal OR hits an obstacle
        terminated = goal_reached | collided

        # Populate extras for evaluation script
        if not hasattr(self, "extras"):
            self.extras = {}

        self.extras["success"] = goal_reached.clone()
        self.extras["collision"] = collided.clone()
        self.extras["timeout"] = time_out.clone()
        self.extras["travel_dist"] = self._travel_dist.clone()
        self.extras["initial_dist"] = self._initial_dist.clone()
        self.extras["episode_steps"] = self.episode_length_buf.clone()

        return terminated, time_out

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)
        h = self.cfg.arena_size / 2.0

        # Per-env crossing axis and direction, shared by robot and goal placement.
        # axis=0: cross along x (left<->right), axis=1: cross along y (bottom<->top).
        # sign=-1: robot on negative side; sign=+1: robot on positive side. Goal always opposite.
        _cross_axis = torch.randint(0, 2, (n,), device=self.device)
        _cross_sign = (torch.randint(0, 2, (n,), device=self.device) * 2 - 1).float()
        _ax_x = (_cross_axis == 0)
        _ax_y = (_cross_axis == 1)
        # Inner boundary: robot/goal each spawn beyond _near from centre.
        _near = (
            min(self.cfg.dense_grid_half_extent, h * 0.8 - 0.1)
            if getattr(self, "_num_dense_obstacles", 0) > 0
            else h * 0.3
        )

        # Random robot pose
        if getattr(self, "_num_gate_palisade_obstacles", 0) > 0:
            # For gate corridor mode, start and goal on opposite sides so
            # the structured gate geometry is useful and starts are not inside gates.
            side_band = min(max(self.cfg.gate_x_span * 0.5 + 0.6, 1.0), h * 0.75)
            r_x = sample_uniform(-h * 0.8, -side_band, (n, 1), self.device)
            r_y = sample_uniform(-h * 0.5, h * 0.5, (n, 1), self.device)
            rxy = torch.cat([r_x, r_y], dim=1)
        else:
            # All non-gate modes: robot spawns in a strip on a randomly chosen side.
            # Applies uniformly to dense, random_dense, random, and none.
            r_primary = _cross_sign * sample_uniform(_near, h * 0.8, (n,), self.device)
            r_secondary = sample_uniform(-h * 0.8, h * 0.8, (n,), self.device)
            rxy = torch.zeros(n, 2, device=self.device)
            rxy[_ax_x, 0] = r_primary[_ax_x]
            rxy[_ax_x, 1] = r_secondary[_ax_x]
            rxy[_ax_y, 1] = r_primary[_ax_y]
            rxy[_ax_y, 0] = r_secondary[_ax_y]
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

        # Random goal (>=1m from robot)
        if getattr(self, "_num_gate_palisade_obstacles", 0) > 0:
            side_band = min(max(self.cfg.gate_x_span * 0.5 + 0.6, 1.0), h * 0.75)
            g_x = sample_uniform(side_band, h * 0.8, (n, 1), self.device)
            g_y = sample_uniform(-h * 0.5, h * 0.5, (n, 1), self.device)
            gxy = torch.cat([g_x, g_y], dim=1)
        else:
            # Goal always on the opposite side from the robot.
            g_primary = -_cross_sign * sample_uniform(_near, h * 0.8, (n,), self.device)
            g_secondary = sample_uniform(-h * 0.8, h * 0.8, (n,), self.device)
            gxy = torch.zeros(n, 2, device=self.device)
            gxy[_ax_x, 0] = g_primary[_ax_x]
            gxy[_ax_x, 1] = g_secondary[_ax_x]
            gxy[_ax_y, 1] = g_primary[_ax_y]
            gxy[_ax_y, 0] = g_secondary[_ax_y]
        too_close = torch.norm(gxy - rxy, dim=1) < 1.0
        if too_close.any():
            gxy[too_close] = -rxy[too_close] * 0.8

        self._goal_positions[env_ids] = gxy
        self._prev_goal_dist[env_ids] = torch.norm(gxy - rxy, dim=1)
        self._prev_angular_vel[env_ids] = 0.0

        # Reset evaluation buffers
        self._initial_dist[env_ids] = self._prev_goal_dist[env_ids].clone()
        self._travel_dist[env_ids] = 0.0
        self._prev_pos_for_dist[env_ids] = rxy.clone()

        # Update Goal marker physical position
        goal_root = self.goal.data.default_root_state[env_ids].clone()
        goal_root[:, 0] = gxy[:, 0] + self.scene.env_origins[env_ids, 0]
        goal_root[:, 1] = gxy[:, 1] + self.scene.env_origins[env_ids, 1]
        goal_root[:, 2] = 0.1
        self.goal.write_root_pose_to_sim(goal_root[:, :7], env_ids)

        # -----------------------------------------------------------------
        # Obstacle Generation Modes
        # -----------------------------------------------------------------
        obs_indices = []
        for env_id in env_ids.tolist():
            start_idx = env_id * self.cfg.num_obstacles
            obs_indices.extend(range(start_idx, start_idx + self.cfg.num_obstacles))

        if len(obs_indices) > 0 and hasattr(self, "obstacles"):
            obs_env_ids = torch.tensor(
                obs_indices, dtype=torch.long, device=self.device
            )
            obs_root = self.obstacles.data.default_root_state[obs_env_ids].clone()
            num_obs_spawning = len(obs_indices)
            # --- 1. Random Obstacles ---
            if self._num_random_obstacles > 0:
                # Calculate exactly how many random obstacles we are spawning
                n_random_spawning = len(env_ids) * self._num_random_obstacles
                
                safe_h = h - 0.35
                oxy = sample_uniform(-safe_h, safe_h, (n_random_spawning, 2), self.device)

                # We map the random objects precisely to their environment offsets
                obs_rxy = torch.repeat_interleave(rxy, self._num_random_obstacles, dim=0)
                obs_gxy = torch.repeat_interleave(gxy, self._num_random_obstacles, dim=0)
                repeated_env_ids = torch.repeat_interleave(env_ids, self._num_random_obstacles)

                # Push away from robot and goal — keep clearance just large enough to
                # prevent spawning inside the robot/goal footprint, without clearing the path.
                _obs_clearance = 0.4
                dist_to_robot = torch.norm(oxy - obs_rxy, dim=1)
                too_close_r = dist_to_robot < _obs_clearance
                if too_close_r.any():
                    dir_r = oxy[too_close_r] - obs_rxy[too_close_r] + 1e-5
                    dir_r = dir_r / torch.norm(dir_r, dim=1, keepdim=True)
                    oxy[too_close_r] = obs_rxy[too_close_r] + dir_r * _obs_clearance

                # Push away from goal
                dist_to_goal = torch.norm(oxy - obs_gxy, dim=1)
                too_close_g = dist_to_goal < _obs_clearance
                if too_close_g.any():
                    dir_g = oxy[too_close_g] - obs_gxy[too_close_g] + 1e-5
                    dir_g = dir_g / torch.norm(dir_g, dim=1, keepdim=True)
                    oxy[too_close_g] = obs_gxy[too_close_g] + dir_g * _obs_clearance

                # Final clamp to keep within bounds
                oxy = torch.clamp(oxy, -safe_h, safe_h)

                # Calculate the exact indices in the 1D obs_root array for these random obstacles
                random_indices = []
                for i in range(len(env_ids)):
                    base_idx = i * self.cfg.num_obstacles
                    start_rand = base_idx
                    end_rand = start_rand + self._num_random_obstacles
                    random_indices.extend(range(start_rand, end_rand))
                
                random_indices_tensor = torch.tensor(random_indices, dtype=torch.long, device=self.device)

                obs_root[random_indices_tensor, 0] = oxy[:, 0] + self.scene.env_origins[repeated_env_ids, 0]
                obs_root[random_indices_tensor, 1] = oxy[:, 1] + self.scene.env_origins[repeated_env_ids, 1]
                obs_root[random_indices_tensor, 2] = 0.25

            # --- 3. Dense Forest Obstacles (Static Grid) ---
            if getattr(self, "_num_dense_obstacles", 0) > 0:
                n_dense_spawning = len(env_ids) * self._num_dense_obstacles
                
                # Build a fixed rectangular grid in the center of the environment.
                # Grid side is computed dynamically so any num_dense_obstacles is supported.
                _grid_side = math.ceil(math.sqrt(self._num_dense_obstacles))
                dense_half_extent = min(self.cfg.dense_grid_half_extent, h - 0.35)
                grid_xs = torch.linspace(-dense_half_extent, dense_half_extent, _grid_side, device=self.device)
                grid_ys = torch.linspace(-dense_half_extent, dense_half_extent, _grid_side, device=self.device)
                grid_x, grid_y = torch.meshgrid(grid_xs, grid_ys, indexing='ij')
                
                fixed_oxy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
                
                # Cap to the exact number of dense obstacles requested
                fixed_oxy = fixed_oxy[:self._num_dense_obstacles]
                
                # Repeat exact same static grid for every active environment
                oxy = fixed_oxy.repeat(len(env_ids), 1)
                
                repeated_env_ids = torch.repeat_interleave(env_ids, self._num_dense_obstacles)

                # Minimal push-away: just enough to prevent spawn-inside-obstacle collision.
                # 0.35m = obstacle_radius(0.15) + collision_threshold(0.12) + buffer(0.08).
                # Grid spacing is 0.85 m so this nudge preserves the grid structure.
                _min_clearance = 0.35
                obs_rxy = torch.repeat_interleave(rxy, self._num_dense_obstacles, dim=0)
                too_close_r = torch.norm(oxy - obs_rxy, dim=1) < _min_clearance
                if too_close_r.any():
                    dir_r = oxy[too_close_r] - obs_rxy[too_close_r] + 1e-5
                    dir_r = dir_r / torch.norm(dir_r, dim=1, keepdim=True)
                    oxy[too_close_r] = obs_rxy[too_close_r] + dir_r * _min_clearance

                obs_gxy = torch.repeat_interleave(gxy, self._num_dense_obstacles, dim=0)
                too_close_g = torch.norm(oxy - obs_gxy, dim=1) < _min_clearance
                if too_close_g.any():
                    dir_g = oxy[too_close_g] - obs_gxy[too_close_g] + 1e-5
                    dir_g = dir_g / torch.norm(dir_g, dim=1, keepdim=True)
                    oxy[too_close_g] = obs_gxy[too_close_g] + dir_g * _min_clearance

                dense_indices = []
                for i in range(len(env_ids)):
                    base_idx = i * self.cfg.num_obstacles
                    start_dense = base_idx + self._num_random_obstacles
                    dense_indices.extend(range(start_dense, start_dense + self._num_dense_obstacles))
                
                dense_indices_tensor = torch.tensor(dense_indices, dtype=torch.long, device=self.device)

                obs_root[dense_indices_tensor, 0] = oxy[:, 0] + self.scene.env_origins[repeated_env_ids, 0]
                obs_root[dense_indices_tensor, 1] = oxy[:, 1] + self.scene.env_origins[repeated_env_ids, 1]
                obs_root[dense_indices_tensor, 2] = 0.25

            # --- 4. Multi-hole Solid Palisade Obstacles ---
            if getattr(self, "_num_gate_palisade_obstacles", 0) > 0:
                n_envs = len(env_ids)
                layers = self.cfg.num_gate_palisade_layers
                rows = self.cfg.num_gate_rows
                holes_per_layer = self.cfg.num_gate_palisade_holes_per_layer
                hole_rows = self.cfg.num_gate_palisade_hole_rows
                rows_kept = rows - holes_per_layer * hole_rows
                safe_h = h - 0.35

                if layers == 1:
                    x_positions = torch.tensor([0.0], device=self.device)
                else:
                    x_positions = torch.linspace(
                        -self.cfg.gate_x_span * 0.5,
                        self.cfg.gate_x_span * 0.5,
                        layers,
                        device=self.device,
                    )

                # Tile centers are exactly packed so closed segments touch with no seam.
                y_slots = torch.linspace(
                    -safe_h + (safe_h * 2.0) / (2.0 * rows),
                    safe_h - (safe_h * 2.0) / (2.0 * rows),
                    rows,
                    device=self.device,
                )
                oxy = torch.zeros((n_envs, layers, rows_kept, 2), device=self.device)

                for env_i in range(n_envs):
                    for layer_i in range(layers):
                        selected_starts: list[int] = []
                        candidate_starts = list(range(rows - hole_rows + 1))

                        while candidate_starts and len(selected_starts) < holes_per_layer:
                            pick_i = torch.randint(
                                low=0,
                                high=len(candidate_starts),
                                size=(1,),
                                device=self.device,
                            ).item()
                            start = candidate_starts.pop(pick_i)
                            overlaps_existing = any(
                                not (start + hole_rows <= s or s + hole_rows <= start)
                                for s in selected_starts
                            )
                            if not overlaps_existing:
                                selected_starts.append(start)

                        if len(selected_starts) < holes_per_layer:
                            for start in range(rows - hole_rows + 1):
                                overlaps_existing = any(
                                    not (start + hole_rows <= s or s + hole_rows <= start)
                                    for s in selected_starts
                                )
                                if not overlaps_existing:
                                    selected_starts.append(start)
                                if len(selected_starts) == holes_per_layer:
                                    break

                        keep_mask = torch.ones(rows, dtype=torch.bool, device=self.device)
                        for start in selected_starts:
                            keep_mask[start : start + hole_rows] = False

                        oxy[env_i, layer_i, :, 0] = x_positions[layer_i]
                        oxy[env_i, layer_i, :, 1] = y_slots[keep_mask]

                # Intentionally no jitter so palisade segments remain tightly connected.
                oxy = oxy.view(n_envs * self._num_gate_palisade_obstacles, 2)
                oxy[:, 0] = torch.clamp(oxy[:, 0], -safe_h, safe_h)
                oxy[:, 1] = torch.clamp(oxy[:, 1], -safe_h, safe_h)

                repeated_env_ids = torch.repeat_interleave(
                    env_ids, self._num_gate_palisade_obstacles
                )

                gate_palisade_indices = []
                for i in range(n_envs):
                    base_idx = i * self.cfg.num_obstacles
                    gate_palisade_indices.extend(
                        range(base_idx, base_idx + self._num_gate_palisade_obstacles)
                    )

                gate_palisade_indices_tensor = torch.tensor(
                    gate_palisade_indices, dtype=torch.long, device=self.device
                )

                obs_root[gate_palisade_indices_tensor, 0] = (
                    oxy[:, 0] + self.scene.env_origins[repeated_env_ids, 0]
                )
                obs_root[gate_palisade_indices_tensor, 1] = (
                    oxy[:, 1] + self.scene.env_origins[repeated_env_ids, 1]
                )
                obs_root[gate_palisade_indices_tensor, 2] = 0.25

            self.obstacles.write_root_pose_to_sim(obs_root[:, :7], obs_env_ids)

        # Set Wall physical positions
        self._place_walls(env_ids)

    def _place_walls(self, env_ids: Sequence[int]) -> None:
        """Move the four boundary walls into place for the given envs.

        The walls are spawned stacked at the arena centre in _setup_scene and
        only positioned here, so any subclass that overrides _reset_idx without
        calling this leaves them piled at the origin.
        """
        h = self.cfg.arena_size / 2.0

        wall_indices = []
        for env_id in env_ids.tolist():
            start_idx = env_id * 4
            wall_indices.extend(range(start_idx, start_idx + 4))

        if len(wall_indices) > 0:
            wall_env_ids = torch.tensor(
                wall_indices, dtype=torch.long, device=self.device
            )
            wall_root = self.walls.data.default_root_state[wall_env_ids].clone()

            # Local offsets matching the alphabetical order of wall primitives
            # Bottom, Left, Right, Top
            local_offsets = torch.tensor(
                [
                    [0.0, -h, 0.5],  # Bottom
                    [-h, 0.0, 0.5],  # Left
                    [h, 0.0, 0.5],  # Right
                    [0.0, h, 0.5],  # Top
                ],
                device=self.device,
            )

            local_offsets = local_offsets.repeat(len(env_ids), 1)
            env_origins_repeated = self.scene.env_origins[
                torch.repeat_interleave(env_ids, 4)
            ]

            wall_root[:, :3] = env_origins_repeated + local_offsets
            self.walls.write_root_pose_to_sim(wall_root[:, :7], wall_env_ids)


# =============================================================================
# Helpers (module-level for potential torch.jit)
# =============================================================================


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz → (N,) yaw."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """(N,) yaw → (N,4) wxyz."""
    h = yaw * 0.5
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(h)
    q[:, 3] = torch.sin(h)
    return q


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi
