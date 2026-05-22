# rob6323_go2_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0

    # - spaces definition
    action_scale = 0.25
    action_space = 12
    # +4 clock inputs for gait phase
    observation_space = 48 + 4
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Disable implicit PD gains and drive with our own torque PD controller in the env.
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # disable implicit P-gain
        damping=0.0,    # disable implicit D-gain
    )

    # custom PD gains (applied in env)
    Kp = 28.0
    Kd = 1.0
    torque_limits = 23.5  # matches actuator effort_limit

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # visualization markers
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # termination
    base_height_min = 0.26  # meters

    # posture target (used in reward)
    base_height_target = 0.32  # meters

    # reward scales (tracking)
    lin_vel_reward_scale = 1.25   # Tier A: 1.0 * 1.25
    yaw_rate_reward_scale = 0.55  # Tier A: 0.5 * 1.10

    # side-step helper (kept as-is)
    y_vel_reward_scale = 0.5

    # reward scales (tutorial additions)
    action_rate_reward_scale = -0.0001
    raibert_heuristic_reward_scale = -10.0
    feet_clearance_reward_scale = -45.0
    tracking_contacts_shaped_force_reward_scale = 4.0

    # reward scales (stability / smoothness shaping)
    orient_reward_scale = -5.0
    base_height_reward_scale = -2.5
    lin_vel_z_reward_scale = -0.02
    dof_pos_reward_scale = -0.005
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.001

    # torque magnitude regularization
    torque_reward_scale = -0.00001
