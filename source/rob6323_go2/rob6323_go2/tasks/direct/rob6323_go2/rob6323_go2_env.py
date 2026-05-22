# rob6323_go2_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # =========================
        # Logging (add all reward terms)
        # =========================
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "track_lin_vel_y_exp",
                "rew_action_rate",
                "raibert_heuristic",
                "orient",
                "base_height",
                "lin_vel_z",
                "dof_pos",
                "dof_vel",
                "ang_vel_xy",
                "feet_clearance",
                "tracking_contacts_shaped_force",
                "torque",
            ]
        }

        # =========================
        # Action history (rate/acc penalty)
        # Shape: (num_envs, action_dim, history_length)
        # =========================
        self.last_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # =========================
        # Custom PD controller params (torque control)
        # =========================
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.motor_offsets = torch.zeros(self.num_envs, 12, device=self.device)
        self.torque_limits = float(cfg.torque_limits)
        self.desired_joint_pos = torch.zeros_like(self._actions)

        # store applied torques for torque magnitude penalty
        self._applied_torques = torch.zeros(self.num_envs, 12, device=self.device)

        # =========================
        # Feet indices and gait / Raibert state
        # =========================
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        # indices in the robot articulation (for kinematics / positions)
        self._feet_ids: list[int] = []
        for name in foot_names:
            ids, _ = self.robot.find_bodies(name)
            if len(ids) == 0:
                raise RuntimeError(f"Could not find foot body '{name}' in articulation.")
            self._feet_ids.append(int(ids[0]))

        # indices in the contact sensor (for forces)
        self._feet_ids_sensor: list[int] = []
        for name in foot_names:
            ids, _ = self._contact_sensor.find_bodies(name)
            if len(ids) == 0:
                raise RuntimeError(f"Could not find foot body '{name}' in contact sensor.")
            self._feet_ids_sensor.append(int(ids[0]))

        # base id in contact sensor for termination contacts
        self._base_id, _ = self._contact_sensor.find_bodies("base")

        # gait / clock inputs
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        # (num_envs, 4) normalized phase for each foot
        self.foot_indices = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    # --------------------------
    # Scene setup
    # --------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add articulation + sensors to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # --------------------------
    # Helpers
    # --------------------------
    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Feet positions in world frame. Shape: (num_envs, 4, 3)."""
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _step_contact_targets(self) -> None:
        """Advance gait clock and compute desired_contact_states and clock_inputs."""
        frequencies = 3.0
        phases = 0.5
        offsets = 0.0
        bounds = 0.0
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,  # FL
            self.gait_indices + offsets,                    # FR
            self.gait_indices + bounds,                     # RL
            self.gait_indices + phases,                     # RR
        ]

        self.foot_indices = torch.remainder(torch.cat([fi.unsqueeze(1) for fi in foot_indices], dim=1), 1.0)

        # Warp indices for stance/swing shaping
        for idxs in foot_indices:
            stance = torch.remainder(idxs, 1.0) < durations
            swing = ~stance
            idxs[stance] = torch.remainder(idxs[stance], 1.0) * (0.5 / durations[stance])
            idxs[swing] = 0.5 + (torch.remainder(idxs[swing], 1.0) - durations[swing]) * (0.5 / (1.0 - durations[swing]))

        two_pi = 2.0 * math.pi
        self.clock_inputs[:, 0] = torch.sin(two_pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(two_pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(two_pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(two_pi * foot_indices[3])

        # Smooth desired contact states (normal CDF smoothing)
        kappa = 0.07
        normal = torch.distributions.normal.Normal(0.0, kappa)
        cdf = normal.cdf

        def smooth(fi: torch.Tensor) -> torch.Tensor:
            r = torch.remainder(fi, 1.0)
            return cdf(r) * (1.0 - cdf(r - 0.5)) + cdf(r - 1.0) * (1.0 - cdf(r - 1.5))

        self.desired_contact_states[:, 0] = smooth(foot_indices[0])
        self.desired_contact_states[:, 1] = smooth(foot_indices[1])
        self.desired_contact_states[:, 2] = smooth(foot_indices[2])
        self.desired_contact_states[:, 3] = smooth(foot_indices[3])

    def _reward_raibert_heuristic(self) -> torch.Tensor:
        """Squared error between Raibert desired foot placements and current ones (penalty)."""
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)

        # rotate into body yaw frame
        root_quat_w = self.robot.data.root_quat_w
        yaw_only_quat_conj = math_utils.quat_conjugate(root_quat_w)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
                yaw_only_quat_conj, cur_footsteps_translated[:, i, :]
            )

        # nominal stance
        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor(
            [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2],
            device=self.device,
        ).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor(
            [desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2],
            device=self.device,
        ).unsqueeze(0)

        # Raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)

        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]

        # Use actual lateral command vy (+ small yaw-induced term)
        y_vel_des = self._commands[:, 1:2] + yaw_vel_des * (desired_stance_length / 2.0)

        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1.0
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys = desired_ys_nom + desired_ys_offset
        desired_xs = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs.unsqueeze(2), desired_ys.unsqueeze(2)), dim=2)
        err = desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2]
        return torch.sum(torch.square(err), dim=(1, 2))

    def _reward_feet_clearance(self) -> torch.Tensor:
        """Penalize insufficient foot lift during swing."""
        swing_w = (1.0 - self.desired_contact_states).clamp(0.0, 1.0)
        foot_heights = self.foot_positions_w[:, :, 2]
        target_clearance = 0.08
        clearance_err = torch.clamp(target_clearance - foot_heights, min=0.0)
        return torch.sum(swing_w * torch.square(clearance_err), dim=1)

    def _reward_tracking_contacts_shaped_force(self) -> torch.Tensor:
        """Shape contact forces to match desired contact schedule."""
        forces_w = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor]
        force_norm = torch.linalg.norm(forces_w, dim=-1)

        desired = self.desired_contact_states.clamp(0.0, 1.0)

        F_stance = 50.0
        sigma_stance = 50.0
        sigma_swing = 10.0

        desired_force = desired * F_stance

        shaped_stance = torch.exp(-torch.square(force_norm - desired_force) / (2.0 * (sigma_stance**2)))
        shaped_swing = torch.exp(-torch.square(force_norm - 0.0) / (2.0 * (sigma_swing**2)))

        shaped = desired * shaped_stance + (1.0 - desired) * shaped_swing
        return torch.sum(shaped, dim=1)

    # --------------------------
    # RL hooks
    # --------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self.desired_joint_pos = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        torques = self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        self._applied_torques = torques
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,
                    self.robot.data.root_ang_vel_b,
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                    self.robot.data.joint_vel,
                    self._actions,
                    self.clock_inputs,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._step_contact_targets()

        # Tier A (3): tighten linear tracking slightly
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        track_lin = torch.exp(-lin_vel_error / 0.2125)  # 0.25 * 0.85 = 0.2125

        y_vel_error = torch.square(self._commands[:, 1] - self.robot.data.root_lin_vel_b[:, 1])
        track_y = torch.exp(-y_vel_error / 0.25)

        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        track_yaw = torch.exp(-yaw_rate_error / 0.25)

        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (
            self.cfg.action_scale**2
        )
        rew_action_rate += torch.sum(
            torch.square(self._actions - 2.0 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1
        ) * (self.cfg.action_scale**2)

        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions

        rew_raibert = self._reward_raibert_heuristic()

        rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        base_height = self.robot.data.root_pos_w[:, 2]
        rew_base_height = torch.square(base_height - float(self.cfg.base_height_target))
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        rew_dof_pos = torch.sum(torch.square(self.robot.data.joint_pos - self.robot.data.default_joint_pos), dim=1)
        rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)

        rew_feet_clearance = self._reward_feet_clearance()
        rew_contact_force = self._reward_tracking_contacts_shaped_force()

        rew_torque = torch.sum(torch.square(self._applied_torques), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": track_lin * self.cfg.lin_vel_reward_scale,
            "track_lin_vel_y_exp": track_y * self.cfg.y_vel_reward_scale,
            "track_ang_vel_z_exp": track_yaw * self.cfg.yaw_rate_reward_scale,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "raibert_heuristic": rew_raibert * self.cfg.raibert_heuristic_reward_scale,
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "base_height": rew_base_height * self.cfg.base_height_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_pos": rew_dof_pos * self.cfg.dof_pos_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "tracking_contacts_shaped_force": rew_contact_force * self.cfg.tracking_contacts_shaped_force_reward_scale,
            "torque": rew_torque * self.cfg.torque_reward_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        cstr_termination_contacts = torch.any(
            torch.max(torch.linalg.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )

        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0.0

        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < float(self.cfg.base_height_min)

        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.desired_joint_pos[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self.gait_indices[env_ids] = 0.0
        self.clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 0.0
        self.foot_indices[env_ids] = 0.0
        self._applied_torques[env_ids] = 0.0

        # command sampling (side-step biased)
        n = len(env_ids)
        r = torch.rand(n, device=self.device)

        cmds = torch.zeros(n, 3, device=self.device)

        mask_lat = r < 0.50
        mask_gen = (r >= 0.50) & (r < 0.90)
        mask_fwd = r >= 0.90

        if torch.any(mask_lat):
            idx = torch.where(mask_lat)[0]
            cmds[idx, 0] = 0.0
            cmds[idx, 1] = torch.zeros(len(idx), device=self.device).uniform_(-1.0, 1.0)
            cmds[idx, 2] = torch.zeros(len(idx), device=self.device).uniform_(-0.5, 0.5)

        if torch.any(mask_gen):
            idx = torch.where(mask_gen)[0]
            cmds[idx] = torch.zeros(len(idx), 3, device=self.device).uniform_(-1.0, 1.0)

        if torch.any(mask_fwd):
            idx = torch.where(mask_fwd)[0]
            cmds[idx, 0] = torch.zeros(len(idx), device=self.device).uniform_(-1.0, 1.0)
            cmds[idx, 1] = 0.0
            cmds[idx, 2] = torch.zeros(len(idx), device=self.device).uniform_(-0.5, 0.5)

        self._commands[env_ids] = cmds

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)

        extras = {}
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    # --------------------------
    # Debug visualization
    # --------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat
