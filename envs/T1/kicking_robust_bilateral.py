from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import get_euler_xyz, quat_rotate, quat_rotate_inverse, to_torch

assert gymtorch

import numpy as np
import torch

from envs.T1.kicking import Kicking as BaseKicking
from utils.utils import apply_randomization


class KickingRobustBilateral(BaseKicking):

    def _init_buffers(self):
        super()._init_buffers()
        self.kick_detected_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.time_since_kick_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stable_hold_time_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.kick_ball_start_x_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.kick_detected_buf[env_ids] = False
        self.time_since_kick_buf[env_ids] = 0.0
        self.stable_hold_time_buf[env_ids] = 0.0
        self.kick_ball_start_x_buf[env_ids] = self.root_states[env_ids, 1, 0]

    def _reset_ball_at_robot_front(self, env_ids_to_reset_ball):
        if len(env_ids_to_reset_ball) == 0:
            return

        n = len(env_ids_to_reset_ball)
        robot_pos  = self.root_states[env_ids_to_reset_ball, 0, 0:3]
        robot_quat = self.root_states[env_ids_to_reset_ball, 0, 3:7]

        # Polar-coordinate sampling: ball appears at random angle in a fan in front of robot
        angle_min = self.cfg["randomization"].get("ball_init_angle_min", -0.785)  # -45 deg
        angle_max = self.cfg["randomization"].get("ball_init_angle_max",  0.785)  # +45 deg
        dist_min  = self.cfg["randomization"].get("ball_init_dist_min",   0.30)
        dist_max  = self.cfg["randomization"].get("ball_init_dist_max",   0.45)

        angles = torch.rand(n, device=self.device) * (angle_max - angle_min) + angle_min
        dists  = torch.rand(n, device=self.device) * (dist_max  - dist_min)  + dist_min

        ball_local_offset = torch.zeros((n, 3), dtype=torch.float, device=self.device)
        ball_local_offset[:, 0] = dists * torch.cos(angles)
        ball_local_offset[:, 1] = dists * torch.sin(angles)

        ball_world_offset = quat_rotate(robot_quat, ball_local_offset)
        ball_target_xy = robot_pos[:, 0:2] + ball_world_offset[:, 0:2]

        if hasattr(self, "terrain"):
            ball_target_z = self.terrain.terrain_heights(ball_target_xy) + self.ball_radius
        else:
            ball_target_z = torch.full_like(ball_target_xy[:, 0], self.ball_radius)

        self.root_states[env_ids_to_reset_ball, 1, 0] = ball_target_xy[:, 0]
        self.root_states[env_ids_to_reset_ball, 1, 1] = ball_target_xy[:, 1]
        self.root_states[env_ids_to_reset_ball, 1, 2] = ball_target_z
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(n, 1)
        self.root_states[env_ids_to_reset_ball, 1, 3:7] = identity_quat
        self.root_states[env_ids_to_reset_ball, 1, 7:13] = 0.0

        ball_actor_indices = (2 * env_ids_to_reset_ball + 1).to(dtype=torch.int32)
        if len(ball_actor_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(ball_actor_indices),
                len(ball_actor_indices),
            )

    def _compute_observations(self):
        ball_pos_world_frame = self.ball_pos - self.base_pos
        relative_ball_pos = quat_rotate_inverse(self.base_quat, ball_pos_world_frame)

        ball_pos_norm = self.cfg["normalization"]["ball_pos"]

        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],  # 3
                apply_randomization(self.base_ang_vel,      self.cfg["noise"].get("ang_vel"))  * self.cfg["normalization"]["ang_vel"],  # 3
                apply_randomization(relative_ball_pos[:, 0:2], self.cfg["noise"].get("ball_pos")) * ball_pos_norm,                     # 2
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],  # 12
                apply_randomization(self.dof_vel,  self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],           # 12
                self.actions,                                                                                                          # 12
            ),
            dim=-1,
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.ball_lin_vel[:, 0:2],
                self.feet_pos[:, 0, 0:2],
                self.feet_pos[:, 1, 0:2],
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    def step(self, actions):
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions

        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        prev_ball_lin_vel_world = self.root_states[:, 1, 7:10].clone()

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.ball_pos[:] = self.root_states[:, 1, 0:3]
        self.ball_lin_vel[:] = self.body_states[:, -1, 7:10]
        self.ball_ang_vel[:] = self.body_states[:, -1, 10:13]

        self.base_pos[:] = self.root_states[:, 0, 0:3]
        self.base_quat[:] = self.root_states[:, 0, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self._refresh_feet_state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        current_ball_vel_world = self.root_states[:, 1, 7:10]
        ball_forward_speed_increase = current_ball_vel_world[:, 0] - prev_ball_lin_vel_world[:, 0]
        kick_detected_now = (~self.kick_detected_buf) & (
            ball_forward_speed_increase > self.cfg["rewards"].get("kick_detection_speed_increase_threshold", 0.5)
        ) & (current_ball_vel_world[:, 0] > self.cfg["rewards"].get("min_kick_forward_speed", 0.6))
        if kick_detected_now.any():
            self.kick_detected_buf[kick_detected_now] = True
            self.time_since_kick_buf[kick_detected_now] = 0.0
            self.stable_hold_time_buf[kick_detected_now] = 0.0
            self.kick_ball_start_x_buf[kick_detected_now] = self.ball_pos[kick_detected_now, 0]

        self.time_since_kick_buf = torch.where(
            self.kick_detected_buf,
            self.time_since_kick_buf + self.dt,
            torch.zeros_like(self.time_since_kick_buf),
        )
        post_kick_stable = self._post_kick_stable_mask()
        self.stable_hold_time_buf = torch.where(
            self.kick_detected_buf & post_kick_stable,
            self.stable_hold_time_buf + self.dt,
            torch.zeros_like(self.stable_hold_time_buf),
        )

        self._kick_robots()
        self._push_robots()
        self._check_termination()

        ball_only_reset_env_ids = (self.reset_ball_buf & ~self.reset_buf).nonzero(as_tuple=False).flatten()
        if len(ball_only_reset_env_ids) > 0:
            self._reset_ball_at_robot_front(ball_only_reset_env_ids)
            self.ball_pos[ball_only_reset_env_ids] = self.root_states[ball_only_reset_env_ids, 1, 0:3]
            ball_quat_reset = self.root_states[ball_only_reset_env_ids, 1, 3:7]
            self.ball_rot[ball_only_reset_env_ids] = ball_quat_reset
            world_lin_vel_reset = self.root_states[ball_only_reset_env_ids, 1, 7:10]
            world_ang_vel_reset = self.root_states[ball_only_reset_env_ids, 1, 10:13]
            self.ball_lin_vel[ball_only_reset_env_ids] = quat_rotate_inverse(ball_quat_reset, world_lin_vel_reset)
            self.ball_ang_vel[ball_only_reset_env_ids] = quat_rotate_inverse(ball_quat_reset, world_ang_vel_reset)
            self.reset_ball_buf[ball_only_reset_env_ids] = False

        self._compute_reward()
        self._log_rewards_to_csv()
        self.last_ball_lin_vel_world[:] = self.body_states[:, -1, 7:10]

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._reset_idx(env_ids)
            self.reset_ball_buf[env_ids] = False
            self.last_ball_lin_vel_world[env_ids] = 0.0

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 0, 7:13]
        self.last_feet_pos[:] = self.feet_pos

        # print(f"env_resets: {self.env_resets}, env_successes: {self.env_successes}, env_falling: {self.env_falling}")
        # if len(self.ball_velocities) > 0:
        #     print(
        #         f"ball_velocities average: {np.mean(self.ball_velocities)}, "
        #         f"std: {np.std(self.ball_velocities)}, max: {np.max(self.ball_velocities)}"
        #     )

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _check_termination(self):
        terminate_contacts = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        terminate_vel = self.root_states[:, 0, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        terminate_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        episode_timeout = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        no_kick_timeout = (~self.kick_detected_buf) & (
            self.episode_length_buf > np.ceil(self.cfg["rewards"].get("kick_timeout_s", 3.0) / self.dt)
        )
        success = self.kick_detected_buf & (
            self._ball_forward_travel() > self.cfg["rewards"].get("min_ball_travel_x", 0.4)
        ) & (self.stable_hold_time_buf > self.cfg["rewards"].get("post_kick_hold_s", 0.35))

        self.reset_buf = terminate_contacts | terminate_vel | terminate_height | episode_timeout | no_kick_timeout | success
        self.time_out_buf = episode_timeout | no_kick_timeout
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time

        self.env_successes += int(success.sum().item())
        self.env_falling += int((terminate_height | terminate_vel).sum().item())

    def _ball_forward_travel(self):
        return torch.clamp(self.ball_pos[:, 0] - self.kick_ball_start_x_buf, min=0.0)

    def _post_kick_stable_mask(self):
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        base_lin_speed = torch.norm(self.base_lin_vel[:, :2], dim=-1)
        base_ang_speed = torch.norm(self.base_ang_vel, dim=-1)
        upright = torch.norm(self.projected_gravity[:, :2], dim=-1) < 0.35
        both_feet_contact = torch.all(self.feet_contact, dim=1)
        return (
            (base_height > self.cfg["rewards"]["terminate_height"] + 0.08)
            & (base_lin_speed < self.cfg["rewards"].get("post_kick_max_base_lin_vel", 0.45))
            & (base_ang_speed < self.cfg["rewards"].get("post_kick_max_base_ang_vel", 1.25))
            & upright
            & both_feet_contact
        )

    def _compute_reward(self):
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            raw_reward_values = self.reward_functions[i]()
            normal_scale = self.reward_scales[name]
            effective_scales_for_envs = torch.full_like(raw_reward_values, normal_scale)
            ball_moving_specific_scale = self.reward_scales_ball_rolling.get(name)
            if ball_moving_specific_scale is not None:
                effective_scales_for_envs = torch.where(
                    self.kick_detected_buf,
                    torch.full_like(raw_reward_values, ball_moving_specific_scale),
                    effective_scales_for_envs,
                )

            rew = raw_reward_values * effective_scales_for_envs
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _reward_ball_velocity_target_direction(self):
        target_position = to_torch(
            self.cfg["rewards"].get("kick_target_pos_world", [5.0, 0.0, 0.05]), device=self.device
        ).unsqueeze(0)
        ball_pos_world = self.body_states[:, -1, 0:3]
        ball_vel_world = self.body_states[:, -1, 7:10]
        ball_to_target = target_position - ball_pos_world
        distance_to_target = torch.norm(ball_to_target, dim=-1, keepdim=True)
        ball_to_target_normalized = ball_to_target / (distance_to_target + 1e-6)
        velocity_towards_target = torch.sum(ball_vel_world * ball_to_target_normalized, dim=-1)
        decay_time_constant = self.cfg["rewards"].get("ball_velocity_decay_time", 2.0)
        decay_factor = torch.exp(-self.time_since_kick_buf / decay_time_constant)
        reward = velocity_towards_target * decay_factor
        max_reward = self.cfg["rewards"].get("max_ball_vel_target_reward", 5.0)
        reward = torch.where(self.kick_detected_buf, reward, torch.zeros_like(reward))
        return torch.clamp(reward, min=0.0, max=max_reward)

    def _reward_kicking_foot_approach_ball_stationary(self):
        current_ball_pos_world = self.body_states[:, -1, 0:3]
        # Use whichever foot is closer — robot learns to pick the more stable one
        dist_left  = torch.norm(self.feet_pos[:, 0, :] - current_ball_pos_world, dim=-1)
        dist_right = torch.norm(self.feet_pos[:, 1, :] - current_ball_pos_world, dim=-1)
        foot_ball_dist = torch.min(dist_left, dist_right)
        proximity_sigma = self.cfg["rewards"].get("approach_proximity_sigma", 0.1)
        proximity_value = torch.exp(-foot_ball_dist / proximity_sigma)
        ball_stationary = torch.norm(self.body_states[:, -1, 7:10], dim=-1) < self.cfg["rewards"].get("ball_stationary_speed_threshold", 0.1)
        reward = torch.where(~self.kick_detected_buf & ball_stationary, proximity_value, torch.zeros_like(proximity_value))
        max_reward = self.cfg["rewards"].get("max_approach_reward", 2.0)
        return torch.clamp(reward, min=0.0, max=max_reward)

    def _reward_body_alignment_for_kick(self):
        robot_pos_world = self.base_pos
        robot_forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        robot_forward_world = quat_rotate(self.base_quat, robot_forward_local)
        kick_target_pos_world = to_torch(
            self.cfg["rewards"].get("kick_target_pos_world", [5.0, 0.0, self.ball_radius]), device=self.device
        ).unsqueeze(0)
        robot_to_target_world = kick_target_pos_world - robot_pos_world
        robot_to_target_world_normalized = robot_to_target_world / (torch.norm(robot_to_target_world, dim=-1, keepdim=True) + 1e-6)
        alignment_to_target = torch.sum(robot_forward_world * robot_to_target_world_normalized, dim=-1)
        sigma_target = self.cfg["rewards"].get("alignment_to_target_sigma", 0.5)
        reward_align_target = torch.exp((alignment_to_target - 1.0) / sigma_target)
        max_reward = self.cfg["rewards"].get("max_alignment_reward", 1.0)
        reward_align_target = torch.where(self.kick_detected_buf, torch.zeros_like(reward_align_target), reward_align_target)
        return torch.clamp(reward_align_target, min=0.0, max=max_reward)

    def _reward_ball_acceleration(self):
        current_ball_vel_world = self.body_states[:, -1, 7:9]
        prev_ball_vel_world = self.last_ball_lin_vel_world[:, :2]
        ball_acceleration = (current_ball_vel_world - prev_ball_vel_world) / self.dt
        ball_effective_acceleration = ball_acceleration[:, 0] - torch.abs(ball_acceleration[:, 1])
        acceleration_scale = self.cfg["rewards"].get("ball_acceleration_scale", 10.0)
        max_acceleration_reward = self.cfg["rewards"].get("max_ball_acceleration_reward", 1.0)
        reward = torch.tanh(torch.clamp(ball_effective_acceleration, min=0.0) / acceleration_scale) * max_acceleration_reward
        reward = torch.where(
            self.kick_detected_buf & (self.time_since_kick_buf < 0.25),
            reward,
            torch.zeros_like(reward),
        )
        return reward

    def _reward_waiting(self):
        progress = self.episode_length_buf / np.ceil(self.cfg["rewards"].get("kick_timeout_s", 3.0) / self.dt)
        return torch.where(self.kick_detected_buf, torch.zeros_like(progress), progress * progress)

    def _reward_kick_foot_velocity_penalty(self):
        # Penalizes large foot speed before kick to discourage power-kicks that cause falls
        feet_lin_vel = self.body_states[:, self.feet_indices, 7:10]  # (N, 2, 3)
        feet_speed   = torch.norm(feet_lin_vel, dim=-1)              # (N, 2)
        max_foot_speed = torch.max(feet_speed, dim=-1).values        # (N,)
        threshold = self.cfg["rewards"].get("foot_velocity_penalty_threshold", 1.5)
        excess = torch.clamp(max_foot_speed - threshold, min=0.0)
        return torch.where(~self.kick_detected_buf, excess, torch.zeros_like(excess))

    def _reward_post_kick_stability(self):
        # Soft, continuous versions of each _post_kick_stable_mask() condition
        # to provide gradient signal toward post-kick balance.
        gravity_xy_sq = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        upright_reward = torch.exp(-gravity_xy_sq / (2.0 * 0.12 ** 2))

        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        height_threshold = self.cfg["rewards"]["terminate_height"] + 0.08
        height_reward = torch.sigmoid((base_height - height_threshold) / 0.05)

        base_lin_speed_xy = torch.norm(self.base_lin_vel[:, :2], dim=-1)
        lin_vel_reward = torch.exp(-base_lin_speed_xy / 0.3)

        both_feet = torch.all(self.feet_contact, dim=1).float()

        base_stability = upright_reward * height_reward * lin_vel_reward * both_feet

        hold_time_bonus = 1.0 + torch.clamp(
            self.stable_hold_time_buf / self.cfg["rewards"].get("post_kick_hold_s", 0.35),
            min=0.0,
            max=1.0,
        )

        reward = base_stability * hold_time_bonus
        return torch.where(self.kick_detected_buf, reward, torch.zeros_like(reward))

    def _reward_post_kick_return_to_default(self):
        # Encourage robot to return to default joint angles after kicking,
        # so it is ready to walk again.
        dof_error_sq = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)
        sigma_sq = self.cfg["rewards"].get("post_kick_default_pose_sigma_sq", 1.0)
        reward = torch.exp(-dof_error_sq / sigma_sq)
        return torch.where(self.kick_detected_buf, reward, torch.zeros_like(reward))


