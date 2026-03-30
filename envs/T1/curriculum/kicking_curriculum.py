"""
KickingCurriculum — 3-stage curriculum on top of the Kicking environment.

Stage 0  (walk):     Robot learns locomotion by tracking random velocity commands.
                     Ball is present but all kick rewards are disabled.
Stage 1  (approach): Nav commands steer the robot toward the ball.
                     Approach proximity reward is active; kick rewards still off.
Stage 2  (kick):     Robot is within kick range and aligned.
                     Nav commands are zeroed; full kicking rewards activate.

Global stage is gated by total env-steps (configurable thresholds).
Stage 1→2 and 2→1 transitions happen per-env based on ball proximity / ball motion.

Observation layout  (num_observations = 47):
  [gravity(3), ang_vel(3), ball_rel_pos_xy(2), nav_commands(3),
   dof_pos(12), dof_vel(12), actions(12)]
"""

import math
import torch

from isaacgym.torch_utils import (
    torch_rand_float,
    get_euler_xyz,
    to_torch,
)

from envs.T1.kicking import Kicking


class KickingCurriculum(Kicking):
    """Curriculum wrapper around Kicking with walk → approach → kick stages."""

    # Extra dimensions appended to observations: [lin_vel_x, lin_vel_y, ang_vel_yaw]
    NUM_NAV_COMMANDS = 3

    # ------------------------------------------------------------------ #
    #  Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, cfg):
        # Expand observation size before the parent allocates obs_buf.
        cfg["env"]["num_observations"] += self.NUM_NAV_COMMANDS
        super().__init__(cfg)

    # _init_buffers is called by Kicking.__init__ → Python MRO routes here first.
    def _init_buffers(self):
        super()._init_buffers()

        # Per-env stage: 0=walk, 1=approach, 2=kick
        self.env_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Navigation commands exposed to the policy and used by tracking rewards.
        # Stage 0: random walk commands  |  Stage 1: ball-directed  |  Stage 2: zero
        self.nav_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # Per-env time (in steps) at which to draw the next Stage-0 random command.
        self.nav_cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Precomputed per-stage reward multipliers from cfg.
        self._build_stage_multipliers()

        # Rolling averages used for performance-based stage advancement.
        # Exponential moving average: new = alpha * sample + (1-alpha) * old
        self._ema_tracking = 0.0   # tracks walk quality  (stage 0 → 1)
        self._ema_proximity = 0.0  # tracks approach quality (stage 1 → 2)
        self._ema_alpha = 0.005    # smoothing factor (~200-step window)

    def _build_stage_multipliers(self):
        """Load stage_reward_multipliers from cfg into a plain dict of dicts."""
        raw = self.cfg["curriculum"].get("stage_reward_multipliers", {})
        # Keys in YAML are strings ("0", "1", "2"); normalise to ints.
        self.stage_multipliers = {}
        for stage_idx in range(3):
            self.stage_multipliers[stage_idx] = raw.get(str(stage_idx), {})

    # ------------------------------------------------------------------ #
    #  Global curriculum gate                                               #
    # ------------------------------------------------------------------ #

    def _global_unlock_stage(self) -> int:
        """
        Return the maximum stage any env is allowed to be in right now.

        Uses performance-based thresholds when available, with a minimum
        step-count floor to prevent premature advancement on noise.

        Stage 0 → 1: walk tracking EMA  >= stage0_tracking_threshold
                     AND total steps    >= stage0_min_steps
        Stage 1 → 2: approach proximity EMA >= stage1_proximity_threshold
                     AND total steps    >= stage1_min_steps
        """
        initial = self.cfg["curriculum"].get("initial_stage", 0)
        if initial >= 2:
            return 2

        cur = self.cfg["curriculum"]
        total_steps = self.common_step_counter * self.num_envs

        walk_ready = (
            self._ema_tracking >= cur.get("stage0_tracking_threshold", 0.7)
            and total_steps    >= cur.get("stage0_min_steps", 100_000)
        )
        approach_ready = (
            self._ema_proximity >= cur.get("stage1_proximity_threshold", 0.5)
            and total_steps     >= cur.get("stage1_min_steps", 300_000)
        )

        if walk_ready and approach_ready:
            return max(2, initial)
        if walk_ready:
            return max(1, initial)
        return initial

    def _update_ema_metrics(self):
        """
        Update rolling averages used by _global_unlock_stage.

        We call the reward functions directly to obtain un-scaled [0, 1]
        values.  This avoids the bug of trying to reverse dt × scale
        multiplication from the already-scaled extras["rew_terms"].
        """
        a = self._ema_alpha

        # Walk quality: average of velocity-x and yaw tracking (both 0..1).
        track_x   = self._reward_tracking_lin_vel_x()   # shape (num_envs,)
        track_yaw = self._reward_tracking_ang_vel()
        mean_track = ((track_x + track_yaw) * 0.5).mean().item()
        self._ema_tracking = a * mean_track + (1 - a) * self._ema_tracking

        # Approach quality: proximity reward, only for stage-1 envs (0..1).
        stage1_mask = self.env_stage == 1
        if stage1_mask.any():
            prox = self._reward_nav_ball_proximity()     # shape (num_envs,)
            mean_prox = prox[stage1_mask].mean().item()
            self._ema_proximity = a * mean_prox + (1 - a) * self._ema_proximity

    # ------------------------------------------------------------------ #
    #  Per-env stage transitions                                            #
    # ------------------------------------------------------------------ #

    def _update_env_stages(self):
        """
        Transitions (post-physics, called from _compute_reward):
          approach (1) → kick (2): dist_to_ball < kick_range AND heading aligned
          kick     (2) → approach (1): ball starts moving (kick detected)
        """
        unlock = self._global_unlock_stage()

        # ── Approach → Kick ──────────────────────────────────────────────
        if unlock >= 2:
            kick_range = self.cfg["curriculum"].get("kick_approach_range", 0.5)
            max_heading_err = self.cfg["curriculum"].get(
                "kick_approach_heading_threshold", 0.5
            )  # radians (~28°)

            dist_to_ball = torch.norm(
                self.ball_pos[:, :2] - self.base_pos[:, :2], dim=-1
            )
            _, _, robot_yaw = get_euler_xyz(self.base_quat)
            ball_rel = self.ball_pos - self.base_pos
            ball_bearing = torch.atan2(ball_rel[:, 1], ball_rel[:, 0])
            heading_err = torch.abs(
                (ball_bearing - robot_yaw + torch.pi) % (2 * torch.pi) - torch.pi
            )

            can_kick = (dist_to_ball < kick_range) & (heading_err < max_heading_err)
            self.env_stage = torch.where(
                (self.env_stage == 1) & can_kick,
                torch.full_like(self.env_stage, 2),
                self.env_stage,
            )

        # ── Kick → Approach ──────────────────────────────────────────────
        ball_speed = torch.norm(self.ball_lin_vel, dim=-1)
        ball_moving = ball_speed > self.cfg["rewards"].get(
            "ball_stationary_speed_threshold", 0.1
        )
        self.env_stage = torch.where(
            (self.env_stage == 2) & ball_moving,
            torch.ones_like(self.env_stage),  # back to approach
            self.env_stage,
        )

    # ------------------------------------------------------------------ #
    #  Navigation commands                                                  #
    # ------------------------------------------------------------------ #

    def _compute_nav_commands(self):
        """
        Compute nav_commands for the current step based on env_stage:
          Stage 0 → random walk commands (resampled periodically)
          Stage 1 → ball-directed commands
          Stage 2 → zero (kicking rewards take over)
        """
        max_vx = self.cfg["curriculum"].get("nav_max_lin_vel", 0.5)
        max_wz = self.cfg["curriculum"].get("nav_max_ang_vel", 1.0)

        # ── Stage 0: random walk ─────────────────────────────────────────
        self._maybe_resample_walk_commands()

        # ── Stage 1: ball-directed ────────────────────────────────────────
        ball_rel = self.ball_pos - self.base_pos  # world frame, shape (N,3)
        dist_to_ball = torch.norm(ball_rel[:, :2], dim=-1)
        _, _, robot_yaw = get_euler_xyz(self.base_quat)
        ball_bearing = torch.atan2(ball_rel[:, 1], ball_rel[:, 0])
        heading_err = (ball_bearing - robot_yaw + torch.pi) % (2 * torch.pi) - torch.pi

        # Forward speed: proportional to distance, scaled down by heading misalignment.
        vel_fwd = (
            torch.cos(heading_err).clamp(min=0.0) * dist_to_ball.clamp(max=max_vx)
        ).clamp(max=max_vx)
        nav_approach = torch.stack(
            [vel_fwd, torch.zeros_like(vel_fwd), heading_err.clamp(-max_wz, max_wz)],
            dim=-1,
        )

        # ── Select by stage ───────────────────────────────────────────────
        stage = self.env_stage.unsqueeze(-1)  # (N,1) for broadcasting
        # Stage 0: keep random commands already written by _maybe_resample_walk_commands
        # Stage 1: replace with ball-directed
        # Stage 2: zero
        in_approach = (stage == 1)
        in_kick = (stage == 2)
        self.nav_commands = torch.where(
            in_approach,
            nav_approach,
            torch.where(in_kick, torch.zeros_like(self.nav_commands), self.nav_commands),
        )

    def _maybe_resample_walk_commands(self):
        """Resample random walk commands for Stage-0 envs that are due."""
        stage0_mask = self.env_stage == 0
        due_mask = self.episode_length_buf >= self.nav_cmd_resample_time
        resample_envs = (stage0_mask & due_mask).nonzero(as_tuple=False).flatten()

        if len(resample_envs) == 0:
            return

        lin_range = self.cfg["curriculum"].get("stage0_lin_vel_range", [-0.5, 0.5])
        ang_range = self.cfg["curriculum"].get("stage0_ang_vel_range", [-1.0, 1.0])

        self.nav_commands[resample_envs, 0] = torch_rand_float(
            lin_range[0], lin_range[1], (len(resample_envs), 1), device=self.device
        ).squeeze(1)
        self.nav_commands[resample_envs, 1] = torch_rand_float(
            lin_range[0] * 0.5, lin_range[1] * 0.5, (len(resample_envs), 1), device=self.device
        ).squeeze(1)
        self.nav_commands[resample_envs, 2] = torch_rand_float(
            ang_range[0], ang_range[1], (len(resample_envs), 1), device=self.device
        ).squeeze(1)

        # Zero out a fraction of commands ("stand still" training).
        still_prop = self.cfg["commands"].get("still_proportion", 0.1)
        n_still = int(still_prop * len(resample_envs))
        if n_still > 0:
            perm = torch.randperm(len(resample_envs), device=self.device)
            self.nav_commands[resample_envs[perm[:n_still]]] = 0.0

        # Schedule the next resample for each env.
        resample_lo = self.cfg["curriculum"].get("stage0_resample_time_lo_s", 8.0)
        resample_hi = self.cfg["curriculum"].get("stage0_resample_time_hi_s", 12.0)
        self.nav_cmd_resample_time[resample_envs] = (
            self.episode_length_buf[resample_envs]
            + torch.randint(
                int(resample_lo / self.dt),
                int(resample_hi / self.dt),
                (len(resample_envs),),
                device=self.device,
            )
        )

    # ------------------------------------------------------------------ #
    #  Reset                                                                #
    # ------------------------------------------------------------------ #

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        unlock = self._global_unlock_stage()

        # Assign starting stage based on global unlock.
        #   Stage 0 locked  → all envs stay in stage 0
        #   Stage 1+ unlocked → start in approach (1); kick (2) entered dynamically
        initial_env_stage = 0 if unlock == 0 else 1
        self.env_stage[env_ids] = initial_env_stage

        # Reset resample timer so walk commands are drawn immediately after reset.
        self.nav_cmd_resample_time[env_ids] = 0

    # ------------------------------------------------------------------ #
    #  Termination                                                          #
    # ------------------------------------------------------------------ #

    def _check_termination(self):
        """
        Override to disable ball-related termination conditions during
        Stage 0, where the ball is never kicked and would otherwise kill
        every episode after max_ball_still_time_s (2 s).
        """
        super()._check_termination()

        # In stage 0, undo the ball-still and ball-moving resets that the
        # parent just applied.  Keep all other termination conditions
        # (fall, velocity, episode timeout, etc.).
        stage0_mask = self.env_stage == 0
        if stage0_mask.any():
            # Re-evaluate which stage-0 envs were ONLY reset due to ball timers.
            # The safest approach: just clear the reset flag for stage-0 envs
            # that haven't genuinely terminated (still standing, within time).
            genuine_term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # contact termination
            genuine_term |= torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1
            )
            # velocity termination
            genuine_term |= self.root_states[:, 0, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
            # fall termination
            genuine_term |= self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
            # episode timeout
            genuine_term |= self.episode_length_buf > int(math.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt))

            # For stage-0 envs: only reset if a genuine (non-ball) termination occurred.
            self.reset_buf[stage0_mask] = genuine_term[stage0_mask]

    # ------------------------------------------------------------------ #
    #  Step — no override needed; parent Kicking.step() drives everything. #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Observations                                                         #
    # ------------------------------------------------------------------ #

    def _compute_observations(self):
        # Recompute nav commands using the latest (post-reset) state.
        self._compute_nav_commands()

        # Build base 44-dim observation from parent.
        super()._compute_observations()

        # Normalise and append nav commands (3 dims → total 47).
        lin_scale = self.cfg["normalization"]["lin_vel"]
        ang_scale = self.cfg["normalization"]["ang_vel"]
        nav_scale = to_torch([lin_scale, lin_scale, ang_scale], device=self.device)
        self.obs_buf = torch.cat(
            [self.obs_buf, self.nav_commands * nav_scale], dim=-1
        )

    # ------------------------------------------------------------------ #
    #  Rewards                                                              #
    # ------------------------------------------------------------------ #

    def _compute_reward(self):
        """
        Extends parent reward computation with per-env stage multipliers.

        For each reward term, the effective scale is:
            base_scale
            × ball_rolling_override   (if ball is moving and the term has one)
            × stage_multiplier        (per-env, from stage_reward_multipliers cfg)
        """
        # Update per-env stages based on the post-physics state.
        self._update_env_stages()

        ball_is_moving = (
            torch.norm(self.ball_lin_vel, dim=-1)
            >= self.cfg["rewards"].get("ball_stationary_speed_threshold", 0.1)
        )

        self.rew_buf[:] = 0.0
        for i, (name, fn) in enumerate(
            zip(self.reward_names, self.reward_functions)
        ):
            raw = fn()  # shape: (num_envs,)

            # Base scale (already multiplied by dt in _prepare_reward_function).
            eff = torch.full_like(raw, self.reward_scales[name])

            # Ball-rolling override (from parent cfg ball_rolling_scale).
            ball_roll_scale = self.reward_scales_ball_rolling.get(name)
            if ball_roll_scale is not None:
                eff = torch.where(
                    ball_is_moving,
                    torch.full_like(raw, ball_roll_scale),
                    eff,
                )

            # Per-env stage multiplier.
            for stage_idx in range(3):
                mult = self.stage_multipliers[stage_idx].get(name, 1.0)
                if mult != 1.0:
                    mask = self.env_stage == stage_idx
                    eff[mask] *= mult

            rew = raw * eff
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew

        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

        self._update_ema_metrics()

        # Expose curriculum metrics as env attributes — the runner reads
        # these in its record_statistics call (see runner.py modification).
        stage_counts = torch.bincount(self.env_stage, minlength=3)
        self.curriculum_unlock_stage = float(self._global_unlock_stage())
        self.curriculum_ema_tracking = self._ema_tracking
        self.curriculum_ema_proximity = self._ema_proximity
        self.curriculum_stage0_envs = stage_counts[0].item()
        self.curriculum_stage1_envs = stage_counts[1].item()
        self.curriculum_stage2_envs = stage_counts[2].item()

    # ------------------------------------------------------------------ #
    #  Override tracking rewards to use nav_commands                        #
    # ------------------------------------------------------------------ #

    def _reward_tracking_lin_vel_x(self):
        return torch.exp(
            -torch.square(self.nav_commands[:, 0] - self.filtered_lin_vel[:, 0])
            / self.cfg["rewards"]["tracking_sigma"]
        )

    def _reward_tracking_lin_vel_y(self):
        return torch.exp(
            -torch.square(self.nav_commands[:, 1] - self.filtered_lin_vel[:, 1])
            / self.cfg["rewards"]["tracking_sigma"]
        )

    def _reward_tracking_ang_vel(self):
        return torch.exp(
            -torch.square(self.nav_commands[:, 2] - self.filtered_ang_vel[:, 2])
            / self.cfg["rewards"]["tracking_sigma"]
        )

    # ------------------------------------------------------------------ #
    #  New reward: proximity of robot body to ball                          #
    # ------------------------------------------------------------------ #

    def _reward_nav_ball_proximity(self):
        """
        Rewards the robot for being close to the ball.
        Active during Stage 1 (approach); zeroed by stage multiplier in stages 0 and 2.
        """
        dist = torch.norm(self.ball_pos[:, :2] - self.base_pos[:, :2], dim=-1)
        sigma = self.cfg["curriculum"].get("nav_ball_proximity_sigma", 1.5)
        return torch.exp(-dist / sigma)
