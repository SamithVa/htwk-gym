import torch

from isaacgym.torch_utils import get_euler_xyz, torch_rand_float

from envs.T1.kicking_robust_bilateral import KickingRobustBilateral
from utils.utils import apply_randomization


class KickingFromWalk(KickingRobustBilateral):

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        # Initialize gait_frequency to a walking-like value so the oscillator
        # is active at episode start, matching handoff from a walking policy.
        freq_min = self.cfg["randomization"].get("init_gait_frequency_min", 1.5)
        freq_max = self.cfg["randomization"].get("init_gait_frequency_max", 2.2)
        self.gait_frequency[env_ids] = torch_rand_float(
            freq_min, freq_max, (len(env_ids), 1), device=self.device
        ).squeeze(1)

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        # Add forward walking velocity so the robot enters the kick zone with
        # momentum, matching the state produced by a live walking policy.
        lin_vel_cfg = self.cfg["randomization"].get("init_base_lin_vel_x")
        if lin_vel_cfg is not None:
            fwd_vel = apply_randomization(torch.zeros(n, device=self.device), lin_vel_cfg)
            _, _, yaw = get_euler_xyz(self.root_states[env_ids, 0, 3:7])
            self.root_states[env_ids, 0, 7] += fwd_vel * torch.cos(yaw)
            self.root_states[env_ids, 0, 8] += fwd_vel * torch.sin(yaw)
