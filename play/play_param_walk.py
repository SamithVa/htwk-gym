"""
# from the repo root
python play/play_param_walk.py --task T1/Parameter_Walk --checkpoint -1 --num_envs 1

"""
import csv
import os
import sys
import time

import imageio
import isaacgym  # must be first
import torch

sys.path.append(".")

from utils.runner import Runner

# ── Settings (edit these) ────────────────────────────────────────────────────
# [lin_vel_x, lin_vel_y, ang_vel_yaw, gait_frequency,
#  foot_yaw_L, foot_yaw_R, body_pitch_target, body_roll_target,
#  feet_offset_x_target, feet_offset_y_target]
FIXED_CMD  = [0.3, 0.0, 0.0, 1.9,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DURATION_S = 6.0   # seconds to run
TIMESTAMP  = time.strftime("%Y%m%d_%H%M%S")
CSV_PATH   = f"logs/play_{TIMESTAMP}.csv"
VIDEO_PATH = f"logs/play_{TIMESTAMP}.mp4"  # only saved if record_video: true in yaml
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runner = Runner(test=True)
    env, model, device = runner.env, runner.model, runner.device

    obs, _ = env.reset()
    obs = obs.to(device)

    # Freeze commands — manual_control=True makes _resample_commands a no-op
    env.manual_control = True
    cmd = torch.tensor(FIXED_CMD, dtype=torch.float, device=device)
    env.commands[:] = cmd.unsqueeze(0).expand(env.num_envs, -1)
    env.gait_frequency[:] = cmd[3]

    n_steps = int(DURATION_S / env.dt)
    os.makedirs("logs", exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        num_obs = obs.shape[-1]
        w.writerow(["step", "time_s"] + [f"obs_{i}" for i in range(num_obs)])

        for step in range(n_steps):
            # Log obs — contains gravity, ang_vel, commands, gait_phase,
            #            joint_pos, joint_vel, last_actions
            w.writerow([step, round(step * env.dt, 4)] + obs[0].cpu().tolist())

            with torch.no_grad():
                act = model.act(obs).loc  # deterministic mean

            obs, _, _, _ = env.step(act)
            obs = obs.to(device)

    print(f"Saved {n_steps} steps → {CSV_PATH}")

    # Save video if the env collected camera frames (requires record_video: true in yaml)
    if hasattr(env, "camera_frames") and len(env.camera_frames) > 0:
        with imageio.get_writer(VIDEO_PATH, fps=int(1.0 / env.dt)) as writer:
            for frame in env.camera_frames:
                writer.append_data(frame)
        print(f"Saved video → {VIDEO_PATH}")