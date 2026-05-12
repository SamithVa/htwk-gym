"""
Play Kicking_Robust_Bilateral policy with obs CSV logging.

Obs layout (44 dims):
  [0:3]   projected gravity
  [3:6]   base angular velocity (local)
  [6:8]   relative ball position XY (robot frame)
  [8:20]  joint positions - default (12 joints)
  [20:32] joint velocities (12 joints)
  [32:44] last actions (12 joints)

Run:
python play/play_kick.py --task T1/Kicking_Robust_Bilateral --checkpoint -1 --num_envs 1 --headless True
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
DURATION_S = 4.0
TIMESTAMP  = time.strftime("%Y%m%d_%H%M%S")
CSV_PATH   = f"logs/kick_{TIMESTAMP}.csv"
VIDEO_PATH = f"logs/kick_{TIMESTAMP}.mp4"
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runner = Runner(test=True)
    env, model, device = runner.env, runner.model, runner.device

    # Fix ball to same position as sim2sim: 0.35 m straight ahead, no randomization
    env.cfg["randomization"]["ball_init_angle_min"] = 0.0
    env.cfg["randomization"]["ball_init_angle_max"] = 0.0
    env.cfg["randomization"]["ball_init_dist_min"]  = 0.35
    env.cfg["randomization"]["ball_init_dist_max"]  = 0.35

    # Match sim2sim_kick.py camera: distance=2.5, elevation=-20°, azimuth=135°
    # base_task.py: cam_pos = robot_pos + viewer.pos  (set before reset so first render picks it up)
    import math as _math
    _d, _el, _az = 2.5, 20, 315  # 315° = front of robot (+x, -y diagonal)
    env.cfg["viewer"]["pos"] = [
        _d * _math.cos(_math.radians(_el)) * _math.cos(_math.radians(_az)),  # ≈ -1.66
        _d * _math.cos(_math.radians(_el)) * _math.sin(_math.radians(_az)),  # ≈ +1.66
        _d * _math.sin(_math.radians(_el)),                                   # ≈ +0.855
    ]

    obs, _ = env.reset()
    obs = obs.to(device)

    n_steps = int(DURATION_S / env.dt)
    os.makedirs("logs", exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        num_obs = obs.shape[-1]
        w.writerow(["step", "time_s"] + [f"obs_{i}" for i in range(num_obs)])

        for step in range(n_steps):
            w.writerow([step, round(step * env.dt, 4)] + obs[0].cpu().tolist())

            with torch.no_grad():
                act = model.act(obs).loc  # deterministic mean

            obs, _, done, _ = env.step(act)
            obs = obs.to(device)

            if done[0]:
                break

    print(f"Saved {step + 1} steps → {CSV_PATH}")

    if hasattr(env, "camera_frames") and len(env.camera_frames) > 0:
        with imageio.get_writer(VIDEO_PATH, fps=int(1.0 / env.dt)) as writer:
            for frame in env.camera_frames:
                writer.append_data(frame)
        print(f"Saved video → {VIDEO_PATH}")
