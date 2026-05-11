"""
Compare play vs sim2sim observation logs.

Walk obs layout (54 dims, default):
  [0:3]   gravity vector
  [3:6]   angular velocity
  [6:16]  commands: lin_vel_x, lin_vel_y, ang_vel_yaw, gait_freq,
                    foot_yaw_L, foot_yaw_R, body_pitch, body_roll,
                    feet_offset_x, feet_offset_y
  [16:18] gait phase (sin, cos)
  [18:30] joint positions (12 joints)
  [30:42] joint velocities (12 joints)
  [42:54] last actions (12 joints)

Kick obs layout (44 dims, use --kick):
  [0:3]   gravity vector
  [3:6]   angular velocity
  [6:8]   relative ball position XY (robot frame)
  [8:20]  joint positions (12 joints)
  [20:32] joint velocities (12 joints)
  [32:44] last actions (12 joints)

Usage:
    # walk comparison
    python analysis/plot_play_sim2sim.py \
        --play logs/play_TIMESTAMP.csv --sim2sim logs/sim2sim.csv

    # kick comparison
    python analysis/plot_play_sim2sim.py --kick \
        --play logs/kick_TIMESTAMP.csv --sim2sim logs/sim2sim_kick.csv
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

JOINT_NAMES = [
    "L_Hip_Roll", "L_Hip_Yaw", "L_Hip_Pitch",
    "L_Knee_Pitch", "L_Ankle_Pitch", "L_Ankle_Roll",
    "R_Hip_Roll", "R_Hip_Yaw", "R_Hip_Pitch",
    "R_Knee_Pitch", "R_Ankle_Pitch", "R_Ankle_Roll",
]

COMMAND_NAMES = [
    "lin_vel_x", "lin_vel_y", "ang_vel_yaw", "gait_freq",
    "foot_yaw_L", "foot_yaw_R", "body_pitch", "body_roll",
    "feet_offset_x", "feet_offset_y",
]

# Obs index ranges
IDX_GRAVITY   = slice(0, 3)
IDX_ANG_VEL   = slice(3, 6)
IDX_COMMANDS  = slice(6, 16)
IDX_PHASE     = slice(16, 18)
IDX_JOINT_POS = slice(18, 30)
IDX_JOINT_VEL = slice(30, 42)
IDX_ACTIONS   = slice(42, 54)


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    obs = df[obs_cols].values
    t = df["time_s"].values
    return t, obs


def savefig(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")


def plot_group(t_p, data_p, t_s, data_s, names, title, ylabel, out_dir, filename, ncols=3):
    n = len(names)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=False)
    axes = np.array(axes).reshape(nrows, ncols)
    fig.suptitle(title, fontsize=13)

    for i, name in enumerate(names):
        ax = axes[i // ncols, i % ncols]
        ax.plot(t_p, data_p[:, i], label="play",    color="steelblue",  linewidth=1.2)
        ax.plot(t_s, data_s[:, i], label="sim2sim", color="darkorange", linewidth=1.2, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("time [s]", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.tight_layout()
    savefig(fig, out_dir, filename)
    plt.close(fig)


def plot_scalar_group(t_p, data_p, t_s, data_s, names, title, out_dir, filename):
    """Single row of scalar signals (gravity, ang_vel, gait_phase, …)."""
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)

    for i, name in enumerate(names):
        ax = axes[i]
        ax.plot(t_p, data_p[:, i], label="play",    color="steelblue",  linewidth=1.2)
        ax.plot(t_s, data_s[:, i], label="sim2sim", color="darkorange", linewidth=1.2, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("time [s]", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, out_dir, filename)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play",    required=True, help="play CSV path")
    parser.add_argument("--sim2sim", required=True, help="sim2sim CSV path")
    parser.add_argument("--out",     default="logs/plots", help="output directory")
    parser.add_argument("--kick",    action="store_true",
                        help="Use kick obs layout (44 dims) instead of walk (54 dims)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    t_p, obs_p = load(args.play)
    t_s, obs_s = load(args.sim2sim)

    print(f"play:    {len(t_p)} steps  ({t_p[-1]:.2f}s)  obs_dim={obs_p.shape[1]}")
    print(f"sim2sim: {len(t_s)} steps  ({t_s[-1]:.2f}s)  obs_dim={obs_s.shape[1]}")

    # Clip both to the shorter trajectory so all plots share the same time axis
    min_len = min(len(t_p), len(t_s))
    t_p, obs_p = t_p[:min_len], obs_p[:min_len]
    t_s, obs_s = t_s[:min_len], obs_s[:min_len]
    print(f"plotting: {min_len} steps  ({t_p[-1]:.2f}s)")

    # Kick obs layout (44 dims)
    KICK_IDX_GRAVITY   = slice(0, 3)
    KICK_IDX_ANG_VEL   = slice(3, 6)
    KICK_IDX_BALL_XY   = slice(6, 8)
    KICK_IDX_JOINT_POS = slice(8, 20)
    KICK_IDX_JOINT_VEL = slice(20, 32)
    KICK_IDX_ACTIONS   = slice(32, 44)

    # ── 1. Gravity vector ──────────────────────────────────────────────────────
    plot_scalar_group(
        t_p, obs_p[:, IDX_GRAVITY],
        t_s, obs_s[:, IDX_GRAVITY],
        names=["grav_x", "grav_y", "grav_z"],
        title="Projected Gravity Vector",
        out_dir=args.out, filename="gravity.png",
    )

    # ── 2. Angular velocity ────────────────────────────────────────────────────
    plot_scalar_group(
        t_p, obs_p[:, IDX_ANG_VEL],
        t_s, obs_s[:, IDX_ANG_VEL],
        names=["ang_vel_x", "ang_vel_y", "ang_vel_z"],
        title="Base Angular Velocity",
        out_dir=args.out, filename="ang_vel.png",
    )

    if args.kick:
        # ── 3k. Relative ball position ─────────────────────────────────────────
        plot_scalar_group(
            t_p, obs_p[:, KICK_IDX_BALL_XY],
            t_s, obs_s[:, KICK_IDX_BALL_XY],
            names=["ball_rel_x", "ball_rel_y"],
            title="Relative Ball Position (robot frame)",
            out_dir=args.out, filename="ball_pos.png",
        )
        joint_pos_idx = KICK_IDX_JOINT_POS
        joint_vel_idx = KICK_IDX_JOINT_VEL
        actions_idx   = KICK_IDX_ACTIONS
    else:
        # ── 3w. Commands (sanity check — should be flat lines) ─────────────────
        plot_group(
            t_p, obs_p[:, IDX_COMMANDS],
            t_s, obs_s[:, IDX_COMMANDS],
            names=COMMAND_NAMES,
            title="Commands (should be constant)", ylabel="",
            out_dir=args.out, filename="commands.png",
        )

        # ── 4w. Gait phase ─────────────────────────────────────────────────────
        plot_scalar_group(
            t_p, obs_p[:, IDX_PHASE],
            t_s, obs_s[:, IDX_PHASE],
            names=["phase_sin", "phase_cos"],
            title="Gait Phase",
            out_dir=args.out, filename="gait_phase.png",
        )
        joint_pos_idx = IDX_JOINT_POS
        joint_vel_idx = IDX_JOINT_VEL
        actions_idx   = IDX_ACTIONS

    # ── 5. Joint positions ────────────────────────────────────────────────────
    plot_group(
        t_p, obs_p[:, joint_pos_idx],
        t_s, obs_s[:, joint_pos_idx],
        names=JOINT_NAMES,
        title="Joint Positions", ylabel="rad",
        out_dir=args.out, filename="joint_pos.png",
    )

    # ── 6. Joint velocities ───────────────────────────────────────────────────
    plot_group(
        t_p, obs_p[:, joint_vel_idx],
        t_s, obs_s[:, joint_vel_idx],
        names=JOINT_NAMES,
        title="Joint Velocities", ylabel="rad/s",
        out_dir=args.out, filename="joint_vel.png",
    )

    # ── 7. Last actions ───────────────────────────────────────────────────────
    plot_group(
        t_p, obs_p[:, actions_idx],
        t_s, obs_s[:, actions_idx],
        names=JOINT_NAMES,
        title="Last Actions (policy output)", ylabel="rad",
        out_dir=args.out, filename="actions.png",
    )

    # ── 8. Summary: RMS error per joint position ──────────────────────────────
    rms = np.sqrt(np.mean((obs_p[:, joint_pos_idx] - obs_s[:, joint_pos_idx]) ** 2, axis=0))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(JOINT_NAMES, rms, color="steelblue")
    ax.set_title("RMS Error: Joint Positions (play vs sim2sim)")
    ax.set_ylabel("rad")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, args.out, "rms_joint_pos.png")
    plt.close(fig)

    print(f"\nAll plots saved to {args.out}/")


if __name__ == "__main__":
    main()
