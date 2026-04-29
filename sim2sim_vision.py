"""Sim2sim for T1: walk → kick → walk, using head RGBD camera for ball position.

Walk:  Parameter_Walk policy steers toward ball (yaw from camera bearing).
Kick:  Kicking_Robust policy; ball position estimated from head RGBD camera.
After kick detected (ball speed threshold), returns to walk mode.
"""

import argparse
import glob
import math
import os
import time
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import yaml


ISAAC_DOF_NAMES = (
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)

BALL_RADIUS       = 0.075
BASE_START_HEIGHT = 0.70
IDENTITY_QUAT     = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

MODE_WALK = "WALK"
MODE_STOP = "STOP"   # Stand still to match kick-policy training init state
MODE_KICK = "KICK"

KICK_DETECT_SPEED   = 0.5   # m/s — ball speed threshold to declare kick happened
POST_KICK_HOLD      = 100   # control steps to keep running kick policy after kick (~5 s at 20 Hz)
MIN_BASE_Z_STABLE   = 0.55  # m — minimum base height to consider robot stable enough to return to walk
STOP_LIN_VEL_THRESH = 0.1   # m/s — max base linear speed to consider "stopped"
STOP_ANG_VEL_THRESH = 0.3   # rad/s — max base angular speed to consider "stopped"
STOP_MAX_STEPS      = 40    # cap how long we wait in STOP mode before forcing kick (~2s)

# Head camera
CAM_H, CAM_W     = 240, 424
MIN_BALL_PIXELS   = 10
MAX_BALL_DEPTH    = 5.0
HEAD_KP_WALK, HEAD_KD_WALK = 3.0, 1.0     # gentle during walk — avoids jitter/oscillation
HEAD_KP_KICK, HEAD_KD_KICK = 20.0, 2.0    # stiff during stop/kick — holds head at training pose
HEAD_YAW_GAIN     = 0.008   # rad per normalised pixel error (lower = smoother)
HEAD_PITCH_GAIN   = 0.006
HEAD_TARGET_ALPHA = 0.3     # EMA factor for head target smoothing (0 = no update, 1 = instant)
HEAD_SEARCH_AMP   = 0.3     # rad — narrower yaw sweep
HEAD_SEARCH_PITCH = 0.15    # rad
HEAD_SEARCH_SPEED = 0.15    # rad/s


@dataclass
class ModelMaps:
    qpos_idx: np.ndarray
    qvel_idx: np.ndarray
    actuator_idx: np.ndarray
    ball_qpos_adr: int
    ball_qvel_adr: int
    head_yaw_qpos_adr: int
    head_yaw_qvel_adr: int
    head_pitch_qpos_adr: int
    head_pitch_qvel_adr: int
    head_yaw_ctrl_id: int
    head_pitch_ctrl_id: int


@dataclass
class RobotState:
    base_pos: np.ndarray
    base_quat_wxyz: np.ndarray
    base_ang_vel_local: np.ndarray
    proj_gravity: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    ball_pos: np.ndarray   # ground-truth from physics
    dist_xy: float
    ball_speed: float
    base_lin_speed_xy: float
    base_ang_speed: float


@dataclass
class EpisodeState:
    mode: str
    last_actions: np.ndarray
    gait_phase: float
    walk_command: np.ndarray
    post_kick_hold: int        = 0     # steps remaining in post-kick hold before returning to walk
    stop_steps: int            = 0     # steps spent in STOP mode (cap to prevent deadlock)
    head_search_phase: float = 0.0
    head_yaw_target: float   = 0.0
    head_pitch_target: float = 0.3


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(pattern):
    matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found: {pattern}")
    return matches[-1]


def quat_rotate_inverse_wxyz(q_wxyz, v):
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], dtype=np.float32)
    v  = np.asarray(v, dtype=np.float32)
    return v * (2.0 * w * w - 1.0) - np.cross(qv, v) * (2.0 * w) + qv * (2.0 * np.dot(qv, v))


def yaw_to_wxyz_quat(yaw):
    h = yaw * 0.5
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)], dtype=np.float32)


# ── Scene XML ──────────────────────────────────────────────────────────────────

def build_scene_xml(robot_xml_path, ball_x, ball_y):
    with open(robot_xml_path) as f:
        xml = f.read()

    xml = xml.replace("</worldbody>", f"""
        <body name="ball" pos="{ball_x} {ball_y} {BALL_RADIUS}">
            <freejoint/>
            <geom name="ball" type="sphere" size="{BALL_RADIUS}" rgba="0.9 0.1 0.1 1"
                  mass="0.2" friction="1 0.05 0.001" condim="4"/>
        </body>
    </worldbody>""", 1)

    xml = xml.replace("<size njmax=", """
    <visual>
        <global offwidth="1280" offheight="720"/>
    </visual>
    <size njmax=""", 1)

    # Replace static H1/H2 geoms with articulated head + camera
    head_xml = """\
            <body name="H1" pos="0.0625 0 0.243">
                <inertial pos="-0.000508 -0.001403 0.057432" quat="0.763881 -0.0132172 0.00173419 0.645219" mass="0.44391" diaginertia="0.000241549 0.00022351 0.000149941"/>
                <joint name="AAHead_yaw" pos="0 0 0" axis="0 0 1" limited="true" range="-1.57 1.57"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.4 0.4 0.4 1.0" mesh="H1"/>
                <body name="H2" pos="0 0 0.06185">
                    <inertial pos="0.007802 0.001262 0.098631" quat="0.988453 0.106172 -0.0745686 -0.0782786" mass="0.631019" diaginertia="0.00203553 0.00192467 0.00172381"/>
                    <joint name="Head_pitch" pos="0 0 0" axis="0 1 0" limited="true" range="-0.35 1.22"/>
                    <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.4 0.4 0.4 1.0" mesh="H2"/>
                    <camera name="head_cam" pos="0.09 0 0.11" xyaxes="0 -1 0 0 0 1" fovy="70"/>
                </body>
            </body>"""
    xml = xml.replace(
        '            <geom pos="0.0625 0 0.243" type="mesh" contype="0" conaffinity="0" group="1" rgba="0.4 0.4 0.4 1.0" mesh="H1" />\n'
        '            <geom pos="0.0625 0 0.30485" type="mesh" contype="0" conaffinity="0" group="1" rgba="0.4 0.4 0.4 1.0" mesh="H2" />',
        head_xml, 1,
    )
    xml = xml.replace(
        '<motor name="Left_Hip_Pitch"',
        '        <motor name="AAHead_yaw" joint="AAHead_yaw" ctrlrange="-7.0 7.0" ctrllimited="true"/>\n'
        '        <motor name="Head_pitch" joint="Head_pitch" ctrlrange="-7.0 7.0" ctrllimited="true"/>\n'
        '        <motor name="Left_Hip_Pitch"',
        1,
    )
    return xml


# ── Model setup ────────────────────────────────────────────────────────────────

def build_model_maps(model):
    qpos_idx    = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.int64)
    qvel_idx    = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.int64)
    actuator_idx = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.int64)

    for i, name in enumerate(ISAAC_DOF_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if jid < 0: raise ValueError(f"Joint {name} not found")
        if aid < 0: raise ValueError(f"Actuator {name} not found")
        qpos_idx[i]     = model.jnt_qposadr[jid]
        qvel_idx[i]     = model.jnt_dofadr[jid]
        actuator_idx[i] = aid

    def _jnt(n):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0: raise ValueError(f"Joint {n} not found")
        return jid

    def _act(n):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if aid < 0: raise ValueError(f"Actuator {n} not found")
        return aid

    ball_bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_jadr = model.body_jntadr[ball_bid]
    yaw_jid   = _jnt("AAHead_yaw")
    pitch_jid = _jnt("Head_pitch")

    return ModelMaps(
        qpos_idx=qpos_idx, qvel_idx=qvel_idx, actuator_idx=actuator_idx,
        ball_qpos_adr=model.jnt_qposadr[ball_jadr],
        ball_qvel_adr=model.jnt_dofadr[ball_jadr],
        head_yaw_qpos_adr=model.jnt_qposadr[yaw_jid],
        head_yaw_qvel_adr=model.jnt_dofadr[yaw_jid],
        head_pitch_qpos_adr=model.jnt_qposadr[pitch_jid],
        head_pitch_qvel_adr=model.jnt_dofadr[pitch_jid],
        head_yaw_ctrl_id=_act("AAHead_yaw"),
        head_pitch_ctrl_id=_act("Head_pitch"),
    )


def build_defaults(cfg):
    defaults = cfg["init_state"]["default_joint_angles"]
    out = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.float32)
    for i, name in enumerate(ISAAC_DOF_NAMES):
        for key, val in defaults.items():
            if key != "default" and key in name:
                out[i] = val
                break
        else:
            out[i] = defaults["default"]
    return out


def build_pd_gains(cfg):
    stiff = cfg["control"]["stiffness"]
    damp  = cfg["control"]["damping"]
    kp = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.float32)
    kd = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.float32)
    for i, name in enumerate(ISAAC_DOF_NAMES):
        for key in stiff:
            if key in name:
                kp[i] = stiff[key]
                kd[i] = damp[key]
                break
    return kp, kd


def build_cam_intrinsics(model, cam_id):
    fy = (CAM_H / 2.0) / math.tan(math.radians(model.cam_fovy[cam_id]) / 2.0)
    return fy, fy, CAM_W / 2.0, CAM_H / 2.0


# ── Per-step functions ─────────────────────────────────────────────────────────

def read_state(data, mm):
    base_pos  = np.array(data.qpos[0:3], dtype=np.float32)
    base_quat = np.array(data.qpos[3:7], dtype=np.float32)
    ball_pos  = np.array(data.qpos[mm.ball_qpos_adr:mm.ball_qpos_adr + 3], dtype=np.float32)
    base_lin_vel_world = np.array(data.qvel[0:3], dtype=np.float32)
    base_ang_vel_world = np.array(data.qvel[3:6], dtype=np.float32)
    return RobotState(
        base_pos=base_pos,
        base_quat_wxyz=base_quat,
        base_ang_vel_local=quat_rotate_inverse_wxyz(base_quat, base_ang_vel_world).astype(np.float32),
        proj_gravity=quat_rotate_inverse_wxyz(base_quat, [0., 0., -1.]).astype(np.float32),
        joint_pos=np.array(data.qpos[mm.qpos_idx], dtype=np.float32),
        joint_vel=np.array(data.qvel[mm.qvel_idx], dtype=np.float32),
        ball_pos=ball_pos,
        dist_xy=float(np.linalg.norm(ball_pos[:2] - base_pos[:2])),
        ball_speed=float(np.linalg.norm(data.qvel[mm.ball_qvel_adr:mm.ball_qvel_adr + 3])),
        base_lin_speed_xy=float(np.linalg.norm(base_lin_vel_world[:2])),
        base_ang_speed=float(np.linalg.norm(base_ang_vel_world)),
    )


def estimate_ball_from_camera(data, cam_id, head_rgb, depth_renderer, intrinsics):
    """Detect red ball in head RGBD and back-project to world-frame position.
    Returns float32 (3,) or None if ball is not visible.
    """
    fx, fy, cx, cy = intrinsics
    mask = (head_rgb[:, :, 0] > 200) & (head_rgb[:, :, 1] < 100) & (head_rgb[:, :, 2] < 100)
    if np.count_nonzero(mask) < MIN_BALL_PIXELS:
        return None
    ys, xs = np.where(mask)
    u, v = float(xs.mean()), float(ys.mean())

    depth_renderer.update_scene(data, camera=cam_id)
    d = float(depth_renderer.render()[int(round(v)), int(round(u))])
    if d <= 0.0 or d > MAX_BALL_DEPTH:
        return None

    p_cam     = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])
    R_world   = data.cam_xmat[cam_id].reshape(3, 3)
    ray_world = R_world @ (p_cam / np.linalg.norm(p_cam))
    return (data.cam_xpos[cam_id] + R_world @ p_cam + BALL_RADIUS * ray_world).astype(np.float32)


def track_head(data, mm, episode, head_rgb, control_dt):
    """Steer head toward red ball centroid; freeze at training pose during stop/kick."""
    # Freeze head at training pose during STOP and KICK (avoid CoM shifts / reaction torques)
    if episode.mode in (MODE_STOP, MODE_KICK):
        desired_yaw   = 0.0
        desired_pitch = 0.0
        kp, kd = HEAD_KP_KICK, HEAD_KD_KICK
    else:
        kp, kd = HEAD_KP_WALK, HEAD_KD_WALK
        mask = (head_rgb[:, :, 0] > 200) & (head_rgb[:, :, 1] < 100) & (head_rgb[:, :, 2] < 100)
        if np.count_nonzero(mask) >= MIN_BALL_PIXELS:
            episode.head_search_phase = 0.0
            ys, xs = np.where(mask)
            u, v = float(xs.mean()), float(ys.mean())
            desired_yaw   = episode.head_yaw_target   - HEAD_YAW_GAIN   * (u - CAM_W / 2.0) / (CAM_W / 2.0)
            desired_pitch = episode.head_pitch_target + HEAD_PITCH_GAIN * (v - CAM_H / 2.0) / (CAM_H / 2.0)
        else:
            episode.head_search_phase = (episode.head_search_phase + HEAD_SEARCH_SPEED * control_dt) % (2.0 * math.pi)
            desired_yaw   = HEAD_SEARCH_AMP * math.sin(episode.head_search_phase)
            desired_pitch = HEAD_SEARCH_PITCH

    # EMA-smooth the target so pixel-level noise doesn't shake the head
    episode.head_yaw_target   += HEAD_TARGET_ALPHA * (desired_yaw   - episode.head_yaw_target)
    episode.head_pitch_target += HEAD_TARGET_ALPHA * (desired_pitch - episode.head_pitch_target)
    episode.head_yaw_target   = float(np.clip(episode.head_yaw_target,   -1.57, 1.57))
    episode.head_pitch_target = float(np.clip(episode.head_pitch_target, -0.35, 1.22))

    for qpos_adr, qvel_adr, ctrl_id, target in (
        (mm.head_yaw_qpos_adr,   mm.head_yaw_qvel_adr,   mm.head_yaw_ctrl_id,   episode.head_yaw_target),
        (mm.head_pitch_qpos_adr, mm.head_pitch_qvel_adr, mm.head_pitch_ctrl_id, episode.head_pitch_target),
    ):
        tau = float(np.clip(kp * (target - data.qpos[qpos_adr]) - kd * data.qvel[qvel_adr], -7.0, 7.0))
        data.ctrl[ctrl_id] = tau


def build_walk_obs(state, episode, control_dt, args, default_dof_pos, walk_norm, cmd_scale):
    span  = max(args.slowdown_dist - args.switch_dist, 1e-6)
    scale = float(np.clip((state.dist_xy - args.switch_dist) / span, 0.0, 1.0))
    vx    = args.vx * scale
    gf    = args.gait_freq * max(0.5, scale)

    # Steer toward ball: yaw rate from bearing in robot-local frame
    ball_local = quat_rotate_inverse_wxyz(state.base_quat_wxyz, state.ball_pos - state.base_pos)
    yaw_rate   = float(np.clip(1.5 * math.atan2(float(ball_local[1]), float(ball_local[0])), -1.5, 1.5))

    episode.gait_phase       = (episode.gait_phase + control_dt * gf) % 1.0
    episode.walk_command[0]  = vx
    episode.walk_command[2]  = yaw_rate
    episode.walk_command[3]  = gf

    obs = np.concatenate([
        state.proj_gravity       * walk_norm["gravity"],
        state.base_ang_vel_local * walk_norm["ang_vel"],
        episode.walk_command     * cmd_scale,
        [math.cos(2 * math.pi * episode.gait_phase),
         math.sin(2 * math.pi * episode.gait_phase)],
        (state.joint_pos - default_dof_pos) * walk_norm["dof_pos"],
        state.joint_vel          * walk_norm["dof_vel"],
        episode.last_actions,
    ]).astype(np.float32)
    return obs, walk_norm["clip_actions"]


def build_kick_obs(state, ball_pos, episode, default_dof_pos, kick_norm):
    rel_ball = quat_rotate_inverse_wxyz(state.base_quat_wxyz, ball_pos - state.base_pos).astype(np.float32)
    obs = np.concatenate([
        state.proj_gravity       * kick_norm["gravity"],
        state.base_ang_vel_local * kick_norm["ang_vel"],
        rel_ball[:2]             * kick_norm["ball_pos"],
        (state.joint_pos - default_dof_pos) * kick_norm["dof_pos"],
        state.joint_vel          * kick_norm["dof_vel"],
        episode.last_actions,
    ]).astype(np.float32)
    return obs, kick_norm["clip_actions"]


def infer(policy, obs, clip):
    with torch.no_grad():
        return np.clip(
            policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy(),
            -clip, clip,
        ).astype(np.float32)


def apply_pd(model, data, mm, target, kp, kd, decimation):
    ctrl_range = model.actuator_ctrlrange[mm.actuator_idx]
    for _ in range(decimation):
        q  = np.array(data.qpos[mm.qpos_idx], dtype=np.float32)
        dq = np.array(data.qvel[mm.qvel_idx],  dtype=np.float32)
        data.ctrl[mm.actuator_idx] = np.clip(kp * (target - q) - kd * dq, ctrl_range[:, 0], ctrl_range[:, 1])
        mujoco.mj_step(model, data)


def render_frame(step, render_every, data, renderer, camera, writer, head_rgb):
    if step % render_every != 0:
        return
    camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.5]
    renderer.update_scene(data, camera=camera)
    frame = renderer.render().copy()
    pad = 10
    h, w = head_rgb.shape[:2]
    r0, r1 = pad, pad + h
    c0, c1 = frame.shape[1] - pad - w, frame.shape[1] - pad
    frame[r0-2:r1+2, c0-2:c1+2] = 0
    frame[r0:r1, c0:c1] = head_rgb
    writer.append_data(frame)


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--walk-ckpt",    type=str,   default=None)
    p.add_argument("--kick-ckpt",    type=str,   default=None)
    p.add_argument("--ball-dist",    type=float, default=1.0,  help="Ball distance from robot [m]")
    p.add_argument("--switch-dist",  type=float, default=0.4,  help="Distance to switch to kick [m]")
    p.add_argument("--slowdown-dist",type=float, default=0.6,  help="Distance to start slowing down [m]")
    p.add_argument("--duration",     type=float, default=10.0, help="Simulation duration [s]")
    p.add_argument("--fps",          type=int,   default=30)
    p.add_argument("--width",        type=int,   default=1280)
    p.add_argument("--height",       type=int,   default=720)
    p.add_argument("--out",          type=str,   default=None)
    p.add_argument("--vx",           type=float, default=0.3,  help="Forward walk speed [m/s]")
    p.add_argument("--gait-freq",    type=float, default=1.0,  help="Gait frequency [Hz]")
    p.add_argument("--robot-x",      type=float, default=0.0)
    p.add_argument("--robot-y",      type=float, default=0.0)
    p.add_argument("--robot-yaw",    type=float, default=0.0,  help="Robot start yaw [rad]")
    p.add_argument("--ball-angle",   type=float, default=0.0,  help="Ball angle relative to robot yaw [rad]")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    repo = os.path.dirname(os.path.abspath(__file__))
    walk_ckpt = args.walk_ckpt or find_latest_checkpoint(os.path.join(repo, "logs/T1/T1/Parameter_Walk/**/*.pt"))
    kick_ckpt = args.kick_ckpt or find_latest_checkpoint(os.path.join(repo, "logs/T1/T1/Kicking/**/*.pt"))
    out_path  = args.out or os.path.join(repo, f"videos/sim2sim_vision_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    print(f"Walk: {walk_ckpt}")
    print(f"Kick: {kick_ckpt}")

    walk_cfg  = load_yaml(os.path.join(repo, "envs/T1/Parameter_Walk.yaml"))
    kick_cfg  = load_yaml(os.path.join(repo, "envs/T1/Kicking_Robust.yaml"))
    walk_norm = walk_cfg["normalization"]
    kick_norm = kick_cfg["normalization"]

    sim_dt      = walk_cfg["sim"]["dt"]
    decimation  = walk_cfg["control"]["decimation"]
    control_dt  = sim_dt * decimation
    action_scale = walk_cfg["control"]["action_scale"]
    n_steps     = int(round(args.duration / control_dt))
    render_every = max(1, int(round(1.0 / (control_dt * args.fps))))

    walk_policy = torch.jit.load(walk_ckpt, map_location="cpu").eval()
    kick_policy = torch.jit.load(kick_ckpt, map_location="cpu").eval()

    # Ball placement relative to robot
    angle  = args.robot_yaw + args.ball_angle
    ball_x = args.robot_x + args.ball_dist * math.cos(angle)
    ball_y = args.robot_y + args.ball_dist * math.sin(angle)
    print(f"Robot: ({args.robot_x:.2f}, {args.robot_y:.2f}) yaw={args.robot_yaw:.2f}  "
          f"Ball: ({ball_x:.2f}, {ball_y:.2f})")

    # Build and write scene XML
    xml        = build_scene_xml(os.path.join(repo, "resources/T1/T1_locomotion.xml"), ball_x, ball_y)
    scene_path = os.path.join(repo, "resources/T1/_sim2sim_scene.xml")
    with open(scene_path, "w") as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_path(scene_path)
    model.opt.timestep = sim_dt
    data  = mujoco.MjData(model)
    mm    = build_model_maps(model)

    default_dof_pos = build_defaults(walk_cfg)
    kp, kd          = build_pd_gains(walk_cfg)

    cam_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    intrinsics = build_cam_intrinsics(model, cam_id)

    # Walk command scale vector
    cmd_scale = np.array([
        walk_norm["lin_vel"], walk_norm["lin_vel"], walk_norm["ang_vel"],
        walk_norm["gait_frequency"],
        walk_norm["foot_yaw"], walk_norm["foot_yaw"],
        walk_norm["body_pitch_target"], walk_norm["body_roll_target"],
        walk_norm["feet_offset_x_target"], walk_norm["feet_offset_y_target"],
    ], dtype=np.float32)

    # Initialise simulation state
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = [args.robot_x, args.robot_y, BASE_START_HEIGHT]
    data.qpos[3:7] = yaw_to_wxyz_quat(args.robot_yaw)
    data.qpos[mm.qpos_idx] = default_dof_pos
    data.qpos[mm.ball_qpos_adr:mm.ball_qpos_adr + 3]     = [ball_x, ball_y, BALL_RADIUS]
    data.qpos[mm.ball_qpos_adr + 3:mm.ball_qpos_adr + 7] = IDENTITY_QUAT
    mujoco.mj_forward(model, data)

    episode = EpisodeState(
        mode=MODE_WALK,
        last_actions=np.zeros(len(ISAAC_DOF_NAMES), dtype=np.float32),
        gait_phase=0.0,
        walk_command=np.array([args.vx, 0.0, 0.0, args.gait_freq, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera   = mujoco.MjvCamera()
    camera.type, camera.distance, camera.elevation, camera.azimuth = (
        mujoco.mjtCamera.mjCAMERA_FREE, 3.5, -20, 135
    )
    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264", quality=8)

    head_renderer_rgb   = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    head_renderer_depth = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    head_renderer_depth.enable_depth_rendering()

    print(f"Rendering {n_steps} steps → {out_path}")
    t0 = time.time()
    try:
        for step in range(n_steps):
            state = read_state(data, mm)
            t     = step * control_dt

            # Mode transitions: WALK → STOP → KICK → (WALK)
            if episode.mode == MODE_WALK and state.dist_xy <= args.switch_dist:
                episode.mode = MODE_STOP
                episode.stop_steps = 0
                print(f"[t={t:.2f}s] WALK → STOP  dist={state.dist_xy:.3f}")
            elif episode.mode == MODE_STOP:
                episode.stop_steps += 1
                stopped = (state.base_lin_speed_xy < STOP_LIN_VEL_THRESH
                           and state.base_ang_speed < STOP_ANG_VEL_THRESH)
                if stopped or episode.stop_steps >= STOP_MAX_STEPS:
                    episode.mode = MODE_KICK
                    print(f"[t={t:.2f}s] STOP → KICK  after {episode.stop_steps} steps  "
                          f"lin_v={state.base_lin_speed_xy:.3f}  ang_v={state.base_ang_speed:.3f}")
            elif episode.mode == MODE_KICK:
                if episode.post_kick_hold == 0 and state.ball_speed >= KICK_DETECT_SPEED:
                    episode.post_kick_hold = POST_KICK_HOLD
                    print(f"[t={t:.2f}s] kick detected (ball_speed={state.ball_speed:.3f}), holding {POST_KICK_HOLD} steps")
                elif episode.post_kick_hold > 0:
                    episode.post_kick_hold -= 1
                    if episode.post_kick_hold == 0 and state.base_pos[2] >= MIN_BASE_Z_STABLE:
                        episode.mode = MODE_WALK
                        print(f"[t={t:.2f}s] KICK → WALK  base_z={state.base_pos[2]:.3f}")
                    elif episode.post_kick_hold == 0:
                        # Robot not yet stable — extend hold
                        episode.post_kick_hold = 20
                        print(f"[t={t:.2f}s] base_z={state.base_pos[2]:.3f} < {MIN_BASE_Z_STABLE}, extending kick hold")

            # Render head camera once — shared by head tracker, ball estimator, and PIP
            head_renderer_rgb.update_scene(data, camera=cam_id)
            head_rgb = head_renderer_rgb.render().copy()

            if episode.mode == MODE_WALK:
                obs, clip = build_walk_obs(state, episode, control_dt, args, default_dof_pos, walk_norm, cmd_scale)
                episode.last_actions = infer(walk_policy, obs, clip)
            elif episode.mode == MODE_STOP:
                # Run walk policy with zero commands so the robot decelerates and stands still.
                # This matches the kick-policy training init state (robot at rest, ball ahead).
                episode.walk_command[0] = 0.0  # vx
                episode.walk_command[1] = 0.0  # vy
                episode.walk_command[2] = 0.0  # yaw_rate
                episode.walk_command[3] = args.gait_freq * 0.5  # low gait freq to stay planted
                episode.gait_phase = (episode.gait_phase + control_dt * episode.walk_command[3]) % 1.0
                obs = np.concatenate([
                    state.proj_gravity       * walk_norm["gravity"],
                    state.base_ang_vel_local * walk_norm["ang_vel"],
                    episode.walk_command     * cmd_scale,
                    [math.cos(2 * math.pi * episode.gait_phase),
                     math.sin(2 * math.pi * episode.gait_phase)],
                    (state.joint_pos - default_dof_pos) * walk_norm["dof_pos"],
                    state.joint_vel          * walk_norm["dof_vel"],
                    episode.last_actions,
                ]).astype(np.float32)
                episode.last_actions = infer(walk_policy, obs, walk_norm["clip_actions"])
            else:  # MODE_KICK
                # Keep gait phase ticking so walk policy gets coherent phase on return
                episode.gait_phase = (episode.gait_phase + control_dt * args.gait_freq) % 1.0

                if episode.post_kick_hold > 0:
                    # Ball has been kicked — use real physics position.
                    # frozen_ball_pos caused the robot to "see" the ball behind it as it walked
                    # past, which the kick policy was never trained for.
                    ball_pos = state.ball_pos
                else:
                    # Ball still on ground — estimate from camera, fall back to ground truth
                    cam_ball = estimate_ball_from_camera(data, cam_id, head_rgb, head_renderer_depth, intrinsics)
                    if cam_ball is not None:
                        err = np.linalg.norm(cam_ball - state.ball_pos)
                        print(f"  [t={t:.2f}s] ball vision=({cam_ball[0]:.3f},{cam_ball[1]:.3f},{cam_ball[2]:.3f})"
                              f"  gt=({state.ball_pos[0]:.3f},{state.ball_pos[1]:.3f},{state.ball_pos[2]:.3f})"
                              f"  err={err:.4f}m")
                        ball_pos = cam_ball
                    else:
                        print(f"  [t={t:.2f}s] ball not visible — using ground truth")
                        ball_pos = state.ball_pos
                obs, clip = build_kick_obs(state, ball_pos, episode, default_dof_pos, kick_norm)
                episode.last_actions = infer(kick_policy, obs, clip)

            track_head(data, mm, episode, head_rgb, control_dt)
            apply_pd(model, data, mm, default_dof_pos + action_scale * episode.last_actions, kp, kd, decimation)
            render_frame(step, render_every, data, renderer, camera, writer, head_rgb)

            if step % 50 == 0:
                print(f"  t={t:5.2f}s  {episode.mode}  dist={state.dist_xy:.3f}  z={state.base_pos[2]:.3f}")
    finally:
        writer.close()
        del renderer, head_renderer_rgb, head_renderer_depth

    print(f"Done in {time.time() - t0:.1f}s  →  {out_path}")
    print(f"Final ball: {data.qpos[mm.ball_qpos_adr:mm.ball_qpos_adr + 3]}")
    print(f"Final robot: {data.qpos[0:3]}")


if __name__ == "__main__":
    main()
