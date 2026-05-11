"""sim2sim with head RGBD camera: walk → kick using vision-only ball estimation.

Walk mode : camera bearing → v_yaw, distance → vx, pixel elevation → head_pitch
Kick mode : vision-estimated ball world position fed to kick policy
            (falls back to ground-truth physics if camera loses the ball)

A white bounding box is drawn around the detected ball in the PIP inset.
"""

import argparse
import glob
import math
import os
import time
from dataclasses import dataclass, field

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

MODE_WALK    = "WALK"
MODE_KICK    = "KICK"
MODE_RECOVER = "RECOVER"
MODE_FALLEN  = "FALLEN"

DEPLOY_LEG_JOINT_OFFSET = 11  # leg joints occupy slots 11-22 in the 23-motor deploy arrays

KICK_BALL_THRESHOLD = 0.05  # m — ball must travel this far from kick-start to count as kicked
KICK_SETTLE_STEPS   = 10   # control steps to hold kick policy after ball departs
RECOVER_STEPS       = 30   # control steps to interpolate back to default pose
FALLEN_GRAVITY_Z    = 0.3  # -proj_gravity[2] below this → robot is fallen (overridable via --fallen-thresh)

# Head camera
CAM_H, CAM_W    = 240, 424
MIN_BALL_PIXELS  = 10
MAX_BALL_DEPTH   = 5.0

# Head tracking: pixel error → joint target (normalised by half-image size)
HEAD_YAW_GAIN    = 0.1    # rad per unit normalised horizontal error
HEAD_PITCH_GAIN  = 0.4    # rad per unit normalised vertical error
HEAD_PITCH_BASE  = 0.3    # default downward pitch (rad) when tracking
HEAD_TARGET_ALPHA = 0.3   # EMA smoothing factor for head targets

# Search sweep when ball is not visible
HEAD_SEARCH_AMP   = 0.0   # rad — yaw sweep amplitude
HEAD_SEARCH_PITCH = -0.2  # rad — tilt head up while searching (negative = look up)
HEAD_SEARCH_SPEED = 0.15  # rad/s — sweep rate


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class ControlParams:
    sim_dt: float
    decimation: int
    control_dt: float
    action_scale: float
    n_steps: int
    render_every: int


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
class Policies:
    walk: torch.jit.ScriptModule
    kick: torch.jit.ScriptModule


@dataclass
class RobotState:
    base_pos: np.ndarray
    base_quat_wxyz: np.ndarray
    base_ang_vel_local: np.ndarray
    proj_gravity: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    ball_pos: np.ndarray
    dist_xy: float
    ball_speed: float


@dataclass
class EpisodeState:
    mode: str
    last_actions: np.ndarray
    gait_phase: float
    walk_command: np.ndarray
    kick_ball_start: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    kick_settle_remaining: int = 0
    recover_start_pos: np.ndarray = field(default_factory=lambda: np.zeros(len(ISAAC_DOF_NAMES), dtype=np.float32))
    recover_step: int = 0
    head_yaw_target: float  = 0.0
    head_pitch_target: float = HEAD_PITCH_BASE
    head_search_phase: float = 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(pattern):
    matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No checkpoint: {pattern}")
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
    """Inject a free ball and an articulated head with RGBD camera into the robot XML."""
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

    # Replace the two static H1/H2 geoms with articulated head bodies + camera
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


def build_defaults(deploy_cfg):
    qpos = np.array(deploy_cfg["common"]["default_qpos"], dtype=np.float32)
    return qpos[DEPLOY_LEG_JOINT_OFFSET:DEPLOY_LEG_JOINT_OFFSET + len(ISAAC_DOF_NAMES)]


def build_pd_gains(deploy_cfg):
    s, e = DEPLOY_LEG_JOINT_OFFSET, DEPLOY_LEG_JOINT_OFFSET + len(ISAAC_DOF_NAMES)
    kp = np.array(deploy_cfg["common"]["stiffness"], dtype=np.float32)[s:e]
    kd = np.array(deploy_cfg["common"]["damping"], dtype=np.float32)[s:e]
    return kp, kd


def build_cam_intrinsics(model, cam_id):
    fy = (CAM_H / 2.0) / math.tan(math.radians(model.cam_fovy[cam_id]) / 2.0)
    return fy, fy, CAM_W / 2.0, CAM_H / 2.0


# ── State ──────────────────────────────────────────────────────────────────────

def read_state(data, mm):
    base_pos  = np.array(data.qpos[0:3], dtype=np.float32)
    base_quat = np.array(data.qpos[3:7], dtype=np.float32)
    ball_pos  = np.array(data.qpos[mm.ball_qpos_adr:mm.ball_qpos_adr + 3], dtype=np.float32)
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
    )


# ── Vision ─────────────────────────────────────────────────────────────────────

def detect_ball(rgb):
    """Detect the red ball by colour thresholding.

    Returns (u, v, (x0, y0, x1, y1)) — pixel centroid and tight bounding box — or None.
    """
    mask = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] < 100) & (rgb[:, :, 2] < 100)
    if np.count_nonzero(mask) < MIN_BALL_PIXELS:
        return None
    ys, xs = np.where(mask)
    u   = float(xs.mean())
    v   = float(ys.mean())
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return u, v, bbox


def estimate_ball_world(data, cam_id, u, v, depth_renderer, intrinsics):
    """Back-project pixel (u, v) + depth to a world-frame ball centre.

    Returns float32 (3,) or None if depth is invalid.
    """
    fx, fy, cx, cy = intrinsics
    depth_renderer.update_scene(data, camera=cam_id)
    d = float(depth_renderer.render()[int(round(v)), int(round(u))])
    if d <= 0.0 or d > MAX_BALL_DEPTH:
        return None
    p_cam     = np.array([(u - cx) * d / fx, -(v - cy) * d / fy, -d])
    R_world   = data.cam_xmat[cam_id].reshape(3, 3)
    ray_world = R_world @ (p_cam / np.linalg.norm(p_cam))
    return (data.cam_xpos[cam_id] + R_world @ p_cam + BALL_RADIUS * ray_world).astype(np.float32)


def draw_bbox(rgb, bbox):
    """Draw a red bounding box on a copy of the RGB image."""
    frame = rgb.copy()
    x0, y0, x1, y1 = bbox
    pad = 3
    x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
    x1, y1 = min(x1 + pad, CAM_W - 1), min(y1 + pad, CAM_H - 1)
    white = [255, 255, 255]
    frame[y0:y1+1, x0:x0+2]   = white  # left edge
    frame[y0:y1+1, x1-1:x1+1] = white  # right edge
    frame[y0:y0+2, x0:x1+1]   = white  # top edge
    frame[y1-1:y1+1, x0:x1+1] = white  # bottom edge
    return frame


# ── Head control ───────────────────────────────────────────────────────────────

def apply_head_control(data, mm, episode, control_dt, detection,
                       head_kp_walk, head_kd_walk, active_during_kick=False):
    """Steer the head toward the detected ball during walk.

    During KICK and RECOVER the head motors are zeroed by default so they produce no
    reaction torques on the base — the kick policy was trained without head joints.
    Pass active_during_kick=True to keep stiff PD active (useful for comparison runs).
    """
    if episode.mode in (MODE_KICK, MODE_RECOVER) and not active_during_kick:
        data.ctrl[mm.head_yaw_ctrl_id]   = 0.0
        data.ctrl[mm.head_pitch_ctrl_id] = 0.0
        return
    else:
        kp, kd = head_kp_walk, head_kd_walk
        episode.head_yaw_target = 0.0  # yaw is always fixed at centre
        if detection is not None:
            _, v, _ = detection
            # Only pitch adjusts (via EMA) to keep ball vertically centered
            desired_pitch = episode.head_pitch_target + HEAD_PITCH_GAIN * (v - CAM_H / 2.0) / (CAM_H / 2.0)
            episode.head_pitch_target += HEAD_TARGET_ALPHA * (desired_pitch - episode.head_pitch_target)
        else:
            # Tilt head up while searching
            desired_pitch = HEAD_SEARCH_PITCH
            episode.head_pitch_target += HEAD_TARGET_ALPHA * (desired_pitch - episode.head_pitch_target)
    episode.head_yaw_target   = float(np.clip(episode.head_yaw_target,   -1.57, 1.57))
    episode.head_pitch_target = float(np.clip(episode.head_pitch_target, -0.35, 1.22))

    for qpos_adr, qvel_adr, ctrl_id, target in (
        (mm.head_yaw_qpos_adr,   mm.head_yaw_qvel_adr,   mm.head_yaw_ctrl_id,   episode.head_yaw_target),
        (mm.head_pitch_qpos_adr, mm.head_pitch_qvel_adr, mm.head_pitch_ctrl_id, episode.head_pitch_target),
    ):
        tau = float(np.clip(kp * (target - data.qpos[qpos_adr]) - kd * data.qvel[qvel_adr], -7.0, 7.0))
        data.ctrl[ctrl_id] = tau


# ── Observations ───────────────────────────────────────────────────────────────

def build_walk_obs(state, episode, control_dt, args, default_dof_pos, walk_norm, cmd_scale, detection=None):
    """Walk observation: speed ramps down as robot nears ball; yaw rate steers toward ball."""
    span  = max(args.slowdown_dist - args.switch_dist, 1e-6)
    scale = float(np.clip((state.dist_xy - args.switch_dist) / span, 0.0, 1.0))
    vx    = args.vx * scale
    gf    = args.gait_freq * max(0.5, scale)

    if detection is not None:
        u, _, _ = detection
        # Ball right of centre (u > CAM_W/2) → turn right (negative yaw rate)
        yaw_rate = float(np.clip(-1.5 * (u - CAM_W / 2.0) / (CAM_W / 2.0), -1.5, 1.5))
    else:
        ball_local = quat_rotate_inverse_wxyz(state.base_quat_wxyz, state.ball_pos - state.base_pos)
        yaw_rate   = float(np.clip(1.5 * math.atan2(float(ball_local[1]), float(ball_local[0])), -1.5, 1.5))

    episode.gait_phase      = (episode.gait_phase + control_dt * gf) % 1.0
    episode.walk_command[0] = vx
    episode.walk_command[2] = yaw_rate
    episode.walk_command[3] = gf

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


# ── Actuation ──────────────────────────────────────────────────────────────────

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
        data.ctrl[mm.actuator_idx] = np.clip(kp * (target - q) - kd * dq,
                                             ctrl_range[:, 0], ctrl_range[:, 1])
        mujoco.mj_step(model, data)


# ── Rendering ──────────────────────────────────────────────────────────────────

def render_frame(step, render_every, data, renderer, camera, writer, head_rgb):
    """Render the main view with the head-camera PIP inset in the top-right corner."""
    if step % render_every != 0:
        return
    camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.5]
    renderer.update_scene(data, camera=camera)
    frame = renderer.render().copy()
    pad = 10
    h, w = head_rgb.shape[:2]
    r0, r1 = pad, pad + h
    c0, c1 = frame.shape[1] - pad - w, frame.shape[1] - pad
    frame[r0-2:r1+2, c0-2:c1+2] = 0       # black border
    frame[r0:r1, c0:c1] = head_rgb
    writer.append_data(frame)


# ── Episode loop ───────────────────────────────────────────────────────────────

def run_episode(model, data, mm, policies, episode, cp, args,
                walk_norm, kick_norm, default_dof_pos, kp, kd,
                head_kp_walk, head_kd_walk,
                renderer, camera, writer, cam_id, intrinsics,
                head_renderer_rgb, head_renderer_depth,
                cam_save_dir=None):
    cam_save_dir = cam_save_dir or args.cam_save_dir
    if cam_save_dir:
        os.makedirs(cam_save_dir, exist_ok=True)
    detection_save_count = 0

    cmd_scale = np.array([
        walk_norm["lin_vel"], walk_norm["lin_vel"], walk_norm["ang_vel"],
        walk_norm["gait_frequency"],
        walk_norm["foot_yaw"], walk_norm["foot_yaw"],
        walk_norm["body_pitch_target"], walk_norm["body_roll_target"],
        walk_norm["feet_offset_x_target"], walk_norm["feet_offset_y_target"],
    ], dtype=np.float32)

    for step in range(cp.n_steps):
        state = read_state(data, mm)
        t     = step * cp.control_dt

        # ── Fall detection ────────────────────────────────────────────────────
        if episode.mode not in (MODE_FALLEN,) and -state.proj_gravity[2] < args.fallen_thresh:
            episode.mode = MODE_FALLEN
            print(f"[t={t:.2f}s] FALLEN (proj_gravity_z={state.proj_gravity[2]:.3f})")

        # ── Head camera: detect ball, annotate PIP ────────────────────────────
        head_renderer_rgb.update_scene(data, camera=cam_id)
        head_rgb  = head_renderer_rgb.render().copy()
        detection = detect_ball(head_rgb)
        head_rgb_annotated = draw_bbox(head_rgb, detection[2]) if detection is not None else head_rgb

        # ── Save head-camera image when ball is detected ──────────────────────
        if cam_save_dir and detection is not None:
            if detection_save_count % args.cam_save_every == 0:
                fname = os.path.join(cam_save_dir, f"cam_{step:05d}_t{t:.3f}s.png")
                imageio.imwrite(fname, head_rgb_annotated)
            detection_save_count += 1

        # ── Mode transitions (plain `if` blocks like sim2sim.py so RECOVER can
        #    run on the same step as KICK→RECOVER) ──────────────────────────────
        if episode.mode == MODE_WALK and state.dist_xy <= args.switch_dist:
            episode.mode = MODE_KICK
            episode.kick_ball_start = state.ball_pos.copy()
            episode.kick_settle_remaining = 0
            print(f"[t={t:.2f}s] WALK → KICK  dist={state.dist_xy:.3f}")

        if episode.mode == MODE_KICK:
            ball_moved = np.linalg.norm(state.ball_pos[:2] - episode.kick_ball_start[:2])
            if ball_moved > KICK_BALL_THRESHOLD and episode.kick_settle_remaining == 0:
                episode.kick_settle_remaining = KICK_SETTLE_STEPS
                print(f"[t={t:.2f}s] kick detected (ball_moved={ball_moved:.3f} m), "
                      f"settling for {KICK_SETTLE_STEPS} steps")
            if episode.kick_settle_remaining > 0:
                episode.kick_settle_remaining -= 1
                if episode.kick_settle_remaining == 0:
                    episode.mode = MODE_RECOVER
                    episode.recover_start_pos = state.joint_pos.copy()
                    episode.recover_step = 0
                    print(f"[t={t:.2f}s] KICK → RECOVER")

        if episode.mode == MODE_RECOVER:
            alpha = min(episode.recover_step / RECOVER_STEPS, 1.0)
            dof_target = (1.0 - alpha) * episode.recover_start_pos + alpha * default_dof_pos
            episode.recover_step += 1
            if episode.recover_step >= RECOVER_STEPS:
                episode.mode = MODE_WALK
                print(f"[t={t:.2f}s] RECOVER → WALK")
            apply_head_control(data, mm, episode, cp.control_dt, detection,
                               head_kp_walk, head_kd_walk,
                               active_during_kick=args.active_head_during_kick)
            apply_pd(model, data, mm, dof_target, kp, kd, cp.decimation)
            render_frame(step, cp.render_every, data, renderer, camera, writer, head_rgb_annotated)
            if step % 50 == 0:
                print(f"  t={t:5.2f}s  {episode.mode}  dist={state.dist_xy:.3f}  z={state.base_pos[2]:.3f}")
            continue

        # ── Policy inference ──────────────────────────────────────────────────
        if episode.mode == MODE_FALLEN:
            data.ctrl[:] = 0.0
            render_frame(step, cp.render_every, data, renderer, camera, writer, head_rgb_annotated)
            if step % 50 == 0:
                print(f"  t={t:5.2f}s  {episode.mode}  dist={state.dist_xy:.3f}  z={state.base_pos[2]:.3f}")
            mujoco.mj_step(model, data)
            continue

        if episode.mode == MODE_WALK:
            obs, clip = build_walk_obs(state, episode, cp.control_dt, args,
                                       default_dof_pos, walk_norm, cmd_scale, detection=detection)
            episode.last_actions = infer(policies.walk, obs, clip)

        else:  # MODE_KICK — ball still on ground, estimate position from vision or GT
            episode.gait_phase = (episode.gait_phase + cp.control_dt * args.gait_freq) % 1.0

            if args.use_gt_ball:
                ball_pos = state.ball_pos
            elif detection is not None:
                est = estimate_ball_world(data, cam_id, detection[0], detection[1],
                                          head_renderer_depth, intrinsics)
                if est is not None:
                    err = np.linalg.norm(est - state.ball_pos)
                    print(f"  [t={t:.2f}s] vision=({est[0]:.3f},{est[1]:.3f},{est[2]:.3f})"
                          f"  gt=({state.ball_pos[0]:.3f},{state.ball_pos[1]:.3f},{state.ball_pos[2]:.3f})"
                          f"  err={err:.4f}m")
                    ball_pos = est
                else:
                    ball_pos = state.ball_pos
            else:
                print(f"  [t={t:.2f}s] ball not visible — using ground truth")
                ball_pos = state.ball_pos

            obs, clip = build_kick_obs(state, ball_pos, episode, default_dof_pos, kick_norm)
            episode.last_actions = infer(policies.kick, obs, clip)

        # ── Actuate ───────────────────────────────────────────────────────────
        apply_head_control(data, mm, episode, cp.control_dt, detection,
                           head_kp_walk, head_kd_walk,
                           active_during_kick=args.active_head_during_kick)
        apply_pd(model, data, mm,
                 default_dof_pos + cp.action_scale * episode.last_actions,
                 kp, kd, cp.decimation)
        render_frame(step, cp.render_every, data, renderer, camera, writer, head_rgb_annotated)

        if step % 50 == 0:
            print(f"  t={t:5.2f}s  {episode.mode}  dist={state.dist_xy:.3f}  "
                  f"z={state.base_pos[2]:.3f}  ball_vis={'yes' if detection else 'no'}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--walk-ckpt",       type=str, default="deploy/models/param_walk.pt")
    p.add_argument("--kick-ckpt",       type=str, default="deploy/models/kicking.pt")
    p.add_argument("--walk-deploy-cfg", type=str, default="deploy/configs/Parameter_Walk.yaml",
                   help="Path to walk deploy YAML")
    p.add_argument("--kick-deploy-cfg", type=str, default="deploy/configs/Kicking_Robust.yaml",
                   help="Path to kick deploy YAML")
    p.add_argument("--ball-dist",     type=float, default=1.0,  help="Ball distance from robot [m]")
    p.add_argument("--switch-dist",   type=float, default=0.4,  help="Distance to switch to kick [m]")
    p.add_argument("--slowdown-dist", type=float, default=0.6,  help="Distance to start slowing [m]")
    p.add_argument("--duration",      type=float, default=10.0, help="Simulation duration [s]")
    p.add_argument("--fps",           type=int,   default=30)
    p.add_argument("--width",         type=int,   default=1280)
    p.add_argument("--height",        type=int,   default=720)
    p.add_argument("--out",           type=str,   default=None)
    p.add_argument("--vx",            type=float, default=0.3,  help="Walk forward speed [m/s]")
    p.add_argument("--gait-freq",     type=float, default=1.0,  help="Gait frequency [Hz]")
    p.add_argument("--robot-x",       type=float, default=0.0)
    p.add_argument("--robot-y",       type=float, default=0.0)
    p.add_argument("--robot-yaw",     type=float, default=0.0,  help="Robot start yaw [rad]")
    p.add_argument("--ball-angle",    type=float, default=0.0,  help="Ball angle relative to robot yaw [rad]")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--use-gt-ball",    action="store_true",
                   help="Use ground-truth ball position instead of vision estimate (debug)")
    p.add_argument("--fallen-thresh",  type=float, default=FALLEN_GRAVITY_Z,
                   help="Fall detection threshold on -proj_gravity[2] (default %(default)s; sim2sim.py uses 0.5)")
    p.add_argument("--active-head-during-kick", action="store_true",
                   help="Keep head PD active during kick/recover (debug: tests head torque effect)")
    p.add_argument("--cam-save-dir", type=str, default=None,
                   help="Directory to save head-camera images when ball is detected (default: disabled)")
    p.add_argument("--cam-save-every", type=int, default=5,
                   help="Save one image every N detection steps (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    repo      = os.path.dirname(os.path.abspath(__file__))
    walk_ckpt = args.walk_ckpt or find_latest_checkpoint(
        os.path.join(repo, "logs/T1/T1/Parameter_Walk/**/*.pt"))
    kick_ckpt = args.kick_ckpt or find_latest_checkpoint(
        os.path.join(repo, "logs/T1/T1/Kicking/**/*.pt"))
    out_path  = args.out or os.path.join(
        repo, f"videos/sim2sim_vision_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    print(f"Walk: {walk_ckpt}\nKick: {kick_ckpt}")

    walk_cfg  = load_yaml(args.walk_deploy_cfg)
    kick_cfg  = load_yaml(args.kick_deploy_cfg)
    print(f"Walk deploy cfg: {args.walk_deploy_cfg}")
    print(f"Kick deploy cfg: {args.kick_deploy_cfg}")
    walk_norm = walk_cfg["policy"]["normalization"]
    kick_norm = kick_cfg["policy"]["normalization"]

    sim_dt     = walk_cfg["common"]["dt"]
    decimation = walk_cfg["policy"]["control"]["decimation"]
    cp = ControlParams(
        sim_dt=sim_dt,
        decimation=decimation,
        control_dt=sim_dt * decimation,
        action_scale=walk_cfg["policy"]["control"]["action_scale"],
        n_steps=int(round(args.duration / (sim_dt * decimation))),
        render_every=max(1, int(round(1.0 / (sim_dt * decimation * args.fps)))),
    )

    policies = Policies(
        walk=torch.jit.load(walk_ckpt, map_location="cpu").eval(),
        kick=torch.jit.load(kick_ckpt, map_location="cpu").eval(),
    )

    angle  = args.robot_yaw + args.ball_angle
    ball_x = args.robot_x + args.ball_dist * math.cos(angle)
    ball_y = args.robot_y + args.ball_dist * math.sin(angle)
    print(f"Robot: ({args.robot_x:.2f}, {args.robot_y:.2f}) yaw={args.robot_yaw:.2f}  "
          f"Ball: ({ball_x:.2f}, {ball_y:.2f})")

    scene_path = os.path.join(repo, "resources/T1/_sim2sim_scene.xml")
    with open(scene_path, "w") as f:
        f.write(build_scene_xml(os.path.join(repo, "resources/T1/T1_locomotion.xml"), ball_x, ball_y))

    model = mujoco.MjModel.from_xml_path(scene_path)
    model.opt.timestep = sim_dt
    data  = mujoco.MjData(model)
    mm    = build_model_maps(model)

    default_dof_pos = build_defaults(walk_cfg)
    kp, kd          = build_pd_gains(walk_cfg)
    head_kp_walk    = float(walk_cfg["policy"]["control"]["head_kp_walk"])
    head_kd_walk    = float(walk_cfg["policy"]["control"]["head_kd_walk"])
    cam_id           = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    intrinsics       = build_cam_intrinsics(model, cam_id)

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
        walk_command=np.array([args.vx, 0.0, 0.0, args.gait_freq,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera   = mujoco.MjvCamera()
    camera.type, camera.distance, camera.elevation, camera.azimuth = (
        mujoco.mjtCamera.mjCAMERA_FREE, 3.5, -20, 135
    )
    # Use the actual rendered frame rate (may differ from args.fps due to render_every rounding)
    video_fps = round(1.0 / (cp.render_every * cp.control_dt))
    writer = imageio.get_writer(out_path, fps=video_fps, codec="libx264", quality=8)

    head_renderer_rgb   = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    head_renderer_depth = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    head_renderer_depth.enable_depth_rendering()

    print(f"Rendering {cp.n_steps} steps → {out_path}")
    t0 = time.time()
    try:
        run_episode(model, data, mm, policies, episode, cp, args,
                    walk_norm, kick_norm, default_dof_pos, kp, kd,
                    head_kp_walk, head_kd_walk,
                    renderer, camera, writer, cam_id, intrinsics,
                    head_renderer_rgb, head_renderer_depth)
    finally:
        writer.close()
        del renderer, head_renderer_rgb, head_renderer_depth

    print(f"Done in {time.time() - t0:.1f}s  →  {out_path}")
    print(f"Final ball: {data.qpos[mm.ball_qpos_adr:mm.ball_qpos_adr + 3]}")
    print(f"Final robot: {data.qpos[0:3]}")


if __name__ == "__main__":
    main()
    os._exit(0) # mujoco exit handler
