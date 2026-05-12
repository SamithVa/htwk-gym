"""Headless MuJoCo sim2sim for T1: walk to ball, then kick.

Loads the trained Parameter_Walk and Kicking TorchScript policies, drops the
robot and a ball into a MuJoCo scene, commands the walk policy forward, and
switches to the kick policy when close to the ball. Renders offscreen to MP4.

Run : 
python sim2sim/sim2sim_param_walk.py \
  --csv logs/sim2sim.csv \
  --vx 0.3 --gait-freq 1.9 \
  --duration 6 \
  --ball-dist 999
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
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)

BALL_RADIUS = 0.075
BASE_START_POS = np.array([0.0, 0.0, 0.70], dtype=np.float32)
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
MODE_WALK = "WALK"
MODE_KICK = "KICK"


@dataclass
class SimulationPaths:
    repo: str
    walk_ckpt: str
    kick_ckpt: str
    walk_cfg: str
    kick_cfg: str
    robot_xml: str
    out_path: str


@dataclass
class ControlParams:
    sim_dt: float
    decimation: int
    control_dt: float
    action_scale: float
    render_every: int
    n_ctrl_steps: int


@dataclass
class ModelMaps:
    qpos_idx: np.ndarray
    qvel_idx: np.ndarray
    actuator_idx: np.ndarray
    ball_qpos_adr: int
    left_foot_body_id: int
    right_foot_body_id: int


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
    ball_xy: np.ndarray
    dist_xy: float
    left_foot_pos: np.ndarray
    right_foot_pos: np.ndarray


@dataclass
class EpisodeState:
    mode: str
    last_actions: np.ndarray
    gait_phase: float
    walk_command: np.ndarray
    last_ball_pos: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-ckpt", type=str, default="logs/T1/T1/Parameter_Walk/2026-03-22-14-19-09/nn/model_20000.pt")
    parser.add_argument("--kick-ckpt", type=str, default="logs/T1/T1/Kicking_Robust_Bilateral/2026-05-07-14-13-56/nn/model_15100.pt")
    parser.add_argument("--ball-dist", type=float, default=1.0)
    parser.add_argument("--ball-angle", type=float, default=0.0, help="Ball angle in degrees (0=straight ahead, positive=left)")
    parser.add_argument("--switch-dist", type=float, default=0.5)
    parser.add_argument("--slowdown-dist", type=float, default=0.6)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--vx", type=float, default=0.4, help="Forward walk command [m/s]")
    parser.add_argument("--gait-freq", type=float, default=1.5)
    parser.add_argument("--fixed-cmd", action="store_true",
                        help="Skip ball-steering; hold vx/vy/gait-freq exactly as given (matches play script)")
    parser.add_argument("--csv", type=str, default=None, help="Save walk-mode obs log to this CSV path")
    return parser.parse_args()


def find_latest_checkpoint(glob_pattern):
    """Return the newest .pt file matching the glob pattern."""
    matches = sorted(glob.glob(glob_pattern, recursive=True), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for {glob_pattern}")
    return matches[-1]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def quat_rotate_inverse_wxyz(q_wxyz, v):
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * (2.0 * w)
    c = qv * (2.0 * np.dot(qv, v))
    return a - b + c


def compute_ball_local_direction(base_pos, base_quat_wxyz, ball_xy):
    """Return a unit vector toward the ball in the robot's local XY frame."""
    to_ball_world = np.array([ball_xy[0] - base_pos[0], ball_xy[1] - base_pos[1], 0.0], dtype=np.float32)
    norm = np.linalg.norm(to_ball_world[:2])
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    to_ball_world /= norm
    local = quat_rotate_inverse_wxyz(base_quat_wxyz, to_ball_world)
    return local[:2]


def resolve_paths(args):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    walk_ckpt = args.walk_ckpt or find_latest_checkpoint(
        os.path.join(repo, "logs/T1/T1/Parameter_Walk/**/*.pt")
    )
    kick_ckpt = args.kick_ckpt or find_latest_checkpoint(
        os.path.join(repo, "logs/T1/T1/Kicking_Robust/**/*.pt")
    )
    out_path = args.out or os.path.join(repo, f"videos/sim2sim_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    return SimulationPaths(
        repo=repo,
        walk_ckpt=walk_ckpt,
        kick_ckpt=kick_ckpt,
        walk_cfg=os.path.join(repo, "deploy/configs/Parameter_Walk.yaml"),
        kick_cfg=os.path.join(repo, "deploy/configs/Kicking_Robust.yaml"),
        robot_xml=os.path.join(repo, "resources/T1/T1_locomotion.xml"),
        out_path=out_path,
    )


def build_control_params(walk_cfg, args):
    sim_dt = walk_cfg["common"]["dt"]
    decimation = walk_cfg["policy"]["control"]["decimation"]
    control_dt = sim_dt * decimation
    render_every = max(1, int(round(1.0 / (control_dt * args.fps))))
    n_ctrl_steps = int(round(args.duration / control_dt))
    return ControlParams(
        sim_dt=sim_dt,
        decimation=decimation,
        control_dt=control_dt,
        action_scale=walk_cfg["policy"]["control"]["action_scale"],
        render_every=render_every,
        n_ctrl_steps=n_ctrl_steps,
    )


def load_policies(paths):
    print(f"Walk policy: {paths.walk_ckpt}")
    print(f"Kick policy: {paths.kick_ckpt}")
    return Policies(
        walk=torch.jit.load(paths.walk_ckpt, map_location="cpu").eval(),
        kick=torch.jit.load(paths.kick_ckpt, map_location="cpu").eval(),
    )


def build_scene_xml(robot_xml_path, ball_pos):
    """Inject a free-moving ball and offscreen render settings into the scene XML."""
    with open(robot_xml_path, "r", encoding="utf-8") as file:
        xml = file.read()

    ball_body = f"""
        <body name="ball" pos="{ball_pos[0]} {ball_pos[1]} {ball_pos[2]}">
            <freejoint/>
            <geom name="ball" type="sphere" size="{BALL_RADIUS}" rgba="0.9 0.1 0.1 1"
                  mass="0.2" friction="1 0.05 0.001" condim="4"/>
        </body>
    </worldbody>"""
    xml = xml.replace("</worldbody>", ball_body, 1)

    visual_block = """
    <visual>
        <global offwidth="1280" offheight="720"/>
    </visual>
"""
    return xml.replace("<size njmax=", visual_block + "    <size njmax=", 1)


def write_scene_xml(xml_str, original_xml_path):
    """Write the modified XML next to the source so relative mesh paths still resolve."""
    out_path = os.path.join(os.path.dirname(original_xml_path), "_sim2sim_scene.xml")
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(xml_str)
    return out_path


def build_model_maps(model):
    """Return MuJoCo indices for the 12 Isaac joint ordering plus the ball qpos address."""
    qpos_idx = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.int64)
    qvel_idx = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.int64)
    actuator_idx = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.int64)

    for index, name in enumerate(ISAAC_DOF_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint {name} not found in MuJoCo model")

        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"Actuator {name} not found in MuJoCo model")

        qpos_idx[index] = model.jnt_qposadr[joint_id]
        qvel_idx[index] = model.jnt_dofadr[joint_id]
        actuator_idx[index] = actuator_id

    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if ball_body_id < 0:
        raise ValueError("Ball body not found in MuJoCo model")

    ball_joint_adr = model.body_jntadr[ball_body_id]
    if ball_joint_adr < 0:
        raise ValueError("Ball body does not have an attached joint")

    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
    if left_foot_id < 0 or right_foot_id < 0:
        raise ValueError("Foot body 'left_foot_link' or 'right_foot_link' not found in MuJoCo model")

    return ModelMaps(
        qpos_idx=qpos_idx,
        qvel_idx=qvel_idx,
        actuator_idx=actuator_idx,
        ball_qpos_adr=model.jnt_qposadr[ball_joint_adr],
        left_foot_body_id=left_foot_id,
        right_foot_body_id=right_foot_id,
    )


def build_default_joint_positions(walk_cfg):
    # deploy config has 23 DOFs; leg joints are at indices 11–22
    return np.array(walk_cfg["common"]["default_qpos"][11:23], dtype=np.float32)


def build_joint_pd_gains(walk_cfg):
    # deploy config has 23 DOFs; leg joints are at indices 11–22
    kp = np.array(walk_cfg["common"]["stiffness"][11:23], dtype=np.float32)
    kd = np.array(walk_cfg["common"]["damping"][11:23], dtype=np.float32)
    return kp, kd


def create_simulation(paths, args, sim_dt):
    _angle_rad = math.radians(args.ball_angle)
    _bx = args.ball_dist * math.cos(_angle_rad)
    _by = args.ball_dist * math.sin(_angle_rad)
    scene_xml = build_scene_xml(paths.robot_xml, ball_pos=(_bx, _by, BALL_RADIUS))
    scene_path = write_scene_xml(scene_xml, paths.robot_xml)
    model = mujoco.MjModel.from_xml_path(scene_path)
    model.opt.timestep = sim_dt
    data = mujoco.MjData(model)
    return model, data, build_model_maps(model)


def initialize_episode(data, model, model_maps, default_joint_pos, args):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    data.qpos[0:3] = BASE_START_POS
    data.qpos[3:7] = IDENTITY_QUAT_WXYZ
    data.qpos[model_maps.qpos_idx] = default_joint_pos

    _angle_rad = math.radians(args.ball_angle)
    _bx = args.ball_dist * math.cos(_angle_rad)
    _by = args.ball_dist * math.sin(_angle_rad)
    data.qpos[model_maps.ball_qpos_adr:model_maps.ball_qpos_adr + 3] = [_bx, _by, BALL_RADIUS]
    data.qpos[model_maps.ball_qpos_adr + 3:model_maps.ball_qpos_adr + 7] = IDENTITY_QUAT_WXYZ

    mujoco.mj_forward(model, data)

    return EpisodeState(
        mode=MODE_WALK,
        last_actions=np.zeros(len(ISAAC_DOF_NAMES), dtype=np.float32),
        gait_phase=0.0,
        walk_command=np.array(
            [args.vx, 0.0, 0.0, args.gait_freq, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        last_ball_pos=np.zeros(3, dtype=np.float32),
    )


def create_renderer_and_writer(model, args, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.5
    camera.elevation = -20
    camera.azimuth = 135

    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264", quality=8)
    return renderer, camera, writer


def build_walk_command_scale(walk_norm):
    return np.array(
        [
            walk_norm["lin_vel"],
            walk_norm["lin_vel"],
            walk_norm["ang_vel"],
            walk_norm["gait_frequency"],
            walk_norm["foot_yaw"],
            walk_norm["foot_yaw"],
            walk_norm["body_pitch_target"],
            walk_norm["body_roll_target"],
            walk_norm["feet_offset_x_target"],
            walk_norm["feet_offset_y_target"],
        ],
        dtype=np.float32,
    )


def read_robot_state(data, model_maps):
    base_pos = np.array(data.qpos[0:3], dtype=np.float32)
    base_quat_wxyz = np.array(data.qpos[3:7], dtype=np.float32)
    base_ang_vel_world = np.array(data.qvel[3:6], dtype=np.float32)
    base_ang_vel_local = quat_rotate_inverse_wxyz(base_quat_wxyz, base_ang_vel_world).astype(np.float32)
    proj_gravity = quat_rotate_inverse_wxyz(
        base_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32)
    ).astype(np.float32)

    joint_pos = np.array(data.qpos[model_maps.qpos_idx], dtype=np.float32)
    joint_vel = np.array(data.qvel[model_maps.qvel_idx], dtype=np.float32)

    ball_pos = np.array(data.qpos[model_maps.ball_qpos_adr:model_maps.ball_qpos_adr + 3], dtype=np.float32)
    ball_xy = ball_pos[:2].copy()
    dist_xy = float(np.linalg.norm(ball_xy - base_pos[:2]))

    left_foot_pos = np.array(data.xpos[model_maps.left_foot_body_id], dtype=np.float32)
    right_foot_pos = np.array(data.xpos[model_maps.right_foot_body_id], dtype=np.float32)

    return RobotState(
        base_pos=base_pos,
        base_quat_wxyz=base_quat_wxyz,
        base_ang_vel_local=base_ang_vel_local,
        proj_gravity=proj_gravity,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        ball_pos=ball_pos,
        ball_xy=ball_xy,
        dist_xy=dist_xy,
        left_foot_pos=left_foot_pos,
        right_foot_pos=right_foot_pos,
    )


def maybe_switch_mode(episode, state, control_dt, step, switch_dist):
    if episode.mode == MODE_WALK and state.dist_xy <= switch_dist:
        episode.mode = MODE_KICK
        print(f"[t={step * control_dt:.2f}s] WALK -> KICK at distance {state.dist_xy:.3f} m")


def compute_walk_slowdown_scale(dist_xy, switch_dist, slowdown_dist):
    slowdown_span = max(slowdown_dist - switch_dist, 1.0e-6)
    if dist_xy <= switch_dist:
        return 0.0
    if dist_xy >= slowdown_dist:
        return 1.0
    return (dist_xy - switch_dist) / slowdown_span


def build_walk_observation(
    state,
    episode,
    control,
    args,
    default_joint_pos,
    walk_norm,
    walk_cmd_scale,
):
    if getattr(args, "fixed_cmd", False):
        walk_gait_freq = args.gait_freq
        episode.gait_phase = (episode.gait_phase + control.control_dt * walk_gait_freq) % 1.0
        # leave episode.walk_command untouched — it was set to [vx,0,0,gait_freq,...] at init
    else:
        slowdown_scale = compute_walk_slowdown_scale(
            state.dist_xy,
            switch_dist=args.switch_dist,
            slowdown_dist=args.slowdown_dist,
        )
        walk_speed = args.vx * slowdown_scale
        walk_gait_freq = args.gait_freq * max(0.5, slowdown_scale)

        dir_local = compute_ball_local_direction(state.base_pos, state.base_quat_wxyz, state.ball_xy)
        walk_vx = walk_speed * dir_local[0]
        walk_vy = walk_speed * dir_local[1] + 0.03

        episode.gait_phase = (episode.gait_phase + control.control_dt * walk_gait_freq) % 1.0
        episode.walk_command[0] = walk_vx
        episode.walk_command[1] = walk_vy
        episode.walk_command[3] = walk_gait_freq

    obs = np.concatenate(
        [
            state.proj_gravity * walk_norm["gravity"],
            state.base_ang_vel_local * walk_norm["ang_vel"],
            episode.walk_command * walk_cmd_scale,
            np.array(
                [
                    np.cos(2.0 * np.pi * episode.gait_phase),
                    np.sin(2.0 * np.pi * episode.gait_phase),
                ],
                dtype=np.float32,
            ),
            (state.joint_pos - default_joint_pos) * walk_norm["dof_pos"],
            state.joint_vel * walk_norm["dof_vel"],
            episode.last_actions,
        ]
    ).astype(np.float32)
    return obs, walk_norm["clip_actions"]


def build_kick_observation(state, episode, default_joint_pos, kick_norm):
    ball_pos_norm = kick_norm["ball_pos"]

    rel_ball = quat_rotate_inverse_wxyz(
        state.base_quat_wxyz,
        state.ball_pos - state.base_pos,
    ).astype(np.float32)

    left_foot_to_ball = quat_rotate_inverse_wxyz(
        state.base_quat_wxyz,
        state.ball_pos - state.left_foot_pos,
    ).astype(np.float32)

    right_foot_to_ball = quat_rotate_inverse_wxyz(
        state.base_quat_wxyz,
        state.ball_pos - state.right_foot_pos,
    ).astype(np.float32)

    obs = np.concatenate(
        [
            state.proj_gravity * kick_norm["gravity"],          # 3
            state.base_ang_vel_local * kick_norm["ang_vel"],    # 3
            rel_ball[:2] * ball_pos_norm,                       # 2
            left_foot_to_ball[:2] * ball_pos_norm,             # 2
            right_foot_to_ball[:2] * ball_pos_norm,            # 2
            (state.joint_pos - default_joint_pos) * kick_norm["dof_pos"],  # 12
            state.joint_vel * kick_norm["dof_vel"],             # 12
            episode.last_actions,                               # 12
        ]
    ).astype(np.float32)
    return obs, kick_norm["clip_actions"]


def build_walk_plus_ball_observation(
    state,
    episode,
    control,
    args,
    default_joint_pos,
    kick_norm,
    kick_cmd_scale,
):
    obs_walk, clip_actions = build_walk_observation(
        state=state,
        episode=episode,
        control=control,
        args=args,
        default_joint_pos=default_joint_pos,
        walk_norm=kick_norm,
        walk_cmd_scale=kick_cmd_scale,
    )

    rel_ball_pos = quat_rotate_inverse_wxyz(
        state.base_quat_wxyz,
        state.ball_pos - state.base_pos,
    ).astype(np.float32)
    ball_vel_world = (state.ball_pos - episode.last_ball_pos) / max(control.control_dt, 1.0e-6)
    rel_ball_vel = quat_rotate_inverse_wxyz(
        state.base_quat_wxyz,
        ball_vel_world.astype(np.float32),
    ).astype(np.float32)

    obs = np.concatenate(
        [
            obs_walk,
            rel_ball_pos[:2] * kick_norm["ball_pos"],
            rel_ball_vel[:2] * kick_norm["ball_vel"],
        ]
    ).astype(np.float32)
    return obs, clip_actions


def infer_action(policy, obs, clip_actions):
    with torch.no_grad():
        action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
    return np.clip(action, -clip_actions, clip_actions).astype(np.float32)


def apply_pd_control(model, data, model_maps, dof_target, kp, kd, decimation):
    ctrl_range = model.actuator_ctrlrange[model_maps.actuator_idx]
    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]

    for _ in range(decimation):
        q_now = np.array(data.qpos[model_maps.qpos_idx], dtype=np.float32)
        qd_now = np.array(data.qvel[model_maps.qvel_idx], dtype=np.float32)
        tau = kp * (dof_target - q_now) - kd * qd_now
        data.ctrl[model_maps.actuator_idx] = np.clip(tau, ctrl_min, ctrl_max)
        mujoco.mj_step(model, data)


def render_frame(step, render_every, data, renderer, camera, writer):
    if step % render_every != 0:
        return

    camera.lookat[:] = [data.qpos[0], data.qpos[1], data.qpos[2] - 0.2]
    renderer.update_scene(data, camera=camera)
    writer.append_data(renderer.render())


def log_progress(step, control_dt, state, mode):
    if step % 50 != 0:
        return

    print(
        f"  t={step * control_dt:5.2f}s  mode={mode}  dist={state.dist_xy:.3f}  "
        f"base_z={state.base_pos[2]:.3f}  ball_xy=({state.ball_xy[0]:.2f},{state.ball_xy[1]:.2f})"
    )


def run_episode(
    model,
    data,
    model_maps,
    policies,
    episode,
    control,
    args,
    walk_norm,
    kick_norm,
    default_joint_pos,
    kp,
    kd,
    renderer,
    camera,
    writer,
    kick_num_obs=44,
    csv_path=None,
):
    walk_cmd_scale = build_walk_command_scale(walk_norm)
    kick_is_walk_compatible = kick_num_obs == 58
    kick_cmd_scale = build_walk_command_scale(kick_norm) if kick_is_walk_compatible else None

    _csv_f, _csv_w = None, None
    if csv_path:
        import csv as _csv
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        _csv_f = open(csv_path, "w", newline="")
        _csv_w = _csv.writer(_csv_f)

    start_time = time.time()
    try:
        for step in range(control.n_ctrl_steps):
            state = read_robot_state(data, model_maps)
            maybe_switch_mode(episode, state, control.control_dt, step, args.switch_dist)

            if episode.mode == MODE_WALK:
                obs, clip_actions = build_walk_observation(
                    state=state,
                    episode=episode,
                    control=control,
                    args=args,
                    default_joint_pos=default_joint_pos,
                    walk_norm=walk_norm,
                    walk_cmd_scale=walk_cmd_scale,
                )
                policy = policies.walk
                if _csv_w is not None:
                    if step == 0:
                        _csv_w.writerow(["step", "time_s"] + [f"obs_{i}" for i in range(len(obs))])
                    _csv_w.writerow([step, round(step * control.control_dt, 4)] + obs.tolist())
            elif kick_is_walk_compatible:
                obs, clip_actions = build_walk_plus_ball_observation(
                    state=state,
                    episode=episode,
                    control=control,
                    args=args,
                    default_joint_pos=default_joint_pos,
                    kick_norm=kick_norm,
                    kick_cmd_scale=kick_cmd_scale,
                )
                policy = policies.kick
            else:
                obs, clip_actions = build_kick_observation(
                    state=state,
                    episode=episode,
                    default_joint_pos=default_joint_pos,
                    kick_norm=kick_norm,
                )
                policy = policies.kick

            episode.last_actions = infer_action(policy, obs, clip_actions)
            episode.last_ball_pos = state.ball_pos.copy()
            dof_target = default_joint_pos + control.action_scale * episode.last_actions

            apply_pd_control(
                model=model,
                data=data,
                model_maps=model_maps,
                dof_target=dof_target,
                kp=kp,
                kd=kd,
                decimation=control.decimation,
            )
            render_frame(step, control.render_every, data, renderer, camera, writer)
            log_progress(step, control.control_dt, state, episode.mode)
    finally:
        writer.close()
        del renderer
        if _csv_f is not None:
            _csv_f.close()

    print(f"Done in {time.time() - start_time:.1f}s.")
    print(f"Final ball pos: {data.qpos[model_maps.ball_qpos_adr:model_maps.ball_qpos_adr + 3]}")
    print(f"Final robot pos: {data.qpos[0:3]}")


def main():
    args = parse_args()
    paths = resolve_paths(args)

    walk_cfg = load_yaml(paths.walk_cfg)
    kick_cfg = load_yaml(paths.kick_cfg)
    control = build_control_params(walk_cfg, args)
    policies = load_policies(paths)

    model, data, model_maps = create_simulation(paths, args, control.sim_dt)
    default_joint_pos = build_default_joint_positions(walk_cfg)
    kp, kd = build_joint_pd_gains(walk_cfg)
    episode = initialize_episode(data, model, model_maps, default_joint_pos, args)
    renderer, camera, writer = create_renderer_and_writer(model, args, paths.out_path)

    print(
        f"Rendering to {paths.out_path} "
        f"({control.n_ctrl_steps} control steps, render every {control.render_every})"
    )

    run_episode(
        model=model,
        data=data,
        model_maps=model_maps,
        policies=policies,
        episode=episode,
        control=control,
        args=args,
        walk_norm=walk_cfg["policy"]["normalization"],
        kick_norm=kick_cfg["policy"]["normalization"],
        default_joint_pos=default_joint_pos,
        kp=kp,
        kd=kd,
        renderer=renderer,
        camera=camera,
        writer=writer,
        kick_num_obs=int(kick_cfg["policy"]["num_observations"]),
        csv_path=args.csv,
    )

    print(f"Video: {paths.out_path}")


if __name__ == "__main__":
    main()
