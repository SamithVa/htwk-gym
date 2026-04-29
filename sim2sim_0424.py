"""Headless MuJoCo sim2sim: walk to ball → kick → recover → walk."""

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

BALL_RADIUS = 0.075
BASE_START_POS = np.array([0.0, 0.0, 0.70], dtype=np.float32)
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
MODE_WALK = "WALK"
MODE_KICK = "KICK"
MODE_RECOVER = "RECOVER"
MODE_FALLEN = "FALLEN"

RECOVER_STEPS = 30       # control steps to interpolate back to default pose
KICK_SETTLE_STEPS = 25   # control steps kick policy runs after ball departs
BALL_KICK_THRESHOLD = 0.15  # metres ball must move to count as kicked
FALLEN_GRAVITY_Z = 0.5   # -proj_gravity[2] below this → fallen


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
    kick_ball_start: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    kick_settle_remaining: int = 0
    recover_start_pos: np.ndarray = field(default_factory=lambda: np.zeros(len(ISAAC_DOF_NAMES), dtype=np.float32))
    recover_step: int = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--walk-ckpt", type=str, default="logs/T1/T1/Parameter_Walk/2026-03-22-14-19-09/nn/model_20000.pt")
    p.add_argument("--kick-ckpt", type=str, default=None)
    p.add_argument("--ball-dist", type=float, default=1.0)
    p.add_argument("--ball-angle", type=float, default=0.0, help="Ball angle in degrees (0=ahead, positive=left)")
    p.add_argument("--switch-dist", type=float, default=0.4, help="Switch walk→kick distance [m]")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--vx", type=float, default=0.4, help="Walk speed [m/s]")
    p.add_argument("--gait-freq", type=float, default=1.5)
    return p.parse_args()


def find_latest_checkpoint(glob_pattern):
    matches = sorted(glob.glob(glob_pattern, recursive=True), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for {glob_pattern}")
    return matches[-1]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def quat_rotate_inverse_wxyz(q_wxyz, v):
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    return v * (2.0 * w * w - 1.0) - np.cross(qv, v) * (2.0 * w) + qv * (2.0 * np.dot(qv, v))


def compute_ball_local_direction(base_pos, base_quat_wxyz, ball_xy):
    to_ball = np.array([ball_xy[0] - base_pos[0], ball_xy[1] - base_pos[1], 0.0], dtype=np.float32)
    norm = np.linalg.norm(to_ball[:2])
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return quat_rotate_inverse_wxyz(base_quat_wxyz, to_ball / norm)[:2]


def resolve_paths(args):
    repo = os.path.dirname(os.path.abspath(__file__))
    walk_ckpt = args.walk_ckpt or find_latest_checkpoint(os.path.join(repo, "logs/T1/T1/Parameter_Walk/**/*.pt"))
    kick_ckpt = args.kick_ckpt or find_latest_checkpoint(os.path.join(repo, "logs/T1/T1/Kicking_Robust_44obs/**/*.pt"))
    out_path = args.out or os.path.join(repo, f"videos/sim2sim_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    return SimulationPaths(
        repo=repo, walk_ckpt=walk_ckpt, kick_ckpt=kick_ckpt,
        walk_cfg=os.path.join(repo, "envs/T1/Parameter_Walk.yaml"),
        kick_cfg=os.path.join(repo, "envs/T1/Kicking_Robust_44obs.yaml"),
        robot_xml=os.path.join(repo, "resources/T1/T1_locomotion.xml"),
        out_path=out_path,
    )


def build_control_params(walk_cfg, args):
    sim_dt = walk_cfg["sim"]["dt"]
    decimation = walk_cfg["control"]["decimation"]
    control_dt = sim_dt * decimation
    return ControlParams(
        sim_dt=sim_dt, decimation=decimation, control_dt=control_dt,
        action_scale=walk_cfg["control"]["action_scale"],
        render_every=max(1, int(round(1.0 / (control_dt * args.fps)))),
        n_ctrl_steps=int(round(args.duration / control_dt)),
    )


def load_policies(paths):
    print(f"Walk policy: {paths.walk_ckpt}")
    print(f"Kick policy: {paths.kick_ckpt}")
    return Policies(
        walk=torch.jit.load(paths.walk_ckpt, map_location="cpu").eval(),
        kick=torch.jit.load(paths.kick_ckpt, map_location="cpu").eval(),
    )


def build_scene_xml(robot_xml_path, ball_pos):
    with open(robot_xml_path, "r", encoding="utf-8") as f:
        xml = f.read()
    ball_body = f"""
        <body name="ball" pos="{ball_pos[0]} {ball_pos[1]} {ball_pos[2]}">
            <freejoint/>
            <geom name="ball" type="sphere" size="{BALL_RADIUS}" rgba="0.9 0.1 0.1 1"
                  mass="0.2" friction="1 0.05 0.001" condim="4"/>
        </body>
    </worldbody>"""
    xml = xml.replace("</worldbody>", ball_body, 1)
    visual = '\n    <visual>\n        <global offwidth="1280" offheight="720"/>\n    </visual>\n'
    return xml.replace("<size njmax=", visual + "    <size njmax=", 1)


def write_scene_xml(xml_str, original_xml_path):
    out_path = os.path.join(os.path.dirname(original_xml_path), "_sim2sim_scene.xml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return out_path


def build_model_maps(model):
    qpos_idx = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.int64)
    qvel_idx = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.int64)
    actuator_idx = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.int64)
    for i, name in enumerate(ISAAC_DOF_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if jid < 0: raise ValueError(f"Joint {name} not found")
        if aid < 0: raise ValueError(f"Actuator {name} not found")
        qpos_idx[i] = model.jnt_qposadr[jid]
        qvel_idx[i] = model.jnt_dofadr[jid]
        actuator_idx[i] = aid

    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if ball_bid < 0: raise ValueError("Ball body not found")
    ball_jadr = model.body_jntadr[ball_bid]
    if ball_jadr < 0: raise ValueError("Ball has no joint")

    lf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    rf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
    if lf < 0 or rf < 0: raise ValueError("Foot bodies not found")

    return ModelMaps(qpos_idx, qvel_idx, actuator_idx,
                     model.jnt_qposadr[ball_jadr], lf, rf)


def build_default_joint_positions(cfg):
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


def build_joint_pd_gains(cfg):
    kp = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.float32)
    kd = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.float32)
    for i, name in enumerate(ISAAC_DOF_NAMES):
        for key in cfg["control"]["stiffness"]:
            if key in name:
                kp[i] = cfg["control"]["stiffness"][key]
                kd[i] = cfg["control"]["damping"][key]
                break
    return kp, kd


def create_simulation(paths, args, sim_dt):
    angle = math.radians(args.ball_angle)
    bx, by = args.ball_dist * math.cos(angle), args.ball_dist * math.sin(angle)
    scene_path = write_scene_xml(build_scene_xml(paths.robot_xml, (bx, by, BALL_RADIUS)), paths.robot_xml)
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
    angle = math.radians(args.ball_angle)
    bx, by = args.ball_dist * math.cos(angle), args.ball_dist * math.sin(angle)
    data.qpos[model_maps.ball_qpos_adr:model_maps.ball_qpos_adr + 3] = [bx, by, BALL_RADIUS]
    data.qpos[model_maps.ball_qpos_adr + 3:model_maps.ball_qpos_adr + 7] = IDENTITY_QUAT_WXYZ
    mujoco.mj_forward(model, data)
    return EpisodeState(
        mode=MODE_WALK,
        last_actions=np.zeros(len(ISAAC_DOF_NAMES), dtype=np.float32),
        gait_phase=0.0,
        walk_command=np.array([args.vx, 0.0, 0.0, args.gait_freq, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
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


def read_robot_state(data, model_maps):
    base_pos = np.array(data.qpos[0:3], dtype=np.float32)
    base_quat_wxyz = np.array(data.qpos[3:7], dtype=np.float32)
    base_ang_vel_local = quat_rotate_inverse_wxyz(base_quat_wxyz, data.qvel[3:6]).astype(np.float32)
    proj_gravity = quat_rotate_inverse_wxyz(base_quat_wxyz, [0.0, 0.0, -1.0]).astype(np.float32)
    joint_pos = np.array(data.qpos[model_maps.qpos_idx], dtype=np.float32)
    joint_vel = np.array(data.qvel[model_maps.qvel_idx], dtype=np.float32)
    ball_pos = np.array(data.qpos[model_maps.ball_qpos_adr:model_maps.ball_qpos_adr + 3], dtype=np.float32)
    ball_xy = ball_pos[:2].copy()
    return RobotState(
        base_pos=base_pos, base_quat_wxyz=base_quat_wxyz,
        base_ang_vel_local=base_ang_vel_local, proj_gravity=proj_gravity,
        joint_pos=joint_pos, joint_vel=joint_vel,
        ball_pos=ball_pos, ball_xy=ball_xy,
        dist_xy=float(np.linalg.norm(ball_xy - base_pos[:2])),
        left_foot_pos=np.array(data.xpos[model_maps.left_foot_body_id], dtype=np.float32),
        right_foot_pos=np.array(data.xpos[model_maps.right_foot_body_id], dtype=np.float32),
    )


def build_walk_observation(state, episode, control_dt, args, default_joint_pos, walk_norm, walk_cmd_scale):
    if np.any(episode.kick_ball_start):
        episode.walk_command[0] = 0.3
        episode.walk_command[1] = 0.0
    else:
        dir_local = compute_ball_local_direction(state.base_pos, state.base_quat_wxyz, state.ball_xy)
        episode.walk_command[0] = args.vx * dir_local[0]
        episode.walk_command[1] = args.vx * dir_local[1]
    episode.walk_command[3] = args.gait_freq
    episode.gait_phase = (episode.gait_phase + control_dt * args.gait_freq) % 1.0
    obs = np.concatenate([
        state.proj_gravity * walk_norm["gravity"],
        state.base_ang_vel_local * walk_norm["ang_vel"],
        episode.walk_command * walk_cmd_scale,
        [np.cos(2.0 * np.pi * episode.gait_phase), np.sin(2.0 * np.pi * episode.gait_phase)],
        (state.joint_pos - default_joint_pos) * walk_norm["dof_pos"],
        state.joint_vel * walk_norm["dof_vel"],
        episode.last_actions,
    ]).astype(np.float32)
    return obs, walk_norm["clip_actions"]


def build_kick_observation(state, episode, default_joint_pos, kick_norm):
    ball_pos_norm = kick_norm["ball_pos"]
    rel_ball = quat_rotate_inverse_wxyz(state.base_quat_wxyz, state.ball_pos - state.base_pos).astype(np.float32)
    obs = np.concatenate([
        state.proj_gravity * kick_norm["gravity"],          # 3
        state.base_ang_vel_local * kick_norm["ang_vel"],    # 3
        rel_ball[:2] * ball_pos_norm,                       # 2
        (state.joint_pos - default_joint_pos) * kick_norm["dof_pos"],  # 12
        state.joint_vel * kick_norm["dof_vel"],             # 12
        episode.last_actions,                               # 12
    ]).astype(np.float32)
    return obs, kick_norm["clip_actions"]


def infer_action(policy, obs, clip_actions):
    with torch.no_grad():
        action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
    return np.clip(action, -clip_actions, clip_actions).astype(np.float32)


def apply_pd_control(model, data, model_maps, dof_target, kp, kd, decimation):
    ctrl_range = model.actuator_ctrlrange[model_maps.actuator_idx]
    for _ in range(decimation):
        q = np.array(data.qpos[model_maps.qpos_idx], dtype=np.float32)
        qd = np.array(data.qvel[model_maps.qvel_idx], dtype=np.float32)
        tau = np.clip(kp * (dof_target - q) - kd * qd, ctrl_range[:, 0], ctrl_range[:, 1])
        data.ctrl[model_maps.actuator_idx] = tau
        mujoco.mj_step(model, data)


def render_frame(step, render_every, data, renderer, camera, writer):
    if step % render_every != 0:
        return
    camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.5]
    renderer.update_scene(data, camera=camera)
    writer.append_data(renderer.render())


def log_progress(step, control_dt, state, mode):
    if step % 50 != 0:
        return
    print(f"  t={step * control_dt:5.2f}s  mode={mode}  dist={state.dist_xy:.3f}  "
          f"base_z={state.base_pos[2]:.3f}  ball_xy=({state.ball_xy[0]:.2f},{state.ball_xy[1]:.2f})")


def run_episode(model, data, model_maps, policies, episode, control, args,
                walk_norm, kick_norm, default_joint_pos, kp, kd, renderer, camera, writer):
    walk_cmd_scale = np.array([
        walk_norm["lin_vel"], walk_norm["lin_vel"], walk_norm["ang_vel"],
        walk_norm["gait_frequency"], walk_norm["foot_yaw"], walk_norm["foot_yaw"],
        walk_norm["body_pitch_target"], walk_norm["body_roll_target"],
        walk_norm["feet_offset_x_target"], walk_norm["feet_offset_y_target"],
    ], dtype=np.float32)

    start_time = time.time()
    try:
        for step in range(control.n_ctrl_steps):
            state = read_robot_state(data, model_maps)

            if episode.mode != MODE_FALLEN and -state.proj_gravity[2] < FALLEN_GRAVITY_Z:
                episode.mode = MODE_FALLEN
                print(f"[t={step * control.control_dt:.2f}s] FALLEN (proj_gravity_z={state.proj_gravity[2]:.3f})")

            if episode.mode == MODE_FALLEN:
                data.ctrl[:] = 0.0
                render_frame(step, control.render_every, data, renderer, camera, writer)
                log_progress(step, control.control_dt, state, episode.mode)
                mujoco.mj_step(model, data)
                continue

            if episode.mode == MODE_WALK and state.dist_xy <= args.switch_dist:
                episode.mode = MODE_KICK
                episode.kick_ball_start = state.ball_pos.copy()
                print(f"[t={step * control.control_dt:.2f}s] WALK -> KICK at dist {state.dist_xy:.3f} m")

            if episode.mode == MODE_KICK:
                ball_moved = np.linalg.norm(state.ball_pos[:2] - episode.kick_ball_start[:2])
                if ball_moved > BALL_KICK_THRESHOLD and episode.kick_settle_remaining == 0:
                    episode.kick_settle_remaining = KICK_SETTLE_STEPS
                    print(f"[t={step * control.control_dt:.2f}s] ball kicked ({ball_moved:.3f} m)")
                if episode.kick_settle_remaining > 0:
                    episode.kick_settle_remaining -= 1
                    if episode.kick_settle_remaining == 0:
                        episode.mode = MODE_RECOVER
                        episode.recover_start_pos = state.joint_pos.copy()
                        episode.recover_step = 0
                        print(f"[t={step * control.control_dt:.2f}s] KICK -> RECOVER")

            if episode.mode == MODE_RECOVER:
                alpha = min(episode.recover_step / RECOVER_STEPS, 1.0)
                dof_target = (1.0 - alpha) * episode.recover_start_pos + alpha * default_joint_pos
                episode.recover_step += 1
                if episode.recover_step >= RECOVER_STEPS:
                    episode.mode = MODE_WALK
                    print(f"[t={step * control.control_dt:.2f}s] RECOVER -> WALK")
                episode.last_ball_pos = state.ball_pos.copy()
                apply_pd_control(model, data, model_maps, dof_target, kp, kd, control.decimation)
                render_frame(step, control.render_every, data, renderer, camera, writer)
                log_progress(step, control.control_dt, state, episode.mode)
                continue

            if episode.mode == MODE_WALK:
                obs, clip_actions = build_walk_observation(
                    state, episode, control.control_dt, args, default_joint_pos, walk_norm, walk_cmd_scale)
                policy = policies.walk
            else:  # MODE_KICK
                obs, clip_actions = build_kick_observation(state, episode, default_joint_pos, kick_norm)
                policy = policies.kick

            episode.last_actions = infer_action(policy, obs, clip_actions)
            episode.last_ball_pos = state.ball_pos.copy()
            dof_target = default_joint_pos + control.action_scale * episode.last_actions
            apply_pd_control(model, data, model_maps, dof_target, kp, kd, control.decimation)
            render_frame(step, control.render_every, data, renderer, camera, writer)
            log_progress(step, control.control_dt, state, episode.mode)
    finally:
        writer.close()
        del renderer

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
    print(f"Rendering to {paths.out_path} ({control.n_ctrl_steps} steps, render every {control.render_every})")
    run_episode(model, data, model_maps, policies, episode, control, args,
                walk_cfg["normalization"], kick_cfg["normalization"],
                default_joint_pos, kp, kd, renderer, camera, writer)
    print(f"Video: {paths.out_path}")


if __name__ == "__main__":
    main()
    os._exit(0)
