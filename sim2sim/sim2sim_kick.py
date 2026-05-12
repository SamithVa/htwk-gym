"""Headless MuJoCo sim2sim for T1: kicking policy only.

Loads the trained Kicking_Robust_Bilateral TorchScript policy, places the robot
in front of the ball, and runs the kick policy for a fixed duration.

Obs layout (44 dims):
  [0:3]   projected gravity
  [3:6]   base angular velocity (local)
  [6:8]   relative ball position XY (robot frame)
  [8:20]  joint positions - default (12 joints)
  [20:32] joint velocities (12 joints)
  [32:44] last actions (12 joints)

Run:
python sim2sim/sim2sim_kick.py \
  --ball-dist 0.35 --ball-angle 0 \
  --duration 4 \
  --csv logs/sim2sim_kick.csv
"""

import argparse
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


@dataclass
class SimulationPaths:
    repo: str
    kick_ckpt: str
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kick-ckpt", type=str,
                        default="logs/T1/T1/Kicking_Robust_Bilateral/2026-05-07-14-13-56/nn/model_15100.pt")
    parser.add_argument("--ball-dist", type=float, default=0.35,
                        help="Ball distance from robot [m]")
    parser.add_argument("--ball-angle", type=float, default=0.0,
                        help="Ball angle in degrees (0=straight ahead, positive=left)")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None,
                        help="Save obs log to this CSV path")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def quat_rotate_inverse_wxyz(q_wxyz, v):
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * (2.0 * w)
    c = qv * (2.0 * np.dot(qv, v))
    return a - b + c


def resolve_paths(args):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(
        repo, f"videos/sim2sim_kick_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    )
    return SimulationPaths(
        repo=repo,
        kick_ckpt=args.kick_ckpt,
        kick_cfg=os.path.join(repo, "deploy/configs/Kicking_Robust.yaml"),
        robot_xml=os.path.join(repo, "resources/T1/T1_locomotion.xml"),
        out_path=out_path,
    )


def build_control_params(kick_cfg, args):
    sim_dt = kick_cfg["common"]["dt"]
    decimation = kick_cfg["policy"]["control"]["decimation"]
    control_dt = sim_dt * decimation
    return ControlParams(
        sim_dt=sim_dt,
        decimation=decimation,
        control_dt=control_dt,
        action_scale=kick_cfg["policy"]["control"]["action_scale"],
        render_every=max(1, int(round(1.0 / (control_dt * args.fps)))),
        n_ctrl_steps=int(round(args.duration / control_dt)),
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
    visual_block = """
    <visual>
        <global offwidth="1280" offheight="720"/>
    </visual>
"""
    return xml.replace("<size njmax=", visual_block + "    <size njmax=", 1)


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
        if jid < 0:
            raise ValueError(f"Joint {name} not found")
        if aid < 0:
            raise ValueError(f"Actuator {name} not found")
        qpos_idx[i] = model.jnt_qposadr[jid]
        qvel_idx[i] = model.jnt_dofadr[jid]
        actuator_idx[i] = aid

    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if ball_bid < 0:
        raise ValueError("Ball body not found")
    ball_jadr = model.body_jntadr[ball_bid]
    if ball_jadr < 0:
        raise ValueError("Ball has no joint")

    return ModelMaps(
        qpos_idx=qpos_idx,
        qvel_idx=qvel_idx,
        actuator_idx=actuator_idx,
        ball_qpos_adr=model.jnt_qposadr[ball_jadr],
    )


def build_kick_observation(data, model_maps, default_joint_pos, last_actions, norm):
    base_pos = np.array(data.qpos[0:3], dtype=np.float32)
    base_quat_wxyz = np.array(data.qpos[3:7], dtype=np.float32)
    ang_vel_world = np.array(data.qvel[3:6], dtype=np.float32)
    ang_vel_local = quat_rotate_inverse_wxyz(base_quat_wxyz, ang_vel_world)
    proj_gravity = quat_rotate_inverse_wxyz(
        base_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32)
    )

    joint_pos = np.array(data.qpos[model_maps.qpos_idx], dtype=np.float32)
    joint_vel = np.array(data.qvel[model_maps.qvel_idx], dtype=np.float32)

    ball_pos = np.array(data.qpos[model_maps.ball_qpos_adr:model_maps.ball_qpos_adr + 3], dtype=np.float32)
    ball_rel_world = ball_pos - base_pos
    ball_rel_local = quat_rotate_inverse_wxyz(base_quat_wxyz, ball_rel_world)

    return np.concatenate([
        proj_gravity * norm["gravity"],                         # 3
        ang_vel_local * norm["ang_vel"],                        # 3
        ball_rel_local[:2] * norm["ball_pos"],                  # 2
        (joint_pos - default_joint_pos) * norm["dof_pos"],      # 12
        joint_vel * norm["dof_vel"],                            # 12
        last_actions,                                           # 12
    ]).astype(np.float32)


def apply_pd_control(model, data, model_maps, dof_target, kp, kd, decimation):
    ctrl_range = model.actuator_ctrlrange[model_maps.actuator_idx]
    for _ in range(decimation):
        q = np.array(data.qpos[model_maps.qpos_idx], dtype=np.float32)
        qd = np.array(data.qvel[model_maps.qvel_idx], dtype=np.float32)
        tau = np.clip(kp * (dof_target - q) - kd * qd,
                      ctrl_range[:, 0], ctrl_range[:, 1])
        data.ctrl[model_maps.actuator_idx] = tau
        mujoco.mj_step(model, data)


def main():
    args = parse_args()
    paths = resolve_paths(args)

    kick_cfg = load_yaml(paths.kick_cfg)
    control = build_control_params(kick_cfg, args)
    norm = kick_cfg["policy"]["normalization"]

    default_joint_pos = np.array(kick_cfg["common"]["default_qpos"][11:23], dtype=np.float32)
    kp = np.array(kick_cfg["common"]["stiffness"][11:23], dtype=np.float32)
    kd = np.array(kick_cfg["common"]["damping"][11:23], dtype=np.float32)

    print(f"Kick policy: {paths.kick_ckpt}")
    policy = torch.jit.load(paths.kick_ckpt, map_location="cpu").eval()

    # Place ball at given distance/angle in front of robot
    angle_rad = math.radians(args.ball_angle)
    bx = args.ball_dist * math.cos(angle_rad)
    by = args.ball_dist * math.sin(angle_rad)

    scene_xml = build_scene_xml(paths.robot_xml, ball_pos=(bx, by, BALL_RADIUS))
    scene_path = write_scene_xml(scene_xml, paths.robot_xml)
    model = mujoco.MjModel.from_xml_path(scene_path)
    model.opt.timestep = control.sim_dt
    data = mujoco.MjData(model)
    model_maps = build_model_maps(model)

    # Initialize robot pose
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = BASE_START_POS
    data.qpos[3:7] = IDENTITY_QUAT_WXYZ
    data.qpos[model_maps.qpos_idx] = default_joint_pos
    data.qpos[model_maps.ball_qpos_adr:model_maps.ball_qpos_adr + 3] = [bx, by, BALL_RADIUS]
    data.qpos[model_maps.ball_qpos_adr + 3:model_maps.ball_qpos_adr + 7] = IDENTITY_QUAT_WXYZ
    mujoco.mj_forward(model, data)

    last_actions = np.zeros(len(ISAAC_DOF_NAMES), dtype=np.float32)

    # Renderer
    os.makedirs(os.path.dirname(paths.out_path), exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.5  # closer for kick
    camera.elevation = -20
    camera.azimuth = 135
    writer = imageio.get_writer(paths.out_path, fps=args.fps, codec="libx264", quality=8)

    _csv_f, _csv_w = None, None
    if args.csv:
        import csv as _csv
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        _csv_f = open(args.csv, "w", newline="")
        _csv_w = _csv.writer(_csv_f)

    print(f"Rendering to {paths.out_path}  ({control.n_ctrl_steps} steps)")
    start = time.time()
    try:
        for step in range(control.n_ctrl_steps):
            obs = build_kick_observation(data, model_maps, default_joint_pos, last_actions, norm)

            if _csv_w is not None:
                if step == 0:
                    _csv_w.writerow(["step", "time_s"] + [f"obs_{i}" for i in range(len(obs))])
                _csv_w.writerow([step, round(step * control.control_dt, 4)] + obs.tolist())

            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
            last_actions = np.clip(action, -norm["clip_actions"], norm["clip_actions"]).astype(np.float32)
            dof_target = default_joint_pos + control.action_scale * last_actions

            apply_pd_control(model, data, model_maps, dof_target, kp, kd, control.decimation)

            if step % control.render_every == 0:
                camera.lookat[:] = [data.qpos[0], data.qpos[1], data.qpos[2] - 0.2]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())

    finally:
        writer.close()
        del renderer
        if _csv_f:
            _csv_f.close()

    print(f"Done in {time.time() - start:.1f}s")
    print(f"Video: {paths.out_path}")
    if args.csv:
        print(f"CSV:   {args.csv}")


if __name__ == "__main__":
    main()
