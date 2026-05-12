# T1 Curriculum: Walk → Approach → Kick

Four-stage curriculum that trains the T1 humanoid to walk, approach a ball, orbit into position, and kick it toward a target. Each stage fine-tunes from the previous checkpoint. All stages share the same environment class (`ParameterWalkKick`) and observation/action layout so checkpoints transfer without modification.

---

## Environment

**Class**: `ParameterWalkKick` (`envs/T1/curriculum/parameter_walk_kick.py`)  
**Task key**: `T1/ParameterWalkKick`  
**Terrain**: flat plane (all stages)  
**Physics**: dt = 0.002 s, decimation = 10 → policy dt = 0.02 s  
**Two actors per env**: robot (index 0) + ball (index 1)

### Observation space — 56 dims

| Slice | Dims | Description | Scale |
|-------|------|-------------|-------|
| projected gravity | 3 | gravity vector in base frame | 1.0 |
| base angular velocity | 3 | body-frame ω | 1.0 |
| commands | 10 | see command table below | per-command |
| ball position (robot frame) | 2 | ball xy relative to base, body-frame | 1.0 |
| gait clock (cos, sin) | 2 | gated by gait_frequency > 0 | — |
| joint position error | 12 | dof_pos − default_dof_pos | 1.0 |
| joint velocity | 12 | | 0.1 |
| last actions | 12 | | — |

### Privileged observation space — 20 dims

| Slice | Dims | Description |
|-------|------|-------------|
| base mass / CoM noise | 4 | randomized mass+CoM offsets |
| base linear velocity | 3 | body-frame v |
| base height | 1 | height above terrain |
| ball linear velocity (xy) | 2 | world frame |
| left foot position (xy) | 2 | world frame |
| right foot position (xy) | 2 | world frame |
| push force | 3 | applied base force |
| push torque | 3 | applied base torque |

### Action space — 12 dims

Position targets for 12 joints (6 per leg: Hip Roll/Pitch/Yaw, Knee Pitch, Ankle Pitch/Roll), expressed as offsets from default pose. Action scale = 1.0 rad.

### Command space — 10 dims

| Index | Command | Stage 1 range | Stages 3–4 range |
|-------|---------|---------------|-----------------|
| 0 | lin_vel_x (m/s) | [−1.2, 1.2] | [−1.2, 1.2] |
| 1 | lin_vel_y (m/s) | [−1.2, 1.2] | [−1.2, 1.2] |
| 2 | ang_vel_yaw (rad/s) | [−1.6, 1.6] | [−1.6, 1.6] |
| 3 | gait_frequency (Hz) | [1.2, 2.2] | [1.2, 2.2] |
| 4 | foot_yaw_L (rad) | [−0.7, 0.7] | [−0.7, 0.7] |
| 5 | foot_yaw_R (rad) | [−0.7, 0.7] | [−0.7, 0.7] |
| 6 | body_pitch_target (rad) | [−0.1, 0.3] | [−0.1, 0.3] |
| 7 | body_roll_target (rad) | [−0.1, 0.1] | [−0.1, 0.1] |
| 8 | feet_offset_x_target (m) | [−0.25, 0.25] | [−0.25, 0.25] |
| 9 | feet_offset_y_target (m) | [−0.2, 0.2] | [−0.2, 0.2] |

Stage 2 narrows velocity commands ([0.0, 0.8] / [−0.3, 0.3] / [−0.5, 0.5]) to avoid interfering with approach/orbit learning.

---

## Ball reset & termination

The ball spawns in front of the robot each episode (randomized forward/lateral offset). If the ball wanders out of range mid-episode without the robot falling, the ball alone is reset to the robot's front — the episode continues.

Ball-based terminations (configurable per stage):

| Condition | Stage 1–2 | Stage 3–4 |
|-----------|-----------|-----------|
| Ball stationary for > N s | disabled | 2 s |
| Ball moving for > N s | disabled | 15 s |

---

## Stage 1 — Locomotion pretraining

**Config**: `ParameterWalkKick_Stage1.yaml`  
**Purpose**: Learn robust bipedal walking with the ball present in observations but no kicking pressure. Ball-based terminations are disabled.

Ball spawns 0.2–0.8 m in front of the robot.

### Reward scales

| Reward | Scale | Description |
|--------|-------|-------------|
| survival | +0.25 | alive bonus |
| tracking_lin_vel_x | +2.5 | track commanded vx |
| tracking_lin_vel_y | +3.0 | track commanded vy |
| tracking_ang_vel | +1.5 | track commanded yaw rate |
| feet_swing | +3.0 | foot lift during swing phase |
| base_height | −20 | penalize deviation from 0.68 m |
| orientation | −20 | penalize roll/pitch error vs. commands |
| action_rate | −1.5 | penalize jerky actions |
| collision | −1.0 | penalize upper-body contacts |
| torques | −3e-4 | |
| torque_tiredness | −1e-2 | |
| power | −3e-3 | |
| lin_vel_z | −2.0 | suppress vertical bobbing |
| ang_vel_xy | −0.2 | suppress roll/pitch rates |
| dof_vel | −2e-4 | |
| dof_acc | −2e-7 | |
| root_acc | −1e-4 | |
| dof_pos_limits | −1.0 | |
| feet_slip | −0.1 | |
| feet_roll | −0.2 | |
| feet_pitch | −0.1 | |
| ankle_vel | −1e-3 | |
| ankle_acc | −1e-6 | |
| feet_offset_x | −20 | track commanded foot x separation |
| feet_offset_y | −20 | track commanded foot y separation |
| foot_yaw_L/R | −1.0 each | track commanded foot yaw angles |
| feet_yaw_diff | −0.5 | track commanded L/R yaw difference |
| feet_yaw_mean | −0.5 | track commanded average foot yaw |
| *kick rewards* | 0 | all disabled |

---

## Stage 2 — Approach & Orbit

**Config**: `ParameterWalkKick_Stage2.yaml`  
**Checkpoint**: load Stage 1  
**Purpose**: Learn to close the distance to the ball and orbit around it at ≤ 0.6 m, facing the ball at all times. No kicking yet. Ball-based terminations still disabled.

Ball spawns 0.8–1.5 m in front (always outside the orbit ring) to ensure the approach phase is exercised every episode.

### Reward scales (locomotion rewards identical to Stage 1, changes shown)

| Reward | Scale | Description |
|--------|-------|-------------|
| approach_ball | +3.0 | radial velocity toward ball while outside ring (> 0.6 m) |
| orbit_ball | +2.0 | tangential speed around ball while inside ring (≤ 0.6 m) |
| stay_in_ring | −5.0 | quadratic penalty on body-to-ball distance exceeding 0.6 m |
| face_ball | +1.0 | reward robot forward axis pointing at ball |
| kicking_foot_approach_ball_stationary | 0 | disabled (conflicts with orbit) |
| *kick rewards* | 0 | all disabled |

**Key parameters**:
- `orbit_max_radius`: 0.6 m
- `face_ball_sigma`: 0.5

---

## Stage 3 — Kick learning

**Config**: `ParameterWalkKick_Stage3.yaml`  
**Checkpoint**: load Stage 2  
**Purpose**: Enable kicking rewards while retaining spatial constraints from Stage 2. Ball-based terminations activate to pressure efficient, directed kicks.

Ball spawns 0.2–1.5 m in front (mixed near/far to exercise both approach and direct kick).

### Reward scales (locomotion rewards identical to Stage 1, changes shown)

| Reward | Scale | Description |
|--------|-------|-------------|
| ball_velocity_target_direction | +5.0 | ball speed toward target [6, 0] m, decayed over 2 s |
| kicking_foot_approach_ball_stationary | +5.0 | foot proximity to ball (σ = 0.1 m) |
| body_alignment_for_kick | +1.0 | robot forward axis aligned to kick target |
| ball_acceleration | +0.25 | instantaneous ball acceleration in +x vs lateral |
| waiting | −1.0 | quadratic episode-length penalty to encourage quick kicks |
| stay_in_ring | −3.0 | soft ring constraint retained from Stage 2 |
| face_ball | +0.5 | ball-facing retained from Stage 2 |
| body_angle | 0 | disabled — `orientation` reward already covers this |

**Key parameters**:
- `ball_target_position`: [6.0, 0.0] m (world frame)
- `ball_velocity_decay_time`: 2.0 s
- `approach_proximity_sigma`: 0.1 m
- `orbit_max_radius`: 0.6 m (for `stay_in_ring`)
- `max_ball_still_time_s`: 2.0 s (terminate if ball idle too long)
- `max_ball_moving_time_s`: 15.0 s (terminate if ball never stops after kick)

---

## Stage 4 — Robustness & sim2real hardening

**Config**: `ParameterWalkKick_Stage4.yaml`  
**Checkpoint**: load Stage 3  
**Purpose**: Same reward structure as Stage 3 with full domain randomization, noise, and perturbations enabled. The policy is hardened for sim2real transfer.

Identical reward scales to Stage 3.

**Key differences from Stage 3**:
- `use_wandb: false` by default (set to `true` for logging)
- All randomization ranges at maximum (stiffness ±5%, friction 0–0.3, base mass ±20%, CoM offsets ±0.1 m, push forces up to 15 N)
- Ball spawns 0.2–1.5 m (same as Stage 3)

---

## Running the curriculum

```bash
# Stage 1 — train from scratch
python train.py --config envs/T1/curriculum/ParameterWalkKick_Stage1.yaml

# Stage 2 — fine-tune from Stage 1 checkpoint
python train.py --config envs/T1/curriculum/ParameterWalkKick_Stage2.yaml \
    --checkpoint <stage1_checkpoint.pt>

# Stage 3 — fine-tune from Stage 2 checkpoint
python train.py --config envs/T1/curriculum/ParameterWalkKick_Stage3.yaml \
    --checkpoint <stage2_checkpoint.pt>

# Stage 4 — fine-tune from Stage 3 checkpoint
python train.py --config envs/T1/curriculum/ParameterWalkKick_Stage4.yaml \
    --checkpoint <stage3_checkpoint.pt>
```

---

## Domain randomization summary

Applied from Stage 1 onward:

| Parameter | Range | Operation |
|-----------|-------|-----------|
| joint stiffness | [0.95, 1.05] | scaling |
| joint damping | [0.95, 1.05] | scaling |
| joint friction | [0.0, 2.0] Nm | additive |
| foot contact friction | [0.0, 0.3] | additive |
| foot compliance | [0.5, 1.5] | additive |
| foot restitution | [0.0, 0.3] | additive |
| base CoM offset | ±0.1 m (xyz) | additive |
| base mass | [0.8, 1.2]× | scaling |
| other link CoM | ±0.005 m | additive |
| other link mass | [0.98, 1.02]× | scaling |
| kick impulse (linear) | 0–0.15 m/s | additive, every 12 s |
| push force | 0–15 N | additive, every 8 s for 0.5 s |
| push torque | 0–2 Nm | additive |
| init joint positions | ±0.05 rad | additive |
| init base xy | ±1.0 m | additive |
| init base xy velocity | 0–0.1 m/s | additive |
| ball init x offset | stage-dependent | additive |
| ball init y offset | stage-dependent | additive |
