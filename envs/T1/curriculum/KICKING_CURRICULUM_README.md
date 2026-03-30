# KickingCurriculum Environment

A 3-stage curriculum learning environment that teaches a T1 humanoid robot to **walk**, **navigate toward a ball**, and **kick it** --- all within a single continuous training run.

## Motivation

Training a robot to kick a ball from scratch is hard. The robot must simultaneously learn:
1. How to balance and walk
2. How to navigate toward the ball
3. How to execute a kick motion

Without curriculum learning, the reward signal for kicking is so sparse that the robot never discovers it. By breaking the problem into stages and only introducing harder objectives once easier ones are mastered, the policy learns each skill reliably.

## Architecture

```
KickingCurriculum (envs/T1/kicking_curriculum.py)
    inherits from
Kicking (envs/T1/kicking.py)
    inherits from
BaseTask (envs/base_task.py)
```

`KickingCurriculum` overrides six methods from `Kicking`:
- `_init_buffers` --- adds stage, nav command, and EMA tracking buffers
- `_reset_idx` --- assigns env stage on reset based on global unlock
- `_check_termination` --- disables ball-timer termination during Stage 0
- `_compute_observations` --- appends 3 nav command dims (44 -> 47 total)
- `_compute_reward` --- applies per-stage reward multipliers + updates EMA
- `_reward_tracking_lin_vel_x/y`, `_reward_tracking_ang_vel` --- track `nav_commands` instead of hardcoded zero

It also adds:
- `_reward_nav_ball_proximity` --- new reward for body-to-ball distance
- `_compute_nav_commands` --- heuristic controller that computes velocity commands from ball position
- `_update_env_stages` --- per-env stage transition logic
- `_global_unlock_stage` --- performance-based global stage gate

## The Three Stages

### Stage 0: Walk

**Goal**: Learn stable locomotion by tracking random velocity commands.

- **Nav commands**: Random `[lin_vel_x, lin_vel_y, ang_vel_yaw]` resampled every 8-12 seconds
- **Active rewards**: All locomotion rewards (survival, tracking, base_height, orientation, torques, feet stability, etc.)
- **Disabled rewards**: All ball/kick rewards (`ball_velocity_target_direction`, `kicking_foot_approach_ball_stationary`, `body_alignment_for_kick`, `ball_acceleration`, `waiting`, `nav_ball_proximity`)
- **Termination**: Ball-still and ball-moving timers are disabled (the ball is never kicked in this stage). Only genuine termination conditions apply: falling, velocity explosion, or episode timeout (7s).
- **Ball**: Present in the simulation but ignored by the reward function.

**Advances to Stage 1 when:**
- EMA of tracking reward (velocity-x + angular velocity) >= `stage0_tracking_threshold` (default 0.9)
- AND total env-steps >= `stage0_min_steps` (default 2 billion, ~20k iterations with 4096 envs)

### Stage 1: Approach

**Goal**: Navigate toward the ball using directed velocity commands.

- **Nav commands**: Automatically computed from ball position:
  - `lin_vel_x` = distance-to-ball x cos(heading_error), capped at `nav_max_lin_vel`
  - `ang_vel_yaw` = heading error clamped to [-`nav_max_ang_vel`, +`nav_max_ang_vel`]
  - `lin_vel_y` = 0 (no lateral commands)
- **Active rewards**: All locomotion rewards + `nav_ball_proximity` (exp decay with sigma 1.5m) + partial `body_alignment_for_kick` (x0.5) + partial `kicking_foot_approach_ball_stationary` (x0.3)
- **Disabled rewards**: `ball_velocity_target_direction`, `ball_acceleration`, `waiting`
- **Termination**: Normal kicking env termination (including ball timers)

**Advances to Stage 2 when (per-env):**
- Robot body within `kick_approach_range` (0.3m) of ball
- AND heading error < `kick_approach_heading_threshold` (0.5 rad, ~28 deg)
- AND global stage 2 is unlocked (EMA proximity >= 0.5 + min steps)

### Stage 2: Kick

**Goal**: Execute a kick to send the ball toward the target position.

- **Nav commands**: Set to zero (the policy must act autonomously)
- **Active rewards**: Full kicking rewards from the base `Kicking` env (`ball_velocity_target_direction` x10, `kicking_foot_approach_ball_stationary` x10, `body_alignment_for_kick` x1, `ball_acceleration` x0.25, `waiting` x-1)
- **Disabled rewards**: All tracking rewards (`tracking_lin_vel_x/y`, `tracking_ang_vel`) and `nav_ball_proximity`
- **Ball rolling scale**: When ball is moving, `waiting` is zeroed and `survival` halved (inherited from base Kicking env)

**Returns to Stage 1 when:**
- Ball speed exceeds 0.1 m/s (kick detected) -> ball resets in front of robot -> env returns to approach stage

## Observation Space (47 dims)

| Component | Dims | Description |
|-----------|------|-------------|
| Projected gravity | 3 | Gravity vector in robot frame |
| Base angular velocity | 3 | Robot angular velocity in robot frame |
| Relative ball position | 2 | Ball XY position relative to robot (robot frame) |
| Navigation commands | 3 | `[lin_vel_x, lin_vel_y, ang_vel_yaw]` --- **new** |
| Joint positions | 12 | `dof_pos - default_dof_pos` |
| Joint velocities | 12 | `dof_vel` |
| Previous actions | 12 | Actions from last step |

The 3 nav command dims are appended after the base 44-dim Kicking observation. The policy network (ActorCritic) automatically adapts to the 47-dim input.

## Privileged Observation Space (20 dims, unchanged)

| Component | Dims |
|-----------|------|
| Base mass + COM offset | 4 |
| Base linear velocity | 3 |
| Base height | 1 |
| Ball linear velocity XY | 2 |
| Left foot position XY | 2 |
| Right foot position XY | 2 |
| Pushing forces | 3 |
| Pushing torques | 3 |

## Reward Function Summary

### Locomotion rewards (active in all stages unless multiplied to 0)

| Reward | Scale | Description |
|--------|-------|-------------|
| `survival` | +0.25 | Alive bonus |
| `tracking_lin_vel_x` | +1.0 | Track nav_commands[0] (overridden to use nav commands) |
| `tracking_lin_vel_y` | +1.0 | Track nav_commands[1] |
| `tracking_ang_vel` | +0.25 | Track nav_commands[2] |
| `base_height` | -200 | Penalize deviation from 0.68m target height |
| `orientation` | -20 | Penalize non-upright orientation |
| `torques` | -2e-4 | Penalize joint torques |
| `torque_tiredness` | -1e-2 | Penalize sustained torque |
| `power` | -2e-3 | Penalize mechanical power |
| `lin_vel_z` | -1.5 | Penalize vertical body velocity |
| `ang_vel_xy` | -0.1 | Penalize roll/pitch angular velocity |
| `dof_vel` | -3e-4 | Penalize joint velocities |
| `dof_acc` | -1e-7 | Penalize joint accelerations |
| `root_acc` | -1e-5 | Penalize body accelerations |
| `action_rate` | -1.5 | Penalize action changes between steps |
| `dof_pos_limits` | -1 | Penalize hitting joint limits |
| `feet_slip` | -1 | Penalize foot sliding on ground |
| `feet_yaw_diff` | -3 | Penalize feet pointing different directions |
| `feet_yaw_mean` | -3 | Penalize feet yaw diverging from body yaw |
| `feet_roll` | -0.3 | Penalize foot roll |

### Approach rewards (active in Stage 1)

| Reward | Scale | Stage 1 mult | Description |
|--------|-------|-------------|-------------|
| `nav_ball_proximity` | +5.0 | x1.0 | exp(-dist_to_ball / 1.5m) |
| `body_alignment_for_kick` | +1.0 | x0.5 | Robot facing ball-to-goal axis |
| `kicking_foot_approach_ball_stationary` | +10 | x0.3 | Foot proximity to stationary ball |

### Kick rewards (active in Stage 2)

| Reward | Scale | Description |
|--------|-------|-------------|
| `ball_velocity_target_direction` | +10 | Ball velocity projected toward target [6,0], with time decay |
| `kicking_foot_approach_ball_stationary` | +10 | Foot proximity to stationary ball |
| `body_alignment_for_kick` | +1 | Robot facing ball-to-goal axis |
| `ball_acceleration` | +0.25 | Instantaneous ball acceleration toward target |
| `waiting` | -1 | Penalize time spent before kicking (zeroed when ball is rolling) |
| `body_angle` | +0.1 | Penalize body pitch/roll deviation |

### Stage reward multiplier table

Multipliers from `stage_reward_multipliers` in the YAML. A value of 0.0 disables the reward; unlisted rewards default to 1.0.

| Reward | Stage 0 | Stage 1 | Stage 2 |
|--------|---------|---------|---------|
| `nav_ball_proximity` | 0.0 | 1.0 | 0.0 |
| `ball_velocity_target_direction` | 0.0 | 0.0 | 1.0 |
| `kicking_foot_approach_ball_stationary` | 0.0 | 0.3 | 1.0 |
| `body_alignment_for_kick` | 0.0 | 0.5 | 1.0 |
| `body_angle` | 0.0 | 1.0 | 1.0 |
| `ball_acceleration` | 0.0 | 0.0 | 1.0 |
| `waiting` | 0.0 | 0.0 | 1.0 |
| `tracking_lin_vel_x` | 1.0 | 1.0 | 0.0 |
| `tracking_lin_vel_y` | 1.0 | 1.0 | 0.0 |
| `tracking_ang_vel` | 1.0 | 1.0 | 0.0 |

## Stage Advancement

Stage advancement is **performance-based**, not fixed-step. Two signals are tracked as exponential moving averages (EMA, alpha=0.005):

### Stage 0 -> Stage 1 (walk -> approach)

Both conditions must be true:
1. `ema_tracking >= stage0_tracking_threshold` (default **0.9**) --- the robot consistently tracks 90% of random velocity commands
2. `total_env_steps >= stage0_min_steps` (default **2 billion**, ~20k iterations with 4096 envs) --- safety floor to prevent premature advancement

### Stage 1 -> Stage 2 (approach -> kick)

**Global unlock** (both required):
1. `ema_proximity >= stage1_proximity_threshold` (default **0.5**) --- the robot regularly gets within ~1.5m of the ball
2. `total_env_steps >= stage1_min_steps` (default **3 billion**, ~30k iterations)

**Per-env trigger** (once global unlock is achieved):
- `dist_to_ball < kick_approach_range` (0.3m) AND `heading_error < 0.5 rad` (~28 deg)
- That specific env enters Stage 2; others remain in Stage 1

### Stage 2 -> Stage 1 (kick -> approach)

- Per-env: when `ball_speed > 0.1 m/s` (kick detected), the env returns to approach and the ball is reset

## WandB Metrics

The following metrics are logged every iteration:

| Metric | Description |
|--------|-------------|
| `curriculum/unlock_stage` | Current global unlock level (0, 1, or 2) |
| `curriculum/ema_tracking` | Walk quality EMA (0 to 1, triggers at 0.9) |
| `curriculum/ema_proximity` | Approach quality EMA (0 to 1, triggers at 0.5) |
| `curriculum/stage0_envs` | Number of envs in walk stage |
| `curriculum/stage1_envs` | Number of envs in approach stage |
| `curriculum/stage2_envs` | Number of envs in kick stage |
| `episode/tracking_lin_vel_x` | Per-episode tracking reward |
| `episode/nav_ball_proximity` | Per-episode approach proximity reward |
| `episode/ball_velocity_target_direction` | Per-episode kick velocity reward |

## How to Train

```bash
# Full 3-stage curriculum from scratch (recommended)
python train.py --task T1/KickingCurriculum --num_envs 4096 --sim_device cuda:0 --rl_device cuda:0

# Resume from a checkpoint
python train.py --task T1/KickingCurriculum --num_envs 4096 --checkpoint -1

# Skip walk stage (load a pre-trained ParameterWalk checkpoint)
# 1. Set initial_stage: 1 in KickingCurriculum.yaml
# 2. Run:
python train.py --task T1/KickingCurriculum --num_envs 4096 --checkpoint path/to/parameter_walk_model.pth

# Play / visualize a trained policy
python play.py --task T1/KickingCurriculum --checkpoint -1
```

## Configuration Reference

All curriculum-specific parameters live under the `curriculum:` key in `KickingCurriculum.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_stage` | 0 | Skip earlier stages (0=full, 1=skip walk, 2=skip walk+approach) |
| `stage0_tracking_threshold` | 0.9 | EMA tracking quality to exit Stage 0 |
| `stage0_min_steps` | 2,000,000,000 | Min env-steps before Stage 1 unlocks (~20k iter) |
| `stage1_proximity_threshold` | 0.5 | EMA proximity quality to exit Stage 1 |
| `stage1_min_steps` | 3,000,000,000 | Min env-steps before Stage 2 unlocks (~30k iter) |
| `kick_approach_range` | 0.3 m | Per-env distance trigger for kick stage |
| `kick_approach_heading_threshold` | 0.5 rad | Per-env heading trigger for kick stage |
| `nav_max_lin_vel` | 0.5 m/s | Max forward velocity in approach nav commands |
| `nav_max_ang_vel` | 1.0 rad/s | Max angular velocity in approach nav commands |
| `stage0_lin_vel_range` | [-0.5, 0.5] | Random walk command range (m/s) |
| `stage0_ang_vel_range` | [-1.0, 1.0] | Random walk angular command range (rad/s) |
| `stage0_resample_time_lo_s` | 8.0 s | Min time between walk command resamples |
| `stage0_resample_time_hi_s` | 12.0 s | Max time between walk command resamples |
| `nav_ball_proximity_sigma` | 1.5 m | Decay distance for proximity reward |
| `stage_reward_multipliers` | (see above) | Per-stage reward on/off switches |

## Files

| File | Description |
|------|-------------|
| `envs/T1/kicking_curriculum.py` | Environment class (448 lines) |
| `envs/T1/KickingCurriculum.yaml` | Configuration file |
| `envs/T1/kicking.py` | Parent class (base Kicking environment) |
| `envs/__init__.py` | Registers `KickingCurriculum` for dynamic loading |
| `utils/runner.py` | Training loop (modified to log curriculum metrics) |

## Design Decisions

1. **Single policy, not hierarchical**: One neural network learns all three skills. The curriculum only controls which rewards are active. This avoids the complexity of switching between separate policies and allows skills to transfer naturally.

2. **Performance-based advancement**: The curriculum advances based on EMA of actual reward quality, not fixed step counts. This adapts to different hardware speeds and hyperparameter choices. A minimum step floor prevents premature advancement from noise.

3. **Per-env stage in Stage 2**: When kick is unlocked, individual environments independently cycle between approach and kick based on ball proximity. This means the policy sees a natural mix of approach and kick situations in every training batch.

4. **Nav commands as heuristic, not learned**: The approach navigation uses a simple bearing controller (turn toward ball, walk forward). This is not learned --- it's computed from ball position and fed as commands. The policy learns to *follow* these commands, just like it learned to follow random walk commands in Stage 0.

5. **Ball-timer termination disabled in Stage 0**: The base Kicking env terminates episodes after 2 seconds of ball inactivity. Since the ball is never kicked in Stage 0, this would kill every episode at 2s, making walk training impossible. The override recomputes termination for Stage-0 envs excluding ball-related conditions.
