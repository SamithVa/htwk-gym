# Policy Switch Logic for sim2sim_0424

This document summarizes the policy transition state machine in [sim2sim_0424.py](sim2sim_0424.py) and highlights code sections that can be reused for real deployment.

## 1. State Machine Overview

States:
1. WALK
2. KICK
3. RECOVER
4. FALLEN

Defined in [sim2sim_0424.py](sim2sim_0424.py#L28).

Core thresholds and timers:
1. RECOVER_STEPS = 30
2. KICK_SETTLE_STEPS = 25
3. BALL_KICK_THRESHOLD = 0.15 m
4. FALLEN_GRAVITY_Z = 0.5

Defined in [sim2sim_0424.py](sim2sim_0424.py#L33).

Runtime transitions are implemented in [sim2sim_0424.py](sim2sim_0424.py#L385).

## 2. Transition Conditions

### A) Any State to FALLEN
Condition:
1. If robot is not already in FALLEN
2. And projected gravity indicates unsafe orientation: -proj_gravity_z < FALLEN_GRAVITY_Z

Code:
1. [fallen check](sim2sim_0424.py#L399)
2. [fallen handling](sim2sim_0424.py#L403)

Action:
1. Set mode to FALLEN
2. Zero control command
3. Keep simulation stepping

### B) WALK to KICK
Condition:
1. Horizontal distance to ball is below switch distance
2. state.dist_xy <= args.switch_dist

Code:
1. [walk to kick trigger](sim2sim_0424.py#L410)
2. Switch distance argument in [sim2sim_0424.py](sim2sim_0424.py#L110)

Action:
1. Set mode to KICK
2. Save kick start ball position in episode.kick_ball_start

### C) KICK internal kick detection
Condition:
1. Ball displacement from kick start exceeds threshold
2. ball_moved > BALL_KICK_THRESHOLD

Code:
1. [ball moved calculation and threshold](sim2sim_0424.py#L415)

Action:
1. Start settle timer episode.kick_settle_remaining = KICK_SETTLE_STEPS

### D) KICK to RECOVER
Condition:
1. Settle timer counts down to zero

Code:
1. [kick settle countdown and transition](sim2sim_0424.py#L419)

Action:
1. Set mode to RECOVER
2. Save current joint pose as recover_start_pos
3. Reset recover_step

### E) RECOVER to WALK
Condition:
1. recover_step reaches RECOVER_STEPS

Code:
1. [recover interpolation and completion](sim2sim_0424.py#L427)

Action:
1. Interpolate joint target from recover_start_pos back to default_joint_pos
2. Return to WALK mode when done

## 3. Policy Routing

Policy selection per state:
1. WALK uses walk policy and walk observation
2. KICK uses kick policy and kick observation
3. RECOVER bypasses policy and uses interpolation target
4. FALLEN bypasses policy and applies zero control

Code:
1. [walk and kick routing](sim2sim_0424.py#L439)
2. [walk observation builder](sim2sim_0424.py#L324)
3. [kick observation builder](sim2sim_0424.py#L344)
4. [recover interpolation path](sim2sim_0424.py#L427)

## 4. Control Execution Pattern

Each cycle:
1. Read state
2. Update mode based on transition rules
3. Build observation or recovery target
4. Infer action if in WALK or KICK
5. Convert action to joint target
6. Apply PD control with decimation

Code:
1. [run loop start](sim2sim_0424.py#L393)
2. [policy inference and target update](sim2sim_0424.py#L446)
3. [pd execution](sim2sim_0424.py#L358)

## 5. Deployment Mapping Guidance

To apply this on real robot deployment, keep the same state machine and replace sim measurements with onboard sources:
1. state.dist_xy from vision or fused world estimate
2. projected gravity from IMU orientation estimate
3. ball_moved from tracked ball position over time
4. joint state from motor feedback

Practical additions for hardware:
1. Debounce each transition condition for N cycles
2. Add minimum dwell time in each state
3. Add timeout fallback from KICK to RECOVER if ball tracking is lost
4. Keep FALLEN as highest-priority safety override

## 6. Minimal Pseudocode

    if not FALLEN and tilt_bad:
        mode = FALLEN

    if mode == FALLEN:
        cmd = 0
    elif mode == WALK and dist_to_ball <= switch_dist:
        mode = KICK
        kick_ball_start = ball_pos
    elif mode == KICK:
        if norm(ball_pos_xy - kick_ball_start_xy) > kick_threshold and settle == 0:
            settle = KICK_SETTLE_STEPS
        if settle > 0:
            settle -= 1
            if settle == 0:
                mode = RECOVER
                recover_start = joint_pos
                recover_step = 0
    elif mode == RECOVER:
        alpha = min(recover_step / RECOVER_STEPS, 1)
        target = (1 - alpha) * recover_start + alpha * default_joint_pos
        recover_step += 1
        if recover_step >= RECOVER_STEPS:
            mode = WALK
