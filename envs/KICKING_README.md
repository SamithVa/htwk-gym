# Kicking Environments Overview

## `kicking.py` — Base (`Kicking`)

- Ball spawned in front of robot with independent x/y offset randomization
- Termination on contact, velocity, height, episode timeout, and ball still/moving too long
- Approach reward uses whichever foot is closer (min of both feet)
- No kick detection — rewards ball velocity toward target continuously, decaying with `time_since_ball_is_moving_buf`
- Body alignment reward is always active

---

## `kicking_robust.py` — (`KickingRobust`)

**Adds kick detection** (`kick_detected_buf`):
- Kick is detected when ball forward speed increases > threshold and ball speed > minimum
- Tracks `time_since_kick_buf` and `stable_hold_time_buf` post-kick

**Termination**:
- No-kick timeout if kick not detected within `kick_timeout_s`
- Success = kick detected + ball forward travel > `min_ball_travel_x` + stable hold > `post_kick_hold_s`

**Rewards**:
- Ball velocity toward target only active after kick, decays with `time_since_kick_buf`
- Body alignment reward zeroed after kick detected
- Approach reward uses **fixed foot index** (`kicking_foot_index` from config)
- Waiting reward zeroed after kick detected
- Adds `_reward_post_kick_stability`: soft continuous stability signal after kick

---

## `kicking_robust_bilateral.py` — (`KickingRobustBilateral`)

**Extends KickingRobust with bilateral kicking + stability focus:**

- **Dual-foot kicking**: approach reward uses `min(left_dist, right_dist)` — robot chooses the better foot
- **Polar-coordinate ball spawn**: ball appears at random angle (±45°) and distance (0.30–0.45 m) in a fan in front, replacing independent x/y offsets
- **Init height**: `0.72 → 0.70 m` to match walk policy
- Adds `_reward_kick_foot_velocity_penalty`: penalizes foot speed > 1.5 m/s before kick to discourage falls
- Adds `_reward_post_kick_return_to_default`: after kick, rewards returning to default joint angles (`exp(-dof_error² / sigma_sq)`) for walk-ready posture
- Post-kick stability scale raised; ball velocity reward scale lowered post-kick

| Feature | `kicking` | `kicking_robust` | `kicking_robust_bilateral` |
|---|---|---|---|
| Kick detection | No | Yes | Yes |
| Foot selection | Both (min) | Fixed index | Both (min) |
| Ball spawn | x/y offset | x/y offset | Polar (angle + dist) |
| No-kick timeout | No | Yes | Yes |
| Success termination | No | Yes | Yes |
| Foot velocity penalty | No | No | Yes |
| Return-to-default reward | No | No | Yes |
| Post-kick stability reward | No | Yes | Yes |
