"""Shared policy I/O. All positions are radians, distances are metres.

Joint angles use the bundled SO-101 new-calibration MJCF convention, not raw
encoder ticks, LeRobot's normalized actions, or an assumed hardware zero.
"""

import numpy as np

JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
ACTIVE_JOINTS = JOINTS[:3]
HOME = np.array([0.0, -0.5, 0.5, 0.3, 0.0, 0.0], dtype=np.float32)
ACTION_SCALE = np.array([0.35, 0.4, 0.4], dtype=np.float32)
MAX_COMMAND_SPEED = np.deg2rad(8.0)  # Match calibration's software slew limiter.
PHYSICS_DT = 0.002
DECIMATION = 10
CONTROL_DT = PHYSICS_DT * DECIMATION
EPISODE_SECONDS = 4.0
EPISODE_STEPS = round(EPISODE_SECONDS / CONTROL_DT)
SUCCESS_DISTANCE = 0.015
SUCCESS_SPEED = 0.15  # max absolute joint velocity, rad/s
HOLD_STEPS = round(1.0 / CONTROL_DT)
SITE = "gripperframe"
OBS_DIM = 21
ACTION_DIM = 3
OBS_LAYOUT = {
    "joint_position_minus_home_rad": [0, 6],
    "joint_velocity_rad_s_times_0.1": [6, 12],
    "target_position_in_base_m": [12, 15],
    "target_minus_tool_position_in_base_m": [15, 18],
    "previous_command_minus_home_divided_by_action_scale": [18, 21],
}


def observation(q, dq, goal, tool, command, center=None, scale=None):
    """Raw, unnormalized actor input. ONNX contains the trained normalizer."""
    center = HOME[:3] if center is None else center
    scale = ACTION_SCALE if scale is None else scale
    return np.concatenate([
        np.asarray(q) - HOME,
        np.asarray(dq) * 0.1,
        np.asarray(goal),
        np.asarray(goal) - tool,
        (np.asarray(command)[..., :3] - center) / scale,
    ], axis=-1).astype(np.float32)


def next_command(action, previous, bounds=None, center=None, scale=None):
    """Absolute position targets in the selected action range, with a 50 Hz slew limit."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != ACTION_DIM or not np.isfinite(action).all():
        raise ValueError("Expected finite (..., 3) policy actions")
    center = HOME[:3] if center is None else center
    scale = ACTION_SCALE if scale is None else scale
    desired = center + scale * np.clip(action, -1.0, 1.0)
    previous = np.asarray(previous)
    result = np.broadcast_to(HOME, previous.shape).copy()
    limit = MAX_COMMAND_SPEED * CONTROL_DT
    result[..., :3] = previous[..., :3] + np.clip(
        desired - previous[..., :3], -limit, limit
    )
    result[..., :3] = np.clip(
        result[..., :3], center - scale, center + scale
    )
    if bounds is not None:
        result[..., :3] = np.clip(result[..., :3], bounds[0], bounds[1])
    return result.astype(np.float32)


def reward_rate(distance, dq, command_delta, distance_scale=.10):
    """Reward per second. Both simulators multiply it by CONTROL_DT."""
    coarse = 2.0 * (1.0 - np.tanh(distance / distance_scale))
    fine = np.exp(-0.5 * (distance / 0.02) ** 2)
    velocity_cost = 0.01 * np.square(dq).sum(axis=-1)
    command_cost = 0.02 * np.square(
        command_delta / (CONTROL_DT * MAX_COMMAND_SPEED)
    ).sum(axis=-1)
    return coarse + fine - velocity_cost - command_cost


def policy_manifest(workspace=None):
    result = {
        "schema": 2,
        "hardware_ready": False,
        "joint_convention": "TheRobotStudio SO101 so101_new_calib.xml",
        "joint_order": list(JOINTS),
        "active_joint_order": list(ACTIVE_JOINTS),
        "home_rad": HOME.tolist(),
        "action_scale_rad": ACTION_SCALE.tolist(),
        "max_command_speed_rad_s": MAX_COMMAND_SPEED,
        "control_hz": 1.0 / CONTROL_DT,
        "physics_dt": PHYSICS_DT,
        "decimation": DECIMATION,
        "observation_shape": [1, OBS_DIM],
        "action_shape": [1, ACTION_DIM],
        "observation_layout": OBS_LAYOUT,
        "onnx_includes_observation_normalizer": True,
        "target_frame": "fixed robot base, metres",
        "tool_site": SITE,
        "command_quantization": "Round to calibrated encoder ticks after the continuous command slew limiter",
        "velocity_observation": "Simulator joint velocity; physical velocity estimation still needs validation",
        "success": {"distance_m": SUCCESS_DISTANCE,
                    "max_joint_speed_rad_s": SUCCESS_SPEED,
                    "consecutive_hold_s": HOLD_STEPS * CONTROL_DT},
    }

    if workspace is not None and workspace.name == "wide":
        result.update(schema=3, action_center_rad=workspace.center.tolist(),
                      action_scale_rad=workspace.scale.tolist(), **workspace.metadata())
        result["observation_layout"] = dict(OBS_LAYOUT)
        del result["observation_layout"]["previous_command_minus_home_divided_by_action_scale"]
        result["observation_layout"]["previous_command_minus_action_center_divided_by_action_scale"] = [18, 21]
    return result
