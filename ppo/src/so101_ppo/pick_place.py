"""Fixed pick-and-place geometry and shared CPU / batched policy contract.

All lengths are metres, quaternions are wxyz, and rewards are per transition.
The brick is an ordinary rigid body: no welds or scripted attachment.
"""
from dataclasses import asdict, dataclass
import json

import mujoco
import numpy as np
import torch

from .contract import CONTROL_DT, JOINTS, MAX_COMMAND_SPEED, PHYSICS_DT


@dataclass(frozen=True)
class PickPlaceConfig:
    brick_size: tuple = (.032, .016, .010)
    brick_mass: float = .0025
    friction: tuple = (.8, .01, .001)
    brick_position: tuple = (.22, 0., .005)
    brick_quaternion: tuple = (2**-.5, 0., 0., 2**-.5)
    target_position: tuple = (.22, .09, .005)
    square_size: float = .08
    start_q: tuple = (-.0021927307, -.1077904910, .2375676212, 1.4410191963, .0464867841, .5)
    episode_seconds: float = 60.
    lift_clearance: float = .02
    lift_hold_seconds: float = .2
    settle_seconds: float = .5
    settle_linear_speed: float = .02
    settle_angular_speed: float = .2
    touch_bonus: float = 1.
    pickup_bonus: float = 10.
    placement_bonus: float = 100.
    reward_version: int = 2
    approach_distance_scale: float = .08
    carry_distance_scale: float = .15
    approach_reward_rate: float = .4
    grasp_reward_rate: float = .15
    lift_reward_rate: float = .2
    carry_reward_rate: float = .25
    joint_velocity_cost: float = .001
    command_motion_cost: float = .002


CONFIG = PickPlaceConfig()
OBS_LAYOUT = {
    "joint_position_minus_start_rad": [0, 6],
    "joint_velocity_rad_s_times_0.1": [6, 12],
    "previous_command_normalized": [12, 18],
    "brick_position_in_base_m": [18, 21],
    "brick_quaternion_in_base_wxyz": [21, 25],
    "brick_linear_velocity_in_base_m_s": [25, 28],
    "brick_angular_velocity_in_base_rad_s": [28, 31],
    "brick_minus_tool_in_base_m": [31, 34],
    "target_position_in_base_m": [34, 37],
    "contacts_fixed_finger_moving_finger_robot_floor": [37, 41],
    "milestones_touched_picked_placed": [41, 44],
    "lift_and_settle_hold_fraction": [44, 46],
}


def action_bounds(calibration):
    model = mujoco.MjModel.from_xml_path(str(calibration.xml))
    ranges = np.array([model.joint(j).range for j in JOINTS])
    lower = np.maximum(calibration.lower, ranges[:, 0]).astype(np.float32)
    upper = np.minimum(calibration.upper, ranges[:, 1]).astype(np.float32)
    if np.any(lower >= upper) or np.any(np.array(CONFIG.start_q) < lower) or np.any(np.array(CONFIG.start_q) > upper):
        raise ValueError("Pick-place start or action range is outside calibrated joint limits")
    return lower, upper


def manifest(calibration):
    lower, upper = action_bounds(calibration)
    return {
        "schema": 5, "task": "pick-place", "hardware_ready": False,
        "joint_order": list(JOINTS), "active_joint_order": list(JOINTS),
        "observation_shape": [1, 46], "action_shape": [1, 6],
        "collision_model": "original jaw meshes clipped into eight convex slabs each, v1",
        "contact_solver": {"cone": "elliptic", "impratio": 10, "iterations": 100},
        "observation_layout": OBS_LAYOUT, "pick_place": json.loads(json.dumps(asdict(CONFIG))),
        "command_bounds_rad": [lower.tolist(), upper.tolist()],
        "action_center_rad": ((lower + upper) / 2).tolist(),
        "action_scale_rad": ((upper - lower) / 2).tolist(),
        "control_hz": 1 / CONTROL_DT, "physics_dt": PHYSICS_DT,
        "max_command_speed_rad_s": MAX_COMMAND_SPEED,
        "onnx_includes_observation_normalizer": True,
        "target_frame": "fixed robot base, metres", "tool_site": "gripperframe",
        "command_quantization": "Round to calibrated encoder ticks after the continuous command slew limiter",
    }


def validate_options(mode="fixed", start_mode="home", workspace=None, robust=False, targets="auto"):
    if mode != "fixed" or start_mode != "home" or robust or targets == "rotate":
        raise ValueError("Pick-place v1 requires fixed layout, home start, nominal physics, and fixed previews")
    if workspace not in (None, "near"):
        raise ValueError("--workspace applies to reaching; pick-place uses its own six-joint bounds")


def add_brick(spec):
    body = spec.worldbody.add_body(name="brick", pos=CONFIG.brick_position, quat=CONFIG.brick_quaternion)
    body.add_freejoint(name="brick_free")
    body.add_geom(name="brick_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=np.array(CONFIG.brick_size) / 2, mass=CONFIG.brick_mass,
                  friction=CONFIG.friction, condim=4, rgba=(.85, .18, .08, 1))


def brick_spec():
    spec = mujoco.MjSpec()
    add_brick(spec)
    return spec


def split_finger_collisions(spec):
    """Slab hulls preserve the fingertip profile lost by a whole-jaw convex hull.

    Clip original mesh edges at slab boundaries, then hull each section. This
    keeps the original outer surface and explicit body inertias; no extra pads.
    """
    model = spec.compile()
    for name, axis in (("fixed_finger", 2), ("moving_finger", 1)):
        geom = spec.geom(name)
        gid = model.geom(name).id
        mid = model.geom_dataid[gid]
        vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]].astype(float)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
        vertices = vertices @ rotation.reshape(3, 3).T + model.geom_pos[gid]
        faces = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid]+model.mesh_facenum[mid]]
        edges = np.concatenate((faces[:, :2], faces[:, 1:], faces[:, (0, 2)]))
        a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
        cuts = np.linspace(vertices[:, axis].min(), vertices[:, axis].max(), 9)
        for i, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
            points = [vertices[(vertices[:, axis] >= lo) & (vertices[:, axis] <= hi)]]
            for cut in (lo, hi):
                crossing = (a[:, axis]-cut)*(b[:, axis]-cut) < 0
                aa, bb = a[crossing], b[crossing]
                t = (cut-aa[:, axis])/(bb[:, axis]-aa[:, axis])
                points.append(aa+t[:, None]*(bb-aa))
            points = np.unique(np.concatenate(points).round(9), axis=0)
            meshname = f"{name}_slab_{i}"
            spec.add_mesh(name=meshname, uservert=points.ravel())
            geom.parent.add_geom(name=f"{name}_{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=meshname, group=3, contype=1, conaffinity=1,
                friction=CONFIG.friction, condim=4, mass=0.)
        spec.delete(geom)


def robot_spec(calibration, with_brick=False):
    # Read the original snapshot: reaching's robot_spec deliberately disables collisions.
    spec = mujoco.MjSpec.from_file(str(calibration.xml))
    for actuator in spec.actuators:
        actuator.ctrllimited = actuator.forcelimited = False
    for i, geom in enumerate(spec.geoms):
        if geom.group == 3:
            geom.name = f"collision_{geom.parent.name}_{i}"
            geom.contype = geom.conaffinity = 1
            geom.friction = CONFIG.friction
            geom.condim = 4
            if geom.meshname == "wrist_roll_follower_so101_v1":
                geom.name = "fixed_finger"
            elif geom.parent.name == "moving_jaw_so101_v1":
                geom.name = "moving_finger"
    split_finger_collisions(spec)
    floors = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    if not floors:
        floors = [spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1, 1, .01))]
    for floor in floors:
        floor.name = "floor"
        floor.contype = floor.conaffinity = 1
        floor.friction = CONFIG.friction
        floor.rgba = (.18, .20, .23, 1)
    x, y, _ = CONFIG.target_position
    half = CONFIG.square_size / 2
    for i, (pos, size) in enumerate((
        ((x-half, y, .0002), (.001, half, .0001)),
        ((x+half, y, .0002), (.001, half, .0001)),
        ((x, y-half, .0002), (half, .001, .0001)),
        ((x, y+half, .0002), (half, .001, .0001)),
    )):
        spec.worldbody.add_geom(name=f"square_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos, size=size, contype=0, conaffinity=0, rgba=(.2, .95, .35, 1))
    spec.worldbody.add_light(name="task_light", pos=(.1, -.3, .8), dir=(.1, .3, -.8),
        diffuse=(.8, .8, .8), ambient=(.25, .25, .25))
    if with_brick:
        add_brick(spec)
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.iterations = 100
    spec.option.timestep = PHYSICS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return spec


def next_command(action, previous, lower, upper):
    action = np.asarray(action)
    if action.shape != (6,) or not np.isfinite(action).all():
        raise ValueError("Expected six finite pick-place actions")
    center, scale = (lower + upper) / 2, (upper - lower) / 2
    desired = center + scale * np.clip(action, -1, 1)
    return np.clip(previous + np.clip(desired - previous,
        -MAX_COMMAND_SPEED * CONTROL_DT, MAX_COMMAND_SPEED * CONTROL_DT), lower, upper)


def rotation_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


class TaskState:
    """Shared vectorized milestone logic; callers advance exactly once per step."""
    def __init__(self, count, device="cpu"):
        self.milestones = torch.zeros(count, 3, dtype=torch.bool, device=device)
        self.holds = torch.zeros(count, 2, dtype=torch.long, device=device)
        self.reward = torch.zeros(count, device=device)
        # Unscaled per-transition components for training diagnostics.
        self.reward_components = torch.zeros(count, 5, device=device)
        self.failed = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids=slice(None)):
        for value in (self.milestones, self.holds, self.reward, self.reward_components, self.failed):
            value[ids] = 0

    def dense_reward(self, pos, tool, target, bottom, contacts):
        """Positive state rewards: nearer > farther, including on the first step.

        Dense terms total at most one reward/second (60 per full episode), below
        the placement bonus. Grasp/lift/carry require current bilateral contact;
        carry also requires a previous pickup and current floor clearance.
        """
        distance = torch.linalg.vector_norm(pos-tool, dim=-1)
        approach = CONFIG.approach_reward_rate * (1-torch.tanh(distance/CONFIG.approach_distance_scale))
        grasped = contacts[:, 0] & contacts[:, 1]
        grasp = CONFIG.grasp_reward_rate * grasped.float()
        lift = CONFIG.lift_reward_rate * grasped * (bottom/CONFIG.lift_clearance).clamp(0, 1)
        carrying = grasped & self.milestones[:, 1] & (bottom >= CONFIG.lift_clearance)
        goal_distance = torch.linalg.vector_norm(pos[:, :2]-target[:, :2], dim=-1)
        carry = CONFIG.carry_reward_rate * carrying * (1-torch.tanh(goal_distance/CONFIG.carry_distance_scale))
        return CONTROL_DT * (approach + grasp + lift + carry)

    def advance(self, q, dq, pos, quat, velocity, tool, target, contacts, delta):
        old = self.milestones.clone()
        half = torch.tensor(CONFIG.brick_size, device=pos.device) / 2
        extent = (rotation_matrix(quat).abs() @ half)
        bottom = pos[:, 2] - extent[:, 2]
        lifting = contacts[:, 0] & contacts[:, 1] & (bottom >= CONFIG.lift_clearance)
        self.holds[:, 0] = torch.where(lifting, self.holds[:, 0] + 1, 0)
        self.milestones[:, 0] |= contacts[:, :2].any(-1)
        self.milestones[:, 1] |= self.holds[:, 0] >= round(CONFIG.lift_hold_seconds / CONTROL_DT)
        inside = ((pos[:, :2]-target[:, :2]).abs() + extent[:, :2] <= CONFIG.square_size/2).all(-1)
        settled = (self.milestones[:, 1] & inside & contacts[:, 3] & ~contacts[:, 2]
            & (torch.linalg.vector_norm(velocity[:, :3], dim=-1) < CONFIG.settle_linear_speed)
            & (torch.linalg.vector_norm(velocity[:, 3:], dim=-1) < CONFIG.settle_angular_speed))
        self.holds[:, 1] = torch.where(settled, self.holds[:, 1] + 1, 0)
        self.milestones[:, 2] |= self.holds[:, 1] >= round(CONFIG.settle_seconds / CONTROL_DT)
        finite = torch.cat((q, dq, pos, quat, velocity), -1).isfinite().all(-1)
        self.failed[:] = (~finite | (dq.abs().amax(-1) > 20) | (pos[:, 2] < -.02)
                          | (torch.linalg.vector_norm(pos, dim=-1) > .65))
        self.milestones[:, 2] &= ~self.failed
        shaping = self.dense_reward(pos, tool, target, bottom, contacts)
        shaping = torch.where(self.failed, 0., shaping)
        bonuses = torch.tensor((CONFIG.touch_bonus, CONFIG.pickup_bonus, CONFIG.placement_bonus), device=q.device)
        penalty = CONTROL_DT * (CONFIG.joint_velocity_cost*dq.square().sum(-1)
            + CONFIG.command_motion_cost*(delta/(CONTROL_DT*MAX_COMMAND_SPEED)).square().sum(-1))
        milestone_rewards = (self.milestones & ~old).float()*bonuses
        self.reward_components[:] = torch.cat((milestone_rewards, shaping[:, None], -penalty[:, None]), -1)
        self.reward_components.nan_to_num_(nan=0., posinf=0., neginf=0.)
        self.reward[:] = torch.nan_to_num(milestone_rewards.sum(-1) + shaping - penalty, nan=0., posinf=0., neginf=0.)

    def observation(self, q, dq, command, lower, upper, pos, quat, velocity, tool, target, contacts):
        home = torch.tensor(CONFIG.start_q, device=q.device)
        hold_limits = torch.tensor((CONFIG.lift_hold_seconds, CONFIG.settle_seconds), device=q.device) / CONTROL_DT
        return torch.cat((q-home, dq*.1, (command-(lower+upper)/2)/((upper-lower)/2),
            pos, quat, velocity, pos-tool, target, contacts.float(), self.milestones.float(),
            (self.holds / hold_limits).clamp(max=1)), -1)
