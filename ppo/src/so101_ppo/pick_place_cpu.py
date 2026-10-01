"""CPU MuJoCo pick-place runtime using the same task state as training."""
from functools import partial
import time

import mujoco
import numpy as np
import torch

from .contract import CONTROL_DT, DECIMATION
from .cpu import CpuArm
from .pick_place import CONFIG, TaskState, action_bounds, next_command, robot_spec


class CpuPickPlace(CpuArm):
    def __init__(self, calibration):
        super().__init__(calibration, check_task=False,
                         spec_factory=partial(robot_spec, with_brick=True))
        self.home = np.array(CONFIG.start_q, np.float32)
        self.active_count = 6
        self.bounds = action_bounds(calibration)
        self.center = (self.bounds[0]+self.bounds[1])/2
        self.scale = (self.bounds[1]-self.bounds[0])/2
        self.episode_steps = round(CONFIG.episode_seconds/CONTROL_DT)
        self.state = TaskState(1)
        self.brick_id = self.model.body("brick").id
        self.brick_geom = self.model.geom("brick_geom").id
        self.finger_ids = [{i for i in range(self.model.ngeom) if self.model.geom(i).name.startswith(name+"_")}
                           for name in ("fixed_finger", "moving_finger")]
        self.floor_id = self.model.geom("floor").id

    def inputs(self):
        contacts = np.zeros(4, bool)
        for contact in self.data.contact:
            if self.brick_geom not in contact.geom or contact.dist >= 0:
                continue
            other = contact.geom[1] if contact.geom[0] == self.brick_geom else contact.geom[0]
            contacts[0] |= other in self.finger_ids[0]
            contacts[1] |= other in self.finger_ids[1]
            contacts[2] |= self.model.geom_bodyid[other] not in (0, self.brick_id)
            contacts[3] |= other == self.floor_id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.brick_id, velocity, 0)
        values = (self.q, self.dq, self.data.xpos[self.brick_id], self.data.xquat[self.brick_id],
                  np.r_[velocity[3:], velocity[:3]], self.tool, self.goal)
        return (*[torch.tensor(v, dtype=torch.float32)[None] for v in values],
                torch.tensor(contacts)[None])

    def observe(self):
        q, dq, pos, quat, vel, tool, target, contacts = self.inputs()
        return self.state.observation(q, dq, torch.tensor(self.command)[None],
            *[torch.tensor(v) for v in self.bounds], pos, quat, vel, tool, target, contacts)[0].numpy().astype(np.float32)

    def reset(self, goal=None, q=None, **kwargs):
        self.state.reset()
        super().reset(np.array(CONFIG.target_position), q=self.home, **kwargs)
        return self.observe()

    def step(self, action):
        previous = self.command.copy()
        self.command = next_command(action, previous, *self.bounds)
        target = self.calibration.quantize(self.command)
        for _ in range(DECIMATION):
            self.step_targets(target)
        # Match mjlab reward/termination's one-physics-substep kinematic lag.
        self.state.advance(*self.inputs(), torch.tensor(self.command-previous, dtype=torch.float32)[None])
        mujoco.mj_forward(self.model, self.data)
        return self.observe(), float(self.state.reward[0])


def run_episode(robot, policy, goal=None, q=None, on_step=None):
    obs = robot.reset()
    total = 0.
    distances = []
    for step in range(robot.episode_steps):
        start = time.monotonic()
        obs, reward = robot.step(policy(obs))
        total += reward
        distance = float(np.linalg.norm(robot.data.xpos[robot.brick_id, :2]-robot.goal[:2]))
        distances.append(distance)
        touched, picked, placed = robot.state.milestones[0].tolist()
        failed = bool(robot.state.failed[0])
        state = dict(step=step+1, time_s=(step+1)*CONTROL_DT, episode_steps=robot.episode_steps,
            wall_step_start=start, distance_m=distance, success=placed, failed=failed,
            touched=touched, picked=picked, placed=placed)
        if on_step is not None and on_step(robot, state) is False:
            return None
        if placed or failed:
            break
    return dict(success=int(placed), failed=int(failed), touched=int(touched), picked=int(picked),
        placed=int(placed), steps=step+1, **{"return": total},
        final_distance_m=distances[-1], mean_distance_m=float(np.mean(distances)),
        max_hold_s=float(robot.state.holds[0, 1])*CONTROL_DT)
