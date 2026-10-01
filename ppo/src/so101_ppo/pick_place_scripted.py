"""Physical feasibility baseline, never used to drive or supervise PPO.

Only servo actions are emitted. The brick is moved by frictional contact, with
no state writes, attachment constraints, or changes to the calibrated controller.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares

from .contract import CONTROL_DT
from .pick_place import CONFIG


class ScriptedPolicy:
    def __init__(self, robot):
        self.center, self.scale = robot.center, robot.scale
        model = robot.model
        data = mujoco.MjData(model)
        gripper = model.body('gripper').id
        # The tool site lies on the fixed fingertip, 8.5 mm left of the brick center.
        x, y, _ = CONFIG.brick_position
        tx, ty, _ = CONFIG.target_position
        phases = (
            ('approach', (x-.0085, y, .009), .5, 10.),
            ('close', (x-.0085, y, .009), -.03, 6.),
            ('lift', (x-.0085, y, .045), -.03, 8.),
            ('carry', (tx-.0085, ty, .045), -.03, 8.),
            ('lower', (tx-.0085, ty, .009), -.03, 8.),
            ('release', (tx-.0085, ty, .009), .5, 6.),
            ('retreat', (tx-.0085, ty, .06), .5, 8.),
        )
        self.targets, self.ends, self.phases = [], [], []
        end = 0
        for name, position, opening, seconds in phases:
            def residual(q):
                data.qpos[robot.qi[:5]] = q
                mujoco.mj_forward(model, data)
                return np.r_[(data.site('gripperframe').xpos-position)*10,
                    (data.xmat[gripper].reshape(3, 3)-np.eye(3)).ravel()]
            solution = least_squares(residual, [0., 0., .5, 1., .05],
                bounds=(robot.bounds[0][:5]+.001, robot.bounds[1][:5]-.001), max_nfev=200)
            if np.linalg.norm(solution.fun) > 1e-4:
                raise ValueError(f'Pick-place {name} is unreachable within this calibration')
            self.targets.append(np.r_[solution.x, opening])
            self.phases.append(name)
            end += round(seconds/CONTROL_DT)
            self.ends.append(end)
        self.step = 0

    def __call__(self, observation):
        index = min(int(np.searchsorted(self.ends, self.step, side='right')), len(self.targets)-1)
        self.step += 1
        return (self.targets[index]-self.center)/self.scale
