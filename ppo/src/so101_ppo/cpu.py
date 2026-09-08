"""CPU MuJoCo rehearsal using the exported calibration and policy contract."""
import mujoco
import numpy as np
from bam.model import load_model
from bam.mujoco import MujocoController

from .calibration import register_12v
from .contract import CONTROL_DT, DECIMATION, HOME, JOINTS, SITE, next_command, observation, reward_rate
from .model import cpu_spec, indices


class CpuArm:
    def __init__(self, calibration, voltage=None, check_task=True):
        self.calibration = calibration
        self.bounds = calibration.task_bounds() if check_task else None
        register_12v()
        self.bam_models = [load_model(path) for path in calibration.paths]
        spec = cpu_spec(calibration)
        for name in JOINTS:
            actuator = spec.actuator(name)
            actuator.set_to_motor()
            actuator.ctrllimited = actuator.forcelimited = False
            actuator.gear = [1, 0, 0, 0, 0, 0]
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qi, self.vi = indices(self.model)
        self.controllers = []
        voltage = calibration.voltage if voltage is None else voltage
        for i, (name, model) in enumerate(zip(JOINTS, self.bam_models)):
            model.actuator.vin = float(voltage[i])
            model.actuator.kp = 16
            model.actuator.max_pwm = float(calibration.max_pwm[i])
            self.controllers.append(MujocoController(model, name, self.model, self.data))
        self.goal = np.zeros(3, np.float32)
        self.command = HOME.copy()
        self.history = np.zeros((int(calibration.lags.max()) + 1, 6))
        self.cursor = 0
        self.initial_friction = self.model.dof_frictionloss.copy()
        self.initial_damping = self.model.dof_damping.copy()

    @property
    def q(self):
        return self.data.qpos[self.qi].copy()

    @property
    def dq(self):
        return self.data.qvel[self.vi].copy()

    @property
    def tool(self):
        return self.data.site(SITE).xpos.copy()

    def observe(self):
        return observation(self.q, self.dq, self.goal, self.tool, self.command)

    def reset(self, goal, q=None, dq=None, initial_target=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qi] = HOME if q is None else q
        self.data.qvel[self.vi] = 0 if dq is None else dq
        self.goal = np.asarray(goal, np.float32).copy()
        self.command = HOME.copy()
        self.command[:3] = self.q[:3]
        target = self.calibration.quantize(self.command) if initial_target is None else np.asarray(initial_target)
        self.history[:] = target
        self.cursor = 0
        self.model.dof_frictionloss[:] = self.initial_friction
        self.model.dof_damping[:] = self.initial_damping
        for i, controller in enumerate(self.controllers):
            controller.reset()
            controller.set_q_target(JOINTS[i], target[i])
            controller.actuator_state = (np.array([target[i]]), [])
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def step_targets(self, targets, apply_delay=True):
        """One physics step. Replay supplies already-delayed recorded targets."""
        target = np.asarray(targets, float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Expected six finite joint targets")
        if apply_delay:
            self.history[self.cursor] = target
            target = self.history[(self.cursor - self.calibration.lags) % len(self.history), np.arange(6)]
            self.cursor = (self.cursor + 1) % len(self.history)
        for name, controller, value in zip(JOINTS, self.controllers, target):
            controller.set_q_target(name, float(value))
            controller.update()
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.q).all() or not np.isfinite(self.dq).all() or any(w.number for w in self.data.warning):
            raise FloatingPointError("MuJoCo numerical warning or non-finite state")
        return self.q

    def step(self, action):
        previous = self.command.copy()
        self.command = next_command(action, previous, self.bounds)
        target = self.calibration.quantize(self.command)
        for _ in range(DECIMATION):
            self.step_targets(target)
        distance = np.linalg.norm(self.goal - self.tool)
        reward = CONTROL_DT * reward_rate(distance, self.dq, (self.command - previous)[:3])
        mujoco.mj_forward(self.model, self.data)
        return self.observe(), float(reward)
