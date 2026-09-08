"""BAM adapter matching the calibration's caps, soft friction and state reset."""
from dataclasses import dataclass, replace

import torch
from bam.mjlab import BamActuator, BamActuatorCfg

from .calibration import delay_steps, register_12v


class CalibratedBamActuator(BamActuator):
    def __init__(self, cfg, *args, **kwargs):
        register_12v()
        super().__init__(cfg, *args, **kwargs)
        self._bam_model.actuator.max_pwm = cfg.max_pwm

    def edit_spec(self, spec, target_names):
        super().edit_spec(spec, target_names)
        for actuator in self._mjs_actuators:
            # The reference limits PWM inside BAM, not torque after back-EMF.
            actuator.ctrllimited = False
            actuator.forcelimited = False

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._lag = int(delay_steps(self.cfg.command_delay_s))
        self._history = torch.zeros(self._lag + 1, self._num_envs, len(self._target_ids_list), device=self._device)
        self._cursor = 0
        self._first = torch.ones(self._num_envs, 1, dtype=torch.bool, device=self._device)
        self._committed_smooth = None
        self._pending_smooth = None
        self._pending_target = None
        # EntityData is constructed after actuator initialization. The action
        # reset seeds this history once the measured joint state is available.

    def reset_target_history(self, env_ids, target=None):
        ids = slice(None) if env_ids is None else env_ids
        q = self.entity.data.joint_pos[:, self._target_ids_list] if target is None else target
        act = self._bam_model.actuator
        smooth = self._committed_smooth
        if smooth is None:
            smooth = q.clone()
        else:
            smooth[ids] = q[ids]
        self._committed_smooth = smooth
        act.set_state((smooth, []))
        self._history[:, ids] = q[ids]
        self._first[ids] = True

    def apply_delay(self, cmd):
        self._pending_target = cmd.position_target.clone()
        delayed = (cmd.position_target if self._lag == 0 else
                   self._history[(self._cursor - self._lag) % len(self._history)].clone())
        return replace(cmd, position_target=delayed)

    def compute(self, cmd):
        # Match MuJoCoController's dt=0 on each world's first update after reset.
        dt = self._dt
        self._dt = torch.where(self._first, 0., dt)
        act = self._bam_model.actuator
        act.set_state((self._committed_smooth.clone(), []))
        try:
            torque = super().compute(cmd)
            self._pending_smooth = act.q_target_smooth
        finally:
            self._dt = dt
            act.set_state((self._committed_smooth, []))
        return torque

    def update(self, dt):
        # mjlab also calls compute while forwarding a reset, without stepping.
        # Commit state only after integration, never on those extra calls.
        if self._pending_smooth is not None:
            self._committed_smooth.copy_(self._pending_smooth)
            self._history[self._cursor] = self._pending_target
            self._cursor = (self._cursor + 1) % len(self._history)
            self._first[:] = False


@dataclass(kw_only=True)
class CalibratedBamActuatorCfg(BamActuatorCfg):
    max_pwm: float = .97
    command_delay_s: float = 0.
    stiff_frictionloss: bool = False

    def build(self, entity, target_ids, target_names):
        return CalibratedBamActuator(self, entity, target_ids, target_names)
