"""Portable calibration bundle and the exact encoder convention of the interface.

No hardware dependencies or serial access. Robot zeros belong to the encoder
mapping, never to BAM's bench q_offset or to a second shift of the MJCF joints.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .contract import ACTION_SCALE, HOME, JOINTS, PHYSICS_DT

RAD_PER_TICK = 2 * math.pi / 4095
BAM_REVISION = "620a64fe67c1afe94fca81da73b128c7aed17c5f"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register_12v():
    """Same motor registration as the supplied so101_sysid.register_12v()."""
    from bam.actuators import actuators
    from bam.feetech.actuator import STS3215Actuator
    from bam.testbench import Pendulum

    def factory():
        actuator = STS3215Actuator(Pendulum)
        actuator.vin = 12.0
        return actuator

    actuators["sts321512v"] = factory


def delay_steps(seconds):
    """A target becomes available at the first physics tick after its delay."""
    return np.ceil(np.asarray(seconds) / PHYSICS_DT - 1e-10).astype(int)


class Calibration:
    def __init__(self, root, allow_synthetic=False):
        self.root = Path(root).resolve()
        self.document = read_json(self.root / "bundle.json")
        doc = self.document
        if doc.get("schema") != 1 or doc.get("joint_order") != list(JOINTS):
            raise ValueError("Unsupported calibration bundle schema or joint order")
        if doc.get("synthetic") and not allow_synthetic:
            raise ValueError("Synthetic calibration: use --allow-synthetic only for software tests")
        if doc.get("bam_revision") != BAM_REVISION:
            raise ValueError("Calibration bundle belongs to a different BAM implementation")
        for name, expected in doc["files_sha256"].items():
            if sha256(self.path(name)) != expected:
                raise ValueError(f"Calibration bundle checksum mismatch: {name}")
        self.config = read_json(self.path(doc["config_file"]))
        self.xml = self.path(doc["model_xml"])
        self.paths = tuple(str(self.path(p)) for p in doc["motor_files"])
        self.controller = doc["controller"]
        if (self.controller["p"], self.controller["i"], self.controller["d"]) != (16, 0, 0):
            raise ValueError("This task requires the calibration's P=16, I=D=0")
        if self.controller["control_hz"] != 50 or doc["physics_dt_s"] != PHYSICS_DT:
            raise ValueError("Calibration timing differs from the policy contract")
        self.voltage = np.asarray(self.controller["voltage_v"], dtype=float)
        self.max_pwm = np.asarray(self.controller["max_pwm"], dtype=float)
        self.delays = np.asarray(self.controller["command_delay_s"], dtype=float)
        for name, array in (("voltage", self.voltage), ("PWM", self.max_pwm), ("delay", self.delays)):
            if array.shape != (6,) or not np.isfinite(array).all():
                raise ValueError(f"Expected six finite {name} values")
        if np.any(self.voltage <= 0) or np.any((self.max_pwm <= 0) | (self.max_pwm > .97)):
            raise ValueError("Invalid motor voltage or PWM cap")
        if np.any((self.delays < 0) | (self.delays > .08)):
            raise ValueError("Delay outside the fitted 0..80 ms range")
        self.mid = np.array([(self.config["calibration"][j]["range_min"] +
                             self.config["calibration"][j]["range_max"]) / 2 for j in JOINTS])
        self.sign = np.array([self.config["mapping"][j]["sign"] for j in JOINTS])
        self.offset = np.array([self.config["mapping"][j]["offset_rad"] for j in JOINTS])
        if not np.isin(self.sign, [-1, 1]).all() or not np.isfinite(self.offset).all():
            raise ValueError("Invalid joint mapping")
        raw_min = np.array([self.config["calibration"][j]["range_min"] for j in JOINTS])
        raw_max = np.array([self.config["calibration"][j]["range_max"] for j in JOINTS])
        ends = np.stack((self.raw_to_sim(raw_min), self.raw_to_sim(raw_max)))
        self.lower = ends.min(axis=0) + np.deg2rad(3)
        self.upper = ends.max(axis=0) - np.deg2rad(3)
        self.lags = delay_steps(self.delays)

    def path(self, relative):
        result = (self.root / relative).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError("Bundle path escapes its directory")
        return result

    @property
    def digest(self):
        return sha256(self.root / "bundle.json")

    def raw_to_sim(self, raw):
        return (np.asarray(raw) - self.mid) * RAD_PER_TICK * self.sign + self.offset

    def sim_to_raw(self, q):
        # Feedback already includes the firmware homing offset. Never add it again.
        return np.rint((np.asarray(q) - self.offset) / self.sign / RAD_PER_TICK + self.mid).astype(int)

    def quantize(self, q):
        return self.raw_to_sim(self.sim_to_raw(q))

    def task_bounds(self):
        """Intersect calibrated encoder limits with the baseline action region."""
        from .model import target_bank
        goal_q = target_bank()[1][0]
        for label, q in (("baseline home", HOME), ("baseline goal", goal_q)):
            if np.any((q < self.lower) | (q > self.upper)):
                raise ValueError(f"{label} is outside the calibrated limits plus 3-degree margin")
        lower = np.maximum(self.lower[:3], HOME[:3] - ACTION_SCALE)
        upper = np.minimum(self.upper[:3], HOME[:3] + ACTION_SCALE)
        return lower.astype(np.float32), upper.astype(np.float32)

