"""Hardware worker, pose calibration, and offline identification for the SO-101 UI.

No Qt dependency. Only the worker process owns the serial port. Public motor
commands are position targets; calculated torques are NEVER written to motors.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import queue
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np
import so101_sysid as s

RATE = 50
P = 16
SPEED_DEG_S = 8.0
MOVE_DEG = 10.0
ANCHOR_DEG = 30.0
FOLLOW_ERROR_DEG = 8.0
MARGIN_DEG = 3.0
HEARTBEAT_TIMEOUT = 1.5
MAX_SAMPLE_GAP = 0.25
MAX_ARMED_SECONDS = 300
AMPLITUDES_DEG = np.array([3., 3., 3., 3., 3., 2.])
# Separate frequencies/phases excite all axes without identical correlated input.
PRESETS = [
    ("train_a", "train", 12., [.23, .29, .37, .41, .47, .53], [0, .7, 1.3, 2., 2.7, 3.4]),
    ("train_b", "train", 12., [.31, .43, .27, .51, .39, .33], [1., 2., .5, 2.5, 1.5, 3.]),
    ("holdout", "validation", 14., [.35, .25, .45, .33, .55, .41], [.4, 1.8, 2.8, .9, 2.2, 3.5]),
]


def stamp():
    return time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def vector(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("Expected six finite joint values.")
    return value


def raw_bounds(cfg):
    margin = math.ceil(math.radians(MARGIN_DEG) / s.RAD_PER_TICK)
    lower = np.array([cfg["calibration"][j]["range_min"] + margin for j in s.JOINTS])
    upper = np.array([cfg["calibration"][j]["range_max"] - margin for j in s.JOINTS])
    return lower, upper


def check_raw(raw, cfg):
    raw = vector(raw)
    lower, upper = raw_bounds(cfg)
    bad = np.flatnonzero((raw < lower) | (raw > upper))
    if len(bad):
        raise ValueError("Inside the 3-degree travel margin: " + ", ".join(s.JOINTS[k] for k in bad)
                         + ". With torque OFF, support and reposition away from the stops.")


def multisine(t, preset):
    _, _, duration, frequencies, phases = preset
    if t <= 0 or t >= duration:
        return np.zeros(6)
    edge = min(1., t / 2., (duration - t) / 2.)
    envelope = .5 - .5 * math.cos(math.pi * edge)
    return np.deg2rad(AMPLITUDES_DEG) * envelope * np.sin(2 * np.pi * np.array(frequencies) * t + phases)


def slew(current, desired, dt):
    # Cap elapsed time so a scheduling stall cannot cause a large catch-up step.
    change = math.radians(SPEED_DEG_S) * min(max(dt, 0.), 1.5 / RATE) / s.RAD_PER_TICK
    return np.clip(vector(desired), vector(current) - change, vector(current) + change)


class Mechanics:
    """CAD gravity only. qfrc_bias at zero velocity excludes friction/contact."""
    def __init__(self, xml):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.qadr = [int(self.model.joint(j).qposadr[0]) for j in s.JOINTS]
        self.vadr = [int(self.model.joint(j).dofadr[0]) for j in s.JOINTS]
        if self.model.nq != 6 or self.model.nv != 6:
            raise ValueError("This interface requires the fixed-base six-hinge SO-101 model.")
        self.limits = np.array([self.model.joint(j).range for j in s.JOINTS])
        self.limited = np.array([bool(self.model.joint(j).limited) for j in s.JOINTS])

    def check_q(self, q, margin=True):
        q = vector(q)
        m = math.radians(MARGIN_DEG) if margin else 0.
        bad = np.flatnonzero(self.limited & ((q < self.limits[:, 0] + m) | (q > self.limits[:, 1] - m)))
        if len(bad):
            raise ValueError("Outside model joint range/margin: " + ", ".join(s.JOINTS[k] for k in bad)
                             + ". Check the mapping and starting pose.")

    def gravity(self, q):
        self.data.qpos[self.qadr] = vector(q)
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        self.data.ctrl[:] = 0
        self.mj.mj_forward(self.model, self.data)
        return self.data.qfrc_bias[self.vadr].copy()


def snapshot_model(cfg, xml, root):
    """Freeze the backend's MJCF + assets so later model edits cannot alter logs."""
    import xml.etree.ElementTree as ET
    xml, root = Path(xml), Path(root)
    tree = ET.parse(xml)
    if tree.findall(".//include"):
        raise ValueError("Flatten MJCF includes before using this interface.")
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    compiler = tree.getroot().find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    # Preserve arbitrary local mesh locations by rewriting every path in snapshot.
    assets = model_dir / "assets"
    assets.mkdir()
    for index, node in enumerate(tree.findall(".//asset/mesh")):
        name = node.get("file")
        if name:
            source = (xml.parent / meshdir / name).resolve()
            dest = f"{index:03d}_{source.name}"
            shutil.copyfile(source, assets / dest)
            # Preserve implicit mesh name when replacing its file name.
            if node.get("name") is None:
                node.set("name", Path(name).stem)
            node.set("file", dest)
    if tree.findall(".//asset/texture[@file]") or tree.findall(".//asset/hfield[@file]"):
        raise ValueError("Use the stock SO-101 MJCF with local meshes; external textures/heightfields need packaging first.")
    if compiler is None:
        compiler = ET.SubElement(tree.getroot(), "compiler")
    compiler.set("meshdir", "assets")
    tree.write(model_dir / "model.xml")
    frozen = copy.deepcopy(cfg)
    frozen["xml"] = "model/model.xml"
    atomic_json(root / "config.json", frozen)
    return frozen


def model_digest(xml):
    import xml.etree.ElementTree as ET
    xml = Path(xml)
    root = ET.parse(xml).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    for node in root.findall(".//asset/mesh"):
        name = node.get("file")
        if name:
            if node.get("name") is None: node.set("name", Path(name).stem)
            digest = hashlib.sha256((xml.parent / meshdir / name).read_bytes()).hexdigest()
            node.set("file", digest)
    if compiler is not None: compiler.set("meshdir", "_content_addressed_meshes")
    canonical = ET.canonicalize(ET.tostring(root, encoding="unicode"), strip_text=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def fit_offsets(cfg, pairs):
    if len(pairs) < 3:
        raise ValueError("Capture at least three distinct matched poses before fitting offsets.")
    raws = np.array([p["raw_position"] for p in pairs])
    references = np.array([p["reference_q_rad"] for p in pairs])
    if not np.all(np.isfinite(raws)) or not np.all(np.isfinite(references)):
        raise ValueError("Non-finite pose pair.")
    candidate = copy.deepcopy(cfg)
    report = {}
    for k, joint in enumerate(s.JOINTS):
        cal = cfg["calibration"][joint]
        x = (raws[:, k] - (cal["range_min"] + cal["range_max"]) / 2) * s.RAD_PER_TICK
        y = references[:, k]
        sign = cfg["mapping"][joint]["sign"]
        offset = float(np.median(y - sign * x))
        residual = y - (sign * x + offset)
        opposite = y - (-sign * x + np.median(y + sign * x))
        span = float(np.rad2deg(np.ptp(x)))
        rmse = float(np.rad2deg(np.sqrt(np.mean(residual**2))))
        opposite_rmse = float(np.rad2deg(np.sqrt(np.mean(opposite**2))))
        warnings = []
        if span < 15:
            warnings.append("less than 15 degrees of coverage; direction is weakly checked")
        if span >= 15 and opposite_rmse + .5 < rmse:
            warnings.append("opposite sign fits better; check direction with torque-OFF align before using")
            candidate["mapping_reviewed"] = False
        if rmse > 1:
            warnings.append("pose residual over 1 degree; check measurement, backlash, mapping or geometry")
        candidate["mapping"][joint]["offset_rad"] = offset
        report[joint] = {"sign_preserved": sign, "offset_deg": math.degrees(offset),
                         "reference_residual_rmse_deg": rmse, "sample_span_deg": span,
                         "opposite_sign_rmse_deg": opposite_rmse, "warnings": warnings}
    # Deliberately do not mark an unreviewed direction mapping as reviewed.
    return candidate, report


class DemoBus:
    """Deterministic first-order toy plant. NEVER evidence about a real actuator."""
    def __init__(self, cfg, mechanics, clock=time.perf_counter):
        self.clock, self.cfg = clock, cfg
        q = np.clip(np.zeros(6), mechanics.limits[:, 0] + .2, mechanics.limits[:, 1] - .2)
        q[5] = .6
        self.raw = s.sim_to_raw(q, cfg).astype(float)
        self.target = self.raw.copy()
        self.last = clock()
        self.settings = {}
        for j in s.JOINTS:
            c = cfg["calibration"][j]
            self.settings[j] = {key: 0 for key in s.REG}
            self.settings[j].update(model=777, p=16, d=32, lock=1, phase=12,
                                    min_position=c["range_min"], max_position=c["range_max"],
                                    homing_offset=c["homing_offset"], max_torque=1000, torque_limit=1000)

    def verify(self, cfg):
        if cfg["calibration"] != self.cfg["calibration"]:
            raise ValueError("Demo calibration mismatch")

    def snapshot(self): return copy.deepcopy(self.settings)
    def read(self, id_, key): return self.settings[s.JOINTS[id_ - 1]][key]
    def write(self, id_, key, value): self.settings[s.JOINTS[id_ - 1]][key] = value
    def goals(self, raw): self.target = vector(raw).copy()
    def read_temperatures(self): return [30] * 6
    def disable(self):
        for j in s.JOINTS: self.settings[j]["torque_enable"] = 0
    def restore(self, settings):
        self.disable()
        self.settings = copy.deepcopy(settings)
        self.disable()
    def close(self): pass

    def feedback(self):
        now = self.clock()
        dt = max(0., now - self.last)
        self.last = now
        for k, j in enumerate(s.JOINTS):
            if self.settings[j]["torque_enable"]:
                self.raw[k] += (self.target[k] - self.raw[k]) * (1 - math.exp(-dt / .10))
        return {"t": now, "read_start": now, "read_end": now,
                "raw_position": np.rint(self.raw).astype(int).tolist(), "temperature_c": [30] * 6,
                "voltage_v": [12.] * 6, "status": [0] * 6,
                "velocity_raw": [0] * 6, "current_raw": [0] * 6, "load_raw": [0] * 6}


class Engine:
    def __init__(self, bus, cfg, mechanics, root, synthetic=False, clock=time.perf_counter):
        self.bus, self.cfg, self.mechanics = bus, copy.deepcopy(cfg), mechanics
        self.root, self.synthetic, self.clock = Path(root), synthetic, clock
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = (self.root / "telemetry.jsonl").open("a", buffering=1)
        self.armed = False
        self.owns_settings = False
        self.faulted = False
        self.fit_locked = False
        self.message = "Connected read-only. Torque is OFF."
        self.original = None
        self.effective = None
        self.row = None
        self.last_tick = clock()
        self.last_heartbeat = clock()
        self.ring = deque(maxlen=200)
        self.target = self.desired = self.anchor = None
        self.armed_at = 0.
        self.mode = "OFF"
        self.pairs = []
        self.frozen = None
        self.plan = []
        self.trial = None
        self.run_dir = None
        self.run_manifest = None
        self.fault_path = None
        self.temperature_warning_at = float("-inf")
        self.temperature_peaks = [0] * 6
        bus.verify(cfg)
        settings = bus.snapshot()
        if any(v["torque_enable"] for v in settings.values()):
            self.journal.close()
            raise RuntimeError("Motors already enabled. Support the arm and use torque-OFF align first; close other serial clients.")

    def note(self, kind, **fields):
        self.journal.write(json.dumps({"kind": kind, "host_time": self.clock(), **fields}, allow_nan=False) + "\n")

    def observe_temperature(self, row):
        """Keep even one-sample spikes visible across the slower GUI refresh."""
        if not s.temperature_warning(row):
            return
        now = self.clock()
        if now - self.temperature_warning_at > 5:
            self.temperature_peaks = [0] * 6
        peaks = [max(old, value) if value >= s.TEMPERATURE_WARNING_C else old
                 for old, value in zip(self.temperature_peaks, row["temperature_c"])]
        if peaks != self.temperature_peaks:
            self.note("temperature_warning", temperature_c=row["temperature_c"],
                      recent_peak_c=peaks, threshold_c=s.TEMPERATURE_WARNING_C,
                      action="warning_only_no_temperature_cutoff")
        self.temperature_peaks = peaks
        self.temperature_warning_at = now

    def temperature_banner(self):
        if self.clock() - self.temperature_warning_at > 5:
            return None
        warning = s.temperature_warning({"temperature_c": self.temperature_peaks})
        return warning + " (recent peaks). Temperature alone will NOT stop motion."

    def stable_raw(self):
        if self.row is None:
            raise ValueError("Wait for feedback.")
        now = self.row["t"]
        rows = [r for r in self.ring if now - r["t"] <= .7]
        if len(rows) < 10 or rows[-1]["t"] - rows[0]["t"] < .45:
            raise ValueError("Wait for at least half a second of stable feedback.")
        values = np.array([r["raw_position"] for r in rows])
        if np.max(np.ptp(values, axis=0)) * s.RAD_PER_TICK > math.radians(.4):
            raise ValueError("Arm is moving. Wait until it settles before capturing or starting a sweep.")
        return np.rint(np.median(values, axis=0)).astype(int)

    def check_position(self, raw):
        check_raw(raw, self.cfg)
        self.mechanics.check_q(s.raw_to_sim(raw, self.cfg))

    def arm(self):
        if self.armed or self.faulted or self.fit_locked:
            raise ValueError("Already armed, or fault latched. Restart after inspecting a fault.")
        if not self.cfg.get("mapping_reviewed"):
            raise ValueError("Review and save all joint directions with torque-OFF align first.")
        self.bus.verify(self.cfg)
        original = self.bus.snapshot()
        if any(v["torque_enable"] for v in original.values()):
            raise ValueError("Expected all motors OFF.")
        row = self.bus.feedback()
        self.note("pre_arm", feedback=row)
        self.observe_temperature(row)
        s.telemetry_ok(row, warn_temperature=False)
        raw = vector(row["raw_position"])
        self.check_position(raw)
        self.original = original
        atomic_json(self.root / f"settings-{stamp()}.json", {"settings": original})
        self.owns_settings = True  # Partial setup failures must also restore/disable.
        try:
            for id_ in s.IDS:
                self.bus.write(id_, "lock", 0)
                for key, value in {"p": P, "i": 0, "d": 0, "acceleration": 254,
                                   "max_acceleration": 254, "goal_velocity": 0}.items():
                    self.bus.write(id_, key, value)
            # Last supported pose, read again after setup, then seed EVERY goal.
            row = self.bus.feedback()
            self.note("before_enable", feedback=row)
            self.observe_temperature(row)
            s.telemetry_ok(row, warn_temperature=False)
            raw = vector(row["raw_position"])
            self.check_position(raw)
            self.bus.goals(np.rint(raw).astype(int))
            for id_ in s.IDS:
                self.bus.write(id_, "torque_enable", 1)
                self.bus.write(id_, "lock", 1)
            self.effective = self.bus.snapshot()
            if any(v["p"] != P or v["i"] or v["d"] or v["torque_enable"] != 1
                   for v in self.effective.values()):
                raise RuntimeError("Gain/enable readback differs from requested P=16, I=D=0.")
            self.target = self.desired = raw.copy()
            self.anchor = raw.copy()
            self.armed = True
            self.armed_at = self.last_tick = self.clock()
            self.mode = "HOLD"
            self.message = "Holding. Remove support and keep clear. Five-minute arm timeout is active."
            self.note("armed", effective_settings=self.effective)
        except Exception:
            # Raise to the worker's fault handler, which preserves the last read.
            self.row = row
            raise

    def move(self, q):
        if not self.armed or self.plan:
            raise ValueError("Enable hold first and finish/cancel the current sweep.")
        self.stable_raw()
        raw = s.sim_to_raw(vector(q), self.cfg)
        self.check_position(raw)
        if np.max(abs(raw - self.target)) * s.RAD_PER_TICK > math.radians(MOVE_DEG):
            raise ValueError("Preview is more than 10 degrees from the current sent goal. Use smaller moves.")
        if np.max(abs(raw - self.anchor)) * s.RAD_PER_TICK > math.radians(ANCHOR_DEG):
            raise ValueError("More than 30 degrees from this hold's starting pose. Support, disable, and choose a new start.")
        self.desired = raw.astype(float)
        self.frozen = None
        self.mode = "MOVE"
        self.message = "Moving all changed joints together at up to 8 degrees/second."

    def freeze(self):
        if self.plan or self.mode == "MOVE":
            raise ValueError("Finish the motion first.")
        raw = self.stable_raw()
        self.frozen = {"raw_position": raw.tolist(), "host_time": self.clock(),
                       "mapped_q_rad": s.raw_to_sim(raw, self.cfg).tolist()}
        self.message = "Measurement frozen. Adjust the reference to the physical links, then save the pair."

    def capture(self, reference, method):
        if self.frozen is None:
            raise ValueError("Freeze a stable measurement first.")
        if self.plan or self.mode == "MOVE":
            raise ValueError("Cannot capture during motion.")
        current = self.stable_raw()
        if np.max(abs(current - np.array(self.frozen["raw_position"]))) * s.RAD_PER_TICK > math.radians(.5):
            raise ValueError("Arm moved since freezing. Freeze again and rematch the pose.")
        q = vector(reference)
        self.mechanics.check_q(q, margin=False)
        pair = {**copy.deepcopy(self.frozen), "reference_q_rad": q.tolist(),
                "reference_method": str(method), "synthetic": self.synthetic,
                "gravity_reference_nm": self.mechanics.gravity(q).tolist(),
                "gravity_encoder_nm": self.mechanics.gravity(self.frozen["mapped_q_rad"]).tolist(),
                "torque_kind": "CAD gravity at zero velocity; not measured motor torque"}
        self.pairs.append(pair)
        atomic_json(self.root / "pose_pairs.json", {"schema": 1, "config": self.cfg, "poses": self.pairs})
        self.frozen = None
        self.message = f"Saved matched pose {len(self.pairs)} and model-based gravity torques."

    def candidate_mapping(self):
        if self.armed:
            raise ValueError("Support and disable torque before fitting/saving a new coordinate mapping.")
        candidate, report = fit_offsets(self.cfg, self.pairs)
        if self.synthetic:
            candidate["mapping_reviewed"] = False
            candidate["synthetic"] = True
        atomic_json(self.root / "mapping_candidate.json", candidate)
        atomic_json(self.root / "mapping_report.json", report)
        self.message = "Saved mapping_candidate.json and mapping_report.json. Review them, then restart using the candidate config."

    def start_suite(self):
        if not self.armed or self.plan or self.mode == "MOVE":
            raise ValueError("Start from a settled enabled hold with no active motion.")
        self.stable_raw()
        if self.clock() - self.armed_at > MAX_ARMED_SECONDS - 65:
            raise ValueError("Not enough time left in this hold. Support, disable and enable again before recording.")
        center = np.rint(self.target).astype(int)
        # Both raw and model limits; check both signs separately after mapping.
        delta = np.ceil(np.deg2rad(AMPLITUDES_DEG) / s.RAD_PER_TICK).astype(int)
        for direction in (-1, 1): self.check_position(center + direction * delta)
        if np.max(abs(center - self.anchor) + delta) * s.RAD_PER_TICK > math.radians(ANCHOR_DEG):
            raise ValueError("Sweep would exceed the 30-degree envelope around the enabled pose.")
        self.run_dir = self.root / "runs" / stamp()
        self.run_dir.mkdir(parents=True)
        self.run_manifest = {"schema": 1, "complete": False, "synthetic": self.synthetic,
                             "center_raw": center.tolist(), "logs": []}
        atomic_json(self.run_dir / "manifest.json", self.run_manifest)
        self.plan = list(PRESETS)
        self.center = center
        self.frozen = None
        self._begin_trial(self.clock())

    def _begin_trial(self, now):
        self.preset = self.plan[0]
        self.trial_origin = now
        recent = [r for r in self.ring if self.row["t"] - r["t"] <= .7]
        times = np.array([r["t"] for r in recent])
        centered = times - times.mean()
        positions = np.array([s.raw_to_sim(r["raw_position"], self.cfg) for r in recent])
        # Stable pre-motion window; linear slope is less noisy than differencing
        # adjacent quantized encoder readings. This is an estimate, not a tachometer.
        velocity = centered @ positions / (centered @ centered) if centered @ centered > 0 else np.zeros(6)
        self.trial = {"schema": 1, "complete": False, "synthetic": self.synthetic,
                      "config": copy.deepcopy(self.cfg), "joint": "all", "role": self.preset[1],
                      "trajectory": "coordinated_multisine", "preset": self.preset,
                      "command_rate_hz": RATE, "p": P, "i": 0, "d": 0,
                      "effective_settings": copy.deepcopy(self.effective),
                      "initial_target_raw": np.rint(self.target).astype(int).tolist(),
                      "initial_voltage_v": self.row["voltage_v"],
                      "initial_velocity_rad_s": velocity.tolist(),
                      "initial_velocity_method": "linear slope over preceding stable 0.7-second window",
                      "samples": [], "commands": []}
        self.mode = "RECORD"
        self.message = f"Recording {self.preset[0]}; all six motors move together."

    def finish_trial(self, complete, error=None):
        if self.trial is None:
            return
        self.trial["complete"] = bool(complete)
        if error: self.trial["error"] = error
        filename = self.preset[0] + ".json"
        atomic_json(self.run_dir / filename, self.trial)
        self.run_manifest["logs"].append({"file": filename, "role": self.preset[1], "complete": bool(complete)})
        atomic_json(self.run_dir / "manifest.json", self.run_manifest)
        self.trial = None

    def cancel_sweep(self):
        if self.plan:
            self.finish_trial(False, "User cancelled sweep")
            self.plan = []
        if self.armed:
            # Cancel keeps the last commanded pose; it does not jump to center.
            self.desired = self.target.copy()
            self.mode = "HOLD"
            self.message = "Motion cancelled; holding the last sent goal. Support before disabling."

    def disable(self, reason="User supported the arm and disabled torque"):
        self.cancel_sweep()
        self.bus.disable()  # Explicit operator action, even if previously read-only.
        if self.owns_settings and self.original is not None:
            self.bus.restore(self.original)
            self.owns_settings = False
        if any(self.bus.read(id_, "torque_enable") for id_ in s.IDS):
            raise RuntimeError("Torque-off readback failed; cut motor power")
        self.armed = False
        self.mode = "FAULT" if self.faulted else "OFF"
        self.message = reason + ". Torque OFF; original gains/profile restored."
        self.note("disabled", reason=reason)

    def fault(self, exc):
        self.faulted = True
        self.message = f"FAULT: {exc}"
        # Preserve the actual offending sample BEFORE any further bus reads.
        diagnostic = {"error": str(exc), "synthetic": self.synthetic, "fault_host_time": self.clock(),
                      "fault_feedback": copy.deepcopy(self.row), "recent_feedback": list(self.ring),
                      "settings_restored": False, "torque_off_confirmed": False}
        try: self.finish_trial(False, str(exc))
        except Exception as write_exc: diagnostic["log_save_error"] = str(write_exc)
        self.plan = []
        try:
            self.bus.disable()
            diagnostic["torque_off_confirmed"] = not any(self.bus.read(id_, "torque_enable") for id_ in s.IDS)
        except Exception as off_exc:
            diagnostic["torque_off_error"] = str(off_exc)
        if self.owns_settings and self.original is not None:
            try:
                self.bus.restore(self.original)
                self.owns_settings = False
                diagnostic["settings_restored"] = True
            except Exception as restore_exc:
                diagnostic["restore_error"] = str(restore_exc)
        try:
            diagnostic["independent_temperature_after_stop_c"] = self.bus.read_temperatures()
            diagnostic["temperature_after_stop_host_time"] = self.clock()
        except Exception as temp_exc:
            diagnostic["independent_temperature_error"] = str(temp_exc)
        self.armed = False
        self.mode = "FAULT"
        self.fault_path = str(self.root / f"fault-{stamp()}.json")
        try: atomic_json(self.fault_path, diagnostic)
        except Exception as write_exc: self.message += f"; could not save fault: {write_exc}"
        self.message += ". Support arm; cut motor power if torque-off failed. Restart only after diagnosis."

    def tick(self):
        now = self.clock()
        dt = now - self.last_tick
        self.last_tick = now
        if self.armed:
            if now - self.last_heartbeat > HEARTBEAT_TIMEOUT:
                raise RuntimeError("Control panel heartbeat stopped for 1.5 s")
            if dt > MAX_SAMPLE_GAP:
                raise RuntimeError(f"Hardware loop stalled for {dt:.3f} s")
            if now - self.armed_at > MAX_ARMED_SECONDS:
                raise RuntimeError("Five-minute enabled timeout")
        row = self.bus.feedback()
        self.row = row
        self.ring.append(copy.deepcopy(row))
        self.note("feedback", mode=self.mode, feedback=row)
        self.observe_temperature(row)
        if not self.armed:
            return
        if row["read_end"] - row["read_start"] > MAX_SAMPLE_GAP:
            raise RuntimeError("Servo read exceeded 250 ms")
        # Check after preserving telemetry, including settling/pre-motion rows.
        s.telemetry_ok(row, warn_temperature=False)
        actual = vector(row["raw_position"])
        self.check_position(actual)
        if np.max(abs(actual - self.anchor)) * s.RAD_PER_TICK > math.radians(ANCHOR_DEG + FOLLOW_ERROR_DEG):
            raise RuntimeError("Measured arm left the enabled-pose envelope")
        if np.max(abs(actual - self.target)) * s.RAD_PER_TICK > math.radians(FOLLOW_ERROR_DEG):
            raise RuntimeError("Measured joint differs from last sent target by more than 8 degrees")
        if self.plan:
            elapsed = row["t"] - self.trial_origin
            sample = copy.deepcopy(row)
            sample["q_rad"] = s.raw_to_sim(actual, self.cfg).tolist()
            for key in ("t", "read_start", "read_end"): sample[key] -= self.trial_origin
            self.trial["samples"].append(sample)
            signs = np.array([self.cfg["mapping"][j]["sign"] for j in s.JOINTS])
            self.desired = self.center + multisine(elapsed, self.preset) / signs / s.RAD_PER_TICK
            if elapsed >= self.preset[2] + 3:
                # Three seconds at the original goal at end of each trajectory.
                self.stable_raw()
                self.finish_trial(True)
                self.plan.pop(0)
                if self.plan:
                    self._begin_trial(self.clock())
                else:
                    self.run_manifest["complete"] = True
                    atomic_json(self.run_dir / "manifest.json", self.run_manifest)
                    self.mode = "HOLD"
                    self.message = "Sweep finished. Repeat from another pose, or support and disable before fitting."
                    self.desired = self.center.astype(float)
        next_target = slew(self.target, self.desired, dt)
        raw = np.rint(next_target).astype(int)
        self.check_position(raw)
        before = self.clock()
        self.bus.goals(raw)  # One synchronized position packet for all six motors.
        after = self.clock()
        self.note("command", tx_start=before, tx_end=after, raw_target=raw.tolist())
        self.target = next_target
        if self.trial is not None:
            self.trial["commands"].append({"tx_start": before - self.trial_origin,
                    "t": after - self.trial_origin, "raw_target": raw.tolist(),
                    "q_target_rad": s.raw_to_sim(raw, self.cfg).tolist()})
        if self.mode == "MOVE" and np.max(abs(self.desired - self.target)) < .51:
            self.mode = "HOLD"
            self.message = "Goal reached by command. Wait for measured motion to settle before capturing."

    def state(self):
        return {"mode": self.mode, "armed": self.armed, "faulted": self.faulted,
                "temperature_warning": self.temperature_banner(),
                "message": self.message, "feedback": self.row, "frozen": self.frozen,
                "pair_count": len(self.pairs), "session": str(self.root), "fault_file": self.fault_path,
                "remaining_hold_s": max(0., MAX_ARMED_SECONDS - (self.clock() - self.armed_at)) if self.armed else 0.,
                "target_q_rad": s.raw_to_sim(np.rint(self.target), self.cfg).tolist() if self.target is not None else None}

    def close(self):
        try:
            if self.armed or self.owns_settings: self.disable("Worker closing")
        finally:
            self.journal.close()
            self.bus.close()


def publish(outbox, state):
    try: outbox.put_nowait(state)
    except queue.Full:
        try: outbox.get_nowait()
        except queue.Empty: pass
        try: outbox.put_nowait(state)
        except queue.Full: pass


def worker(config_path, port, session, synthetic, inbox, outbox, stop_event):
    """Serial ownership in a separate process; GUI stalls cannot block telemetry."""
    engine = None
    bus = None
    try:
        cfg, xml = s.load_config(config_path)
        if cfg.get("synthetic") and not synthetic:
            raise ValueError("Refusing hardware with a synthetic/demo mapping.")
        mechanics = Mechanics(xml)
        bus = DemoBus(cfg, mechanics) if synthetic else s.Bus(port)
        if not synthetic:
            # pyserial otherwise defaults to an unbounded blocking write. Reads
            # are already bounded by the SDK packet timeouts.
            bus.port.ser.write_timeout = .1
            bus.port.ser.exclusive = True
        engine = Engine(bus, cfg, mechanics, session, synthetic)
        last_publish = 0.
        while not stop_event.is_set():
            start = time.perf_counter()
            # Separate Event takes priority over queued commands, including a
            # queue backlog. It is also used on viewer closure and Ctrl+C.
            for _ in range(20):
                try: command = inbox.get_nowait()
                except queue.Empty: break
                action = command["action"]
                if action == "heartbeat":
                    # Trust send time, not arrival time: backlog cannot fake life.
                    engine.last_heartbeat = command["sent"]
                    continue
                if engine.faulted:
                    continue
                try:
                    if action == "arm": engine.arm()
                    elif action == "off": engine.disable()
                    elif action == "move": engine.move(command["q"])
                    elif action == "cancel": engine.cancel_sweep()
                    elif action == "freeze": engine.freeze()
                    elif action == "capture": engine.capture(command["q"], command["method"])
                    elif action == "offsets": engine.candidate_mapping()
                    elif action == "suite": engine.start_suite()
                    elif action == "lock_fit":
                        if engine.armed or engine.owns_settings:
                            raise ValueError("Support and disable torque before fitting.")
                        engine.fit_locked = True
                        engine.mode = "OFF_FIT"
                        engine.message = "Torque OFF. Motor enable locked during offline fitting."
                    elif action == "unlock_fit":
                        engine.fit_locked = False
                        engine.mode = "OFF"
                        engine.message = "Offline fitting ended. Torque remains OFF."
                    else: raise ValueError("Unknown interface command")
                except ValueError as exc:
                    engine.message = str(exc)  # Rejected request: no motion change.
                    if engine.owns_settings and not engine.armed:
                        engine.fault(exc)  # Also catch partial arm preparation.
                except Exception as exc:
                    engine.fault(exc)
            if not engine.faulted:
                try: engine.tick()
                except Exception as exc: engine.fault(exc)
            if start - last_publish >= .1:
                publish(outbox, engine.state())
                last_publish = start
            stop_event.wait(max(0., 1 / RATE - (time.perf_counter() - start)))
    except BaseException as exc:
        publish(outbox, {"mode": "FAULT", "armed": False, "faulted": True, "fatal": True, "message": str(exc)})
    finally:
        try:
            if engine is not None:
                engine.close()
                publish(outbox, {**engine.state(), "closed": True})
            elif bus is not None: bus.close()
        except Exception as exc:
            publish(outbox, {"mode": "FAULT", "armed": False, "faulted": True, "closed": True,
                             "message": f"Shutdown failed: {exc}. Support arm and cut motor power."})
