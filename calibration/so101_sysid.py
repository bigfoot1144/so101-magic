#!/usr/bin/env python3
"""SO-101 12 V: inspect, align, record, replay, fit. Python 3.12.

Hardware uses the Feetech SDK directly. LeRobot is used only for the initial
position calibration. Simulation uses upstream BAM, pinned in requirements.txt.
Read README.md before running a command that enables or disables torque.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IDS = list(range(1, 7))
COORDINATED_WEIGHTS = np.array([1.0, .7, -.7, .5, .5, .3])
RAD_PER_TICK = 2 * math.pi / 4095  # Match LeRobot's degree conversion.
SO_SHA = "eecbe3e0a9ebb23e25ad7b2759b03884c6660903"
BAM_SHA = "620a64fe67c1afe94fca81da73b128c7aed17c5f"
PR_SHA = "ce4176b525326e7ac08c6dfd29299dfd01653191"
SEED = {
    "kt": 1.443985897804671, "error_gain_ratio": 0.9754522802666795,
    "R": 2.0295624974250366, "armature": 0.03245999409626578,
    "q_offset": 0.0,  # Bench's 0.0945 rad offset is deliberately NOT a robot zero.
    "max_velocity": 14.597462818084868, "friction_base": 0.1958711784097532,
    "friction_viscous": 0.03047350651248057, "model": "m1",
    "actuator": "sts321512v",
}
REG = {
    "firmware_major": (0, 1), "firmware_minor": (1, 1), "model": (3, 2),
    "return_delay": (7, 1), "min_position": (9, 2), "max_position": (11, 2),
    "max_temperature": (13, 1), "max_torque": (16, 2), "phase": (18, 1),
    "p": (21, 1), "d": (22, 1), "i": (23, 1),
    "cw_dead_zone": (26, 1), "ccw_dead_zone": (27, 1),
    "protection_current": (28, 2), "homing_offset": (31, 2), "mode": (33, 1),
    "overload_torque": (36, 1), "torque_enable": (40, 1), "acceleration": (41, 1),
    "goal_velocity": (46, 2), "torque_limit": (48, 2), "lock": (55, 1),
    "max_acceleration": (85, 1),
}
CHANGED = ("p", "i", "d", "acceleration", "max_acceleration", "goal_velocity")


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; choose a new output name.")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def signed_magnitude(value, bit):
    return -(value & ((1 << bit) - 1)) if value & (1 << bit) else value


def validate_calibration(cal):
    if set(cal) != set(JOINTS):
        raise ValueError("Calibration must contain exactly the six SO-101 joint names.")
    for name, expected_id in zip(JOINTS, IDS):
        c = cal[name]
        if c["id"] != expected_id or c.get("drive_mode", 0) != 0:
            raise ValueError(f"{name}: requires standard ID {expected_id}, drive_mode=0.")
        if not 0 <= c["range_min"] < c["range_max"] <= 4095:
            raise ValueError(f"Invalid encoder limits for {name}.")


def raw_to_sim(raw, cfg):
    mid = np.array([(cfg["calibration"][j]["range_min"] + cfg["calibration"][j]["range_max"]) / 2 for j in JOINTS])
    sign = np.array([cfg["mapping"][j]["sign"] for j in JOINTS])
    offset = np.array([cfg["mapping"][j]["offset_rad"] for j in JOINTS])
    return (np.asarray(raw) - mid) * RAD_PER_TICK * sign + offset


def sim_to_raw(q, cfg):
    mid = np.array([(cfg["calibration"][j]["range_min"] + cfg["calibration"][j]["range_max"]) / 2 for j in JOINTS])
    sign = np.array([cfg["mapping"][j]["sign"] for j in JOINTS])
    offset = np.array([cfg["mapping"][j]["offset_rad"] for j in JOINTS])
    return np.rint((np.asarray(q) - offset) / sign / RAD_PER_TICK + mid).astype(int)


def load_config(path):
    cfg = read_json(path)
    validate_calibration(cfg["calibration"])
    if set(cfg["mapping"]) != set(JOINTS):
        raise ValueError("Mapping must contain exactly the six SO-101 joints.")
    for item in cfg["mapping"].values():
        if item["sign"] not in (-1, 1) or not math.isfinite(item["offset_rad"]):
            raise ValueError("Mapping requires sign +/-1 and finite offsets.")
    xml = (Path(path).resolve().parent / cfg["xml"]).resolve()
    return cfg, xml


class Bus:
    """No motor writes on construction. Standard six-motor STS3215 bus only."""
    def __init__(self, port):
        import scservo_sdk as sdk
        self.sdk = sdk
        self.port = sdk.PortHandler(port)
        self.ph = sdk.PacketHandler(0)
        if not self.port.openPort():
            raise RuntimeError(f"Cannot open {port}")
        try:
            if not self.port.setBaudRate(1_000_000):
                raise RuntimeError("Cannot set baud rate to 1,000,000")
            for id_ in IDS:
                if self.read(id_, "model") != 777:
                    raise RuntimeError(f"Motor {id_} is not the expected STS3215 (777).")
            self.reader = sdk.GroupSyncRead(self.port, self.ph, 56, 15)
            for id_ in IDS:
                if not self.reader.addParam(id_):
                    raise RuntimeError(f"Cannot add motor {id_} to sync reader")
        except BaseException:
            self.port.closePort()
            raise

    def check(self, comm, error=0):
        if comm != self.sdk.COMM_SUCCESS or error:
            raise RuntimeError(f"Servo I/O: {self.ph.getTxRxResult(comm)}; status={error}")

    def read(self, id_, key):
        addr, size = REG[key]
        value, comm, error = getattr(self.ph, f"read{size}ByteTxRx")(self.port, id_, addr)
        self.check(comm, error)
        return signed_magnitude(value, 11) if key == "homing_offset" else value

    def write(self, id_, key, value):
        addr, size = REG[key]
        comm, error = getattr(self.ph, f"write{size}ByteTxRx")(self.port, id_, addr, int(value))
        self.check(comm, error)

    def snapshot(self):
        return {j: {key: self.read(id_, key) for key in REG} for j, id_ in zip(JOINTS, IDS)}

    def verify(self, cfg):
        for j, id_ in zip(JOINTS, IDS):
            c = cfg["calibration"][j]
            for reg, field in (("homing_offset", "homing_offset"), ("min_position", "range_min"), ("max_position", "range_max")):
                if self.read(id_, reg) != c[field]:
                    raise RuntimeError(f"{j}: motor {reg} differs from calibration JSON. Recalibrate/reload with LeRobot first.")
            if self.read(id_, "mode") != 0:
                raise RuntimeError(f"{j} is not in position mode; configure with LeRobot first.")
            if self.read(id_, "phase") & 0x10:
                raise RuntimeError(f"{j}: extended angle feedback is enabled. Connect with LeRobot's standard SO follower configuration first.")

    def feedback(self):
        before = time.perf_counter()
        self.check(self.reader.txRxPacket())
        after = time.perf_counter()
        result = {"read_start": before, "read_end": after, "t": (before + after) / 2}
        fields = {"raw_position": (56, 2), "velocity_raw": (58, 2), "load_raw": (60, 2),
                  "voltage_raw": (62, 1), "temperature_c": (63, 1), "status": (65, 1), "current_raw": (69, 2)}
        for field, (addr, size) in fields.items():
            if not all(self.reader.isAvailable(id_, addr, size) for id_ in IDS):
                raise RuntimeError("Incomplete sync-read reply")
            result[field] = [self.reader.getData(id_, addr, size) for id_ in IDS]
        result["raw_position"] = [signed_magnitude(v, 15) for v in result["raw_position"]]
        result["voltage_v"] = [v / 10 for v in result.pop("voltage_raw")]
        if any(not 0 <= v <= 4095 for v in result["raw_position"]):
            raise RuntimeError("Encoder wrap/out-of-range feedback; inspect calibration.")
        return result

    def read_temperatures(self):
        """Read each temperature independently for comparison with sync-read.

        STS3215 Present_Temperature is one byte at address 63. This is a
        diagnostic read only; it does not enable torque or change settings.
        """
        values = []
        for id_ in IDS:
            value, comm, error = self.ph.read1ByteTxRx(self.port, id_, 63)
            self.check(comm, error)
            values.append(value)
        return values

    def goals(self, raw):
        raw = np.asarray(raw, dtype=int)
        if raw.shape != (6,) or np.any(raw < 0) or np.any(raw > 4095):
            raise ValueError("Goal packet must contain six encoder values in 0..4095.")
        writer = self.sdk.GroupSyncWrite(self.port, self.ph, 42, 2)
        for id_, value in zip(IDS, raw):
            if not writer.addParam(id_, [int(value) & 255, int(value) >> 8]):
                raise RuntimeError("Cannot construct goal packet")
        self.check(writer.txPacket())

    def disable(self):
        errors = []
        for id_ in IDS:
            try:
                self.write(id_, "torque_enable", 0)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Torque-off communication failed: " + "; ".join(errors))

    def restore(self, snapshot):
        self.disable()
        for j, id_ in zip(JOINTS, IDS):
            self.write(id_, "lock", 0)
            for key in CHANGED:
                self.write(id_, key, snapshot[j][key])
            self.write(id_, "lock", snapshot[j]["lock"])
        # Never restore Torque_Enable=1 or a stale goal after the arm was moved.

    def close(self):
        self.port.closePort()


@contextmanager
def open_bus(port):
    bus = Bus(port)
    try:
        yield bus
    finally:
        bus.close()


TEMPERATURE_WARNING_C = 50
_last_temperature_warning = float("-inf")


def temperature_warning(row):
    hot = [f"{joint} (ID {id_}): {value} C"
           for joint, id_, value in zip(JOINTS, IDS, row["temperature_c"])
           if value >= TEMPERATURE_WARNING_C]
    return "TEMPERATURE WARNING: " + "; ".join(hot) if hot else None


def telemetry_ok(row, warn_temperature=True):
    """Temperature is advisory only. Status alarms and voltage still abort."""
    global _last_temperature_warning
    warning = temperature_warning(row)
    now = time.perf_counter()
    if warn_temperature and warning and now - _last_temperature_warning >= 5:
        print("\n*** " + warning + ". Temperature alone will NOT stop motion. ***\n", file=sys.stderr)
        _last_temperature_warning = now
    if any(row["status"]):
        raise RuntimeError(f"Servo status alarm: {row['status']}")
    if min(row["voltage_v"]) < 9.0 or max(row["voltage_v"]) > 12.6:
        raise RuntimeError(f"Voltage outside this 12 V test's 9.0..12.6 V envelope: {row['voltage_v']}")


def download(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "so101-sysid"})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def cmd_init(args):
    cal = read_json(args.calibration)
    validate_calibration(cal)
    root = Path(args.out).resolve()
    if (root / "config.json").exists():
        raise FileExistsError("A config already exists here; choose a new --out directory.")
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{SO_SHA}/Simulation/SO101/"
    source = root / "so101_original.xml"
    download(base + "so101_new_calib.xml", source)
    tree = ET.parse(source)
    for mesh in tree.findall(".//asset/mesh"):
        name = mesh.get("file")
        if not name or Path(name).name != name:
            raise ValueError("Unexpected mesh path in pinned upstream model.")
        download(base + "assets/" + name, assets / name)
    xml = tree.getroot()
    # Remove actuator defaults, including general actuator fields inherited by
    # shortcuts. Old position ctrlrange must never clip BAM torque commands.
    actuator_tags = {"position", "velocity", "general", "motor", "intvelocity", "damper"}
    for default in xml.findall(".//default"):
        for child in list(default):
            if child.tag in actuator_tags:
                default.remove(child)
    actuators = xml.find("actuator")
    if actuators is None:
        raise ValueError("Upstream MJCF has no actuator section.")
    actuators.clear()
    for j in JOINTS:
        ET.SubElement(actuators, "motor", name=j, joint=j, gear="1", ctrllimited="false", forcelimited="false")
    option = xml.find("option")
    if option is None:
        option = ET.SubElement(xml, "option")
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")
    ET.indent(tree)
    tree.write(root / "so101_bam.xml")
    cfg = {"schema": 1, "xml": "so101_bam.xml", "so101_commit": SO_SHA, "bam_commit": BAM_SHA,
           "calibration": cal, "mapping_reviewed": False, "free_space": True,
           "mapping": {j: {"sign": 1, "offset_rad": 0.0} for j in JOINTS}}
    params = {"schema": 1, "source_commit": PR_SHA, "joints": {
        j: {"bam": copy.deepcopy(SEED), "command_delay_s": 0.0} for j in JOINTS}}
    save_json(root / "config.json", cfg)
    save_json(root / "seed_params.json", params)
    print(f"Created {root / 'config.json'} and seed_params.json. Next run align.")


def cmd_inspect(args):
    with open_bus(args.port) as bus:
        result = {"settings": bus.snapshot(), "feedback": bus.feedback()}
        result["temperature_check"] = {
            "joint_order": JOINTS,
            "sync_read_c": result["feedback"]["temperature_c"],
            "individual_read_c": bus.read_temperatures(),
        }
        if args.config:
            cfg, _ = load_config(args.config)
            bus.verify(cfg)
            result["calibration_matches"] = True
        save_json(args.out, result)
    print(json.dumps(result, indent=2))


def cmd_restore(args):
    settings = read_json(args.snapshot)["settings"]
    input("Support the arm. ENTER disables torque and restores the saved gains/profile settings: ")
    with open_bus(args.port) as bus:
        bus.restore(settings)
    print("Settings restored; torque remains OFF.")


def cmd_align(args):
    import glfw
    import mujoco
    import mujoco.viewer
    cfg, xml = load_config(args.config)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    qadr = [model.joint(j).qposadr[0] for j in JOINTS]
    selected = [0]
    saved = [False]
    mutex = threading.Lock()
    # Unassigned viewer shortcuts in MuJoCo 3.12. The passive callback does
    # not consume events, so avoid digits, letters, brackets and F1..F7.
    offset_keys = {
        glfw.KEY_HOME: -.5, glfw.KEY_END: .5,
        glfw.KEY_INSERT: -5., glfw.KEY_DELETE: 5.,
    }
    alignment_keys = {glfw.KEY_F8, glfw.KEY_F9, glfw.KEY_F10, glfw.KEY_F12, *offset_keys}
    print("Move one physical joint at a time. Match both its direction and link orientation.")
    print("Viewer keys: F8/F9 select previous/next joint; F10 flips its sign;")
    print("Home/End change offset by -/+0.5 deg; Insert/Delete by -/+5 deg;")
    print("F12 saves mapping after YOU check all joints. Focus the 3D view, not a text field.")

    def print_selection():
        joint = JOINTS[selected[0]]
        item = cfg["mapping"][joint]
        print(f"Joint {selected[0]+1}/{len(JOINTS)}: {joint}, "
              f"sign={item['sign']:+g}, offset={math.degrees(item['offset_rad']):+.3f} deg")

    print_selection()

    def keypress(key):
        if key not in alignment_keys:
            return
        with mutex:
            if key == glfw.KEY_F8:
                selected[0] = (selected[0] - 1) % len(JOINTS)
            elif key == glfw.KEY_F9:
                selected[0] = (selected[0] + 1) % len(JOINTS)
            item = cfg["mapping"][JOINTS[selected[0]]]
            if key == glfw.KEY_F10:
                item["sign"] *= -1
                saved[0] = False
            elif key in offset_keys:
                item["offset_rad"] += math.radians(offset_keys[key])
                saved[0] = False
            elif key == glfw.KEY_F12:
                cfg["mapping_reviewed"] = True
                save_json(args.config, cfg, overwrite=True)
                saved[0] = True
                print("Mapping saved. Close the viewer when finished.")
            print_selection()

    with open_bus(args.port) as bus:
        bus.verify(cfg)
        input("Support the arm. ENTER disables torque for MANUAL alignment: ")
        bus.disable()
        with mujoco.viewer.launch_passive(model, data, key_callback=keypress) as viewer:
            while viewer.is_running():
                row = bus.feedback()
                with mutex, viewer.lock():
                    data.qpos[qadr] = raw_to_sim(row["raw_position"], cfg)
                    mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(.025)
    if not saved[0]:
        print("No current mapping saved. Run align again and press F12 after checking all joints.")


def waveform(t, duration, amplitude, frequency, kind):
    if t <= 0 or t >= duration:
        return 0.0
    if kind == "sine":
        # Smoothly enter/exit; no sudden full-amplitude first command.
        edge = min(1.0, t / 2, (duration - t) / 2)
        envelope = .5 - .5 * math.cos(math.pi * edge)
        return amplitude * envelope * math.sin(2 * math.pi * frequency * t)
    values = [0, amplitude, 0, -amplitude, 0]
    phase = t / duration * 4
    k = min(3, int(phase))
    w = .5 - .5 * math.cos(math.pi * (phase - k))
    return values[k] * (1 - w) + values[k + 1] * w


def check_motion_bounds(raw, start, cfg, amplitude_deg):
    margin = math.ceil(math.radians(3) / RAD_PER_TICK)
    for k, j in enumerate(JOINTS):
        c = cfg["calibration"][j]
        if not c["range_min"] + margin <= raw[k] <= c["range_max"] - margin:
            raise RuntimeError(f"{j}: too close to calibrated travel limit. Start nearer midrange.")
        if abs(raw[k] - start[k]) * RAD_PER_TICK > math.radians(amplitude_deg + 8):
            raise RuntimeError(f"{j}: unexpected excursion from starting pose.")


def wait_supported(bus):
    done = threading.Event()
    errors = []
    def prompt():
        try:
            input("Support the arm, then ENTER to disable torque and restore settings: ")
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()
    threading.Thread(target=prompt, daemon=True).start()
    while not done.wait(.2):
        telemetry_ok(bus.feedback())
    if errors:
        raise errors[0]


def cmd_record(args):
    cfg, _ = load_config(args.config)
    if not cfg["mapping_reviewed"]:
        raise RuntimeError("Run align and save the checked mapping first.")
    if not (0 < args.amplitude <= 20 and 0 < args.duration <= 120 and 5 <= args.rate <= 100 and 0 < args.p <= 32 and 0 < args.frequency <= 1 and 0 < args.max_speed <= 30):
        raise ValueError("Envelope: amplitude<=20 deg, duration<=120 s, rate=5..100 Hz, P=1..32, frequency<=1 Hz, max_speed<=30 deg/s.")
    if args.joint == "all" and args.amplitude > 5:
        raise ValueError("Coordinated tests are limited to a maximum 5-degree excursion.")
    output = Path(args.out)
    backup = output.with_suffix(".settings.json")
    if output.exists() or backup.exists():
        raise FileExistsError("Output/snapshot exists; choose new filenames.")
    document = {"schema": 1, "complete": False, "config": cfg, "joint": args.joint,
                "trajectory": args.trajectory, "command_rate_hz": args.rate,
                "p": args.p, "i": 0, "d": 0, "commands": [], "samples": []}
    settings = None
    changed = False
    with open_bus(args.port) as bus:
        bus.verify(cfg)
        settings = bus.snapshot()
        if any(v["torque_enable"] != 0 for v in settings.values()):
            raise RuntimeError("Start with torque OFF and the arm supported. Close LeRobot/teleop; use align to relax the arm.")
        save_json(backup, {"settings": settings})  # Persist before the first write.
        document["original_settings"] = settings
        input(f"Clear the workspace. Support a midrange starting pose. ENTER prepares P={args.p}, I=D=0 and holds that pose: ")
        try:
            initial = bus.feedback()
            telemetry_ok(initial)
            start = np.array(initial["raw_position"])
            check_motion_bounds(start, start, cfg, args.amplitude)
            weights = COORDINATED_WEIGHTS.copy() if args.joint == "all" else np.eye(6)[JOINTS.index(args.joint)]
            for sign in (-1, 1):
                extreme = start + np.rint(sign * weights * math.radians(args.amplitude) / RAD_PER_TICK).astype(int)
                check_motion_bounds(extreme, start, cfg, args.amplitude)
            changed = True
            for id_ in IDS:
                bus.write(id_, "lock", 0)
                for key, value in {"p": args.p, "i": 0, "d": 0, "acceleration": 254, "max_acceleration": 254, "goal_velocity": 0}.items():
                    bus.write(id_, key, value)
            bus.goals(start)  # Seed all goals before any motor is enabled.
            for id_ in IDS:
                bus.write(id_, "torque_enable", 1)
                bus.write(id_, "lock", 1)
            print("Holding the starting pose. Remove your support; keep hands clear.")
            settle_end = time.perf_counter() + 3
            while time.perf_counter() < settle_end:
                row = bus.feedback()
                telemetry_ok(row)
                check_motion_bounds(row["raw_position"], start, cfg, args.amplitude)
                time.sleep(.02)
            effective = bus.snapshot()
            if any(v["p"] != args.p or v["i"] != 0 or v["d"] != 0 for v in effective.values()):
                raise RuntimeError("Gain readback differs from requested configuration.")
            document["effective_settings"] = effective
            origin = time.perf_counter()
            document["initial_target_raw"] = start.tolist()
            row = bus.feedback()
            document["initial_voltage_v"] = row["voltage_v"]
            last_offset = np.zeros(6)
            last_time = row["t"] - origin
            next_tick = last_time
            actual_duration = args.duration + 2  # Last two seconds hold original target.
            while True:
                t = row["t"] - origin
                telemetry_ok(row)
                check_motion_bounds(row["raw_position"], start, cfg, args.amplitude)
                row["q_rad"] = raw_to_sim(row["raw_position"], cfg).tolist()
                for key in ("t", "read_start", "read_end"):
                    row[key] -= origin
                document["samples"].append(row)
                if t >= actual_duration:
                    break
                requested = weights * waveform(t, args.duration, math.radians(args.amplitude), args.frequency, args.trajectory)
                max_change = math.radians(args.max_speed) * max(0, t - last_time)
                offset = np.clip(requested, last_offset - max_change, last_offset + max_change)
                target = start + np.rint(offset / RAD_PER_TICK).astype(int)
                check_motion_bounds(target, start, cfg, args.amplitude)
                tx_start = time.perf_counter() - origin
                bus.goals(target)
                tx_end = time.perf_counter() - origin
                document["commands"].append({"tx_start": tx_start, "t": tx_end,
                    "raw_target": target.tolist(), "q_target_rad": raw_to_sim(target, cfg).tolist(),
                    "requested_offset_rad": requested.tolist()})
                last_offset, last_time = offset, t
                next_tick += 1 / args.rate
                remaining = origin + next_tick - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_tick = time.perf_counter() - origin  # Do not burst queued commands.
                row = bus.feedback()
            document["complete"] = True
            print("Recording complete; holding the starting target.")
            wait_supported(bus)
        except BaseException as exc:
            document["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if changed:
                try:
                    bus.restore(settings)
                    document["settings_restored"] = True
                except Exception as exc:
                    document["settings_restored"] = False
                    document["restore_error"] = str(exc)
                    print(f"RESTORE FAILED. Support arm and cut motor power; snapshot: {backup}", file=sys.stderr)
            save_json(output, document)
    print(f"Saved {output}. Torque is OFF. P/I/D/profile settings restored.")


def register_12v():
    from bam.actuators import actuators
    from bam.feetech.actuator import STS3215Actuator
    from bam.testbench import Pendulum
    # Explicit registration fixes the experimental JSON's missing name.
    def factory():
        act = STS3215Actuator(Pendulum)
        act.vin = 12.0
        return act
    actuators["sts321512v"] = factory


class Simulation:
    def __init__(self, xml, cfg, params, log, dt=.002):
        import mujoco
        from bam.model import load_model_from_dict
        from bam.mujoco import MujocoController
        register_12v()
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.model.opt.timestep = dt
        if cfg.get("free_space", True):
            self.model.geom_contype[:] = 0
            self.model.geom_conaffinity[:] = 0
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.joint(j).qposadr[0] for j in JOINTS]
        self.vadr = [self.model.joint(j).dofadr[0] for j in JOINTS]
        self.controllers = []
        q_initial_target = raw_to_sim(log["initial_target_raw"], cfg)
        settings = log["effective_settings"]
        for k, j in enumerate(JOINTS):
            p = copy.deepcopy(params["joints"][j]["bam"])
            p["actuator"] = "sts321512v"
            p["q_offset"] = 0.0
            bam = load_model_from_dict(p)
            bam.actuator.kp = log["p"]
            bam.actuator.vin = log["initial_voltage_v"][k]
            # Initial approximation: the firmware torque-limit percentage caps
            # PWM. Overload/current-protection timing is NOT modeled here.
            ratio = min(settings[j]["max_torque"], settings[j]["torque_limit"]) / 1000
            if not 0 < ratio <= 1:
                raise ValueError(f"Unexpected torque limit for {j}: {ratio}")
            bam.actuator.max_pwm = .97 * ratio
            ctrl = MujocoController(bam, j, self.model, self.data)
            ctrl.reset()
            ctrl.set_q_target(j, q_initial_target[k])
            ctrl.actuator_state = (np.array([q_initial_target[k]]), [])
            self.controllers.append(ctrl)
        # BAM's constructors call mj_setConst, which can reset data.qpos.
        # Restore the measured state AFTER every controller is constructed.
        self.data.qpos[self.qadr] = log["samples"][0]["q_rad"]
        self.data.qvel[self.vadr] = log.get("initial_velocity_rad_s", [0.] * 6)
        mujoco.mj_forward(self.model, self.data)

    def step(self, targets):
        for j, ctrl, target in zip(JOINTS, self.controllers, targets):
            ctrl.set_q_target(j, float(target))
            ctrl.update()
        self.mj.mj_step(self.model, self.data)
        q = self.data.qpos[self.qadr].copy()
        if not np.all(np.isfinite(q)) or any(self.data.warning[i].number for i in range(len(self.data.warning))):
            raise RuntimeError("MuJoCo numerical warning; inspect initial pose/model and timestep.")
        return q


def load_trial(path, cfg):
    log = read_json(path)
    if not log.get("complete") or len(log.get("samples", [])) < 5:
        raise ValueError(f"{path}: incomplete/too-short trial")
    if log["config"]["calibration"] != cfg["calibration"] or log["config"]["mapping"] != cfg["mapping"]:
        raise ValueError("Trial calibration/mapping differs. Re-record after changing coordinates.")
    if log["i"] != 0 or log["d"] != 0:
        raise ValueError("This baseline implements P-only control.")
    ts = np.array([s["t"] for s in log["samples"]])
    if np.any(np.diff(ts) <= 0):
        raise ValueError("Sample times are not strictly increasing.")
    ct = np.array([c["t"] for c in log.get("commands", [])])
    if not len(ct) or not np.all(np.isfinite(ct)) or np.any(np.diff(ct) <= 0):
        raise ValueError("Trial needs finite, strictly increasing command times.")
    if not np.all(np.isfinite(ts)):
        raise ValueError("Non-finite sample times.")
    return log


def rollout(xml, cfg, params, log, dt=.002):
    sim = Simulation(xml, cfg, params, log, dt)
    samples_t = np.array([s["t"] for s in log["samples"]])
    origin = samples_t[0]
    command_t = np.array([c["t"] for c in log["commands"]]) - origin
    commands = np.array([c["q_target_rad"] for c in log["commands"]])
    initial_target = raw_to_sim(log["initial_target_raw"], cfg)
    delay = np.array([params["joints"][j].get("command_delay_s", 0.) for j in JOINTS])
    times = [0.]
    positions = [sim.data.qpos[sim.qadr].copy()]
    end = samples_t[-1] - origin
    while sim.data.time < end:
        indexes = np.searchsorted(command_t, sim.data.time - delay, side="right") - 1
        target = initial_target.copy()
        valid = indexes >= 0
        target[valid] = commands[indexes[valid], np.arange(6)[valid]]
        positions.append(sim.step(target))
        times.append(sim.data.time)
    positions = np.array(positions)
    return np.column_stack([np.interp(samples_t - origin, times, positions[:, k]) for k in range(6)])


def metrics(log, predicted):
    actual = np.array([s["q_rad"] for s in log["samples"]])
    error = np.rad2deg(predicted - actual)
    gaps = np.diff([s["t"] for s in log["samples"]])
    volts = np.array([s["voltage_v"] for s in log["samples"]])
    return {"joint_errors_deg": {j: {"rmse": float(np.sqrt(np.mean(error[:, k]**2))),
            "p95_abs": float(np.percentile(abs(error[:, k]), 95)), "max_abs": float(max(abs(error[:, k])))} for k, j in enumerate(JOINTS)},
            "sample_interval_ms_median": float(np.median(gaps)*1000),
            "sample_interval_ms_p95": float(np.percentile(gaps,95)*1000),
            "observed_voltage_span_v": dict(zip(JOINTS, np.ptp(volts,axis=0).tolist()))}


def cmd_replay(args):
    cfg, xml = load_config(args.config)
    params = read_json(args.params)
    log = load_trial(args.log, cfg)
    prediction = rollout(xml, cfg, params, log, args.dt)
    result = metrics(log, prediction)
    out = Path(args.out)
    save_json(out.with_suffix(".json"), result)
    table = np.column_stack(([s["t"] for s in log["samples"]], prediction))
    np.savetxt(out.with_suffix(".csv"), table, delimiter=",", header="time_s," + ",".join(JOINTS), comments="")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,2,figsize=(12,9),sharex=True)
    actual = np.array([s["q_rad"] for s in log["samples"]])
    t = [s["t"] for s in log["samples"]]
    ct = [c["t"] for c in log["commands"]]
    cq = np.array([c["q_target_rad"] for c in log["commands"]])
    for k, ax in enumerate(axes.flat):
        ax.step(ct,np.rad2deg(cq[:,k]),where="post",color=".7",label="Sent target",lw=1)
        ax.plot(t,np.rad2deg(actual[:,k]),label="Real",lw=1.6)
        ax.plot(t,np.rad2deg(prediction[:,k]),label="MuJoCo + BAM",lw=1.2)
        ax.set(title=JOINTS[k],ylabel="Joint angle (deg)")
        ax.grid(alpha=.2)
    axes[0,0].legend()
    for ax in axes[-1]: ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"),dpi=160)
    plt.close(fig)
    print(json.dumps(result,indent=2))
    print(f"Saved {out.with_suffix('.png')}, .json and .csv")


def cmd_fit(args):
    from scipy.optimize import minimize
    cfg, xml = load_config(args.config)
    params = read_json(args.params)
    logs = [load_trial(p,cfg) for p in args.log]
    if any(log["joint"] != args.joint for log in logs):
        raise ValueError("Fit one joint at a time, using trials that moved that joint.")
    if Path(args.out).exists():
        raise FileExistsError("Choose a new output file.")
    k = JOINTS.index(args.joint)
    base = params["joints"][args.joint]
    # Limited effective whole-arm fit: hold CAD mass, Kt, armature, and target
    # speed limit fixed. Do not pretend small sweeps uniquely identify all of them.
    names = ["R", "friction_base", "friction_viscous", "command_delay_s"]
    bounds = [(0.5, 8.), (0., .6), (0., .3), (0., .08)]
    lower = np.array([b[0] for b in bounds]); upper = np.array([b[1] for b in bounds])
    initial = np.array([base["bam"][n] for n in names[:3]] + [base.get(names[3],0.)])
    best = {"score": float("inf"), "x": initial.copy()}
    count = [0]

    def unpack(z):
        x = lower + np.asarray(z) * (upper-lower)
        candidate = copy.deepcopy(params)
        for n,v in zip(names[:3],x[:3]): candidate["joints"][args.joint]["bam"][n] = float(v)
        candidate["joints"][args.joint][names[3]] = float(x[3])
        return x,candidate

    def objective(z):
        x,candidate = unpack(z)
        count[0] += 1
        try:
            errors = [np.rad2deg(rollout(xml,cfg,candidate,log,args.dt)[:,k] - np.array([s["q_rad"][k] for s in log["samples"]])) for log in logs]
            score = float(np.sqrt(np.mean(np.concatenate(errors)**2)))
        except (RuntimeError,ValueError,FloatingPointError):
            score = 1e6
        if score < best["score"]:
            best.update(score=score,x=x.copy())
        if count[0] % 10 == 0:
            print(f"Evaluation {count[0]}: best training RMSE {best['score']:.3f} deg",flush=True)
        return score

    z0=(initial-lower)/(upper-lower)
    baseline = objective(z0)
    minimize(objective,z0,method="Powell",bounds=[(0.,1.)]*4,
             options={"maxfev":args.max_evals,"xtol":.01,"ftol":.005})
    if best["score"] >= 1e5:
        raise RuntimeError("No valid rollout found; inspect the seed replay before fitting.")
    _,fitted=unpack((best["x"]-lower)/(upper-lower))
    fitted.setdefault("fit_history",[]).append({"joint":args.joint,"logs":args.log,
        "baseline_rmse_deg":baseline,"training_rmse_deg":best["score"],
        "parameters":dict(zip(names,best["x"].tolist())),"physics_dt":args.dt})
    save_json(args.out,fitted)
    print(json.dumps(fitted["fit_history"][-1],indent=2))
    print("Now replay a different real trial with this file. Training error is not validation.")


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("init",help="Download pinned SO-101 MJCF/meshes and create configuration")
    q.add_argument("--calibration",required=True); q.add_argument("--out",default="work"); q.set_defaults(fn=cmd_init)
    q=sub.add_parser("inspect",help="Read-only motor settings and feedback")
    q.add_argument("--port",required=True); q.add_argument("--config"); q.add_argument("--out",required=True); q.set_defaults(fn=cmd_inspect)
    q=sub.add_parser("restore",help="Restore saved gains/profile settings, leaving torque OFF")
    q.add_argument("--port",required=True); q.add_argument("--snapshot",required=True); q.set_defaults(fn=cmd_restore)
    q=sub.add_parser("align",help="Torque OFF; manually align physical and rendered arm")
    q.add_argument("--port",required=True); q.add_argument("--config",default="work/config.json"); q.set_defaults(fn=cmd_align)
    q=sub.add_parser("record",help="Bounded P-only joint sweep on the REAL arm")
    q.add_argument("--port",required=True); q.add_argument("--config",default="work/config.json")
    q.add_argument("--joint",choices=JOINTS+["all"],required=True); q.add_argument("--trajectory",choices=["sine","ramp"],default="sine")
    q.add_argument("--amplitude",type=float,default=5,help="Motor-output excursion in degrees")
    q.add_argument("--frequency",type=float,default=.25); q.add_argument("--duration",type=float,default=16)
    q.add_argument("--rate",type=float,default=50); q.add_argument("--p",type=int,default=16)
    q.add_argument("--max-speed",type=float,default=15); q.add_argument("--out",required=True); q.set_defaults(fn=cmd_record)
    for name,fn in (("replay",cmd_replay),("fit",cmd_fit)):
        q=sub.add_parser(name)
        q.add_argument("--config",default="work/config.json"); q.add_argument("--params",default="work/seed_params.json")
        q.add_argument("--log",action="append" if name=="fit" else "store",required=True)
        q.add_argument("--dt",type=float,default=.002); q.add_argument("--out",required=True)
        if name=="fit":
            q.add_argument("--joint",choices=JOINTS,required=True); q.add_argument("--max-evals",type=int,default=120)
        q.set_defaults(fn=fn)
    return p


if __name__=="__main__":
    args=parser().parse_args()
    try:
        if hasattr(args,"dt") and not .00025 <= args.dt <= .005:
            raise ValueError("Choose a physics dt between 0.00025 and 0.005 seconds.")
        args.fn(args)
    except KeyboardInterrupt:
        print("Interrupted. If communication was lost, support the arm and cut motor power.",file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        sys.exit(1)
