"""Offline checks. No serial ports are opened and no hardware is moved.

Run: python -m unittest -v test_so101_sysid.py
Add SO101_TEST_XML=/absolute/path/to/work/so101_bam.xml for real MuJoCo checks.
"""
import copy
import json
import math
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import so101_sysid as s


def fixture():
    cal = {j: {"id": i, "drive_mode": 0, "homing_offset": 0,
               "range_min": 512, "range_max": 3583}
           for i, j in enumerate(s.JOINTS, 1)}
    cfg = {"schema": 1, "xml": "unused.xml", "calibration": cal,
           "mapping_reviewed": True, "free_space": True,
           "mapping": {j: {"sign": 1, "offset_rad": 0.0} for j in s.JOINTS}}
    params = {"schema": 1, "joints": {j: {"bam": copy.deepcopy(s.SEED),
              "command_delay_s": 0.0} for j in s.JOINTS}}
    settings = {j: {key: 0 for key in s.REG} for j in s.JOINTS}
    for row in settings.values():
        row.update(p=16, d=32, max_torque=1000, torque_limit=1000, lock=1)
    raw = [2048] * 6
    initial_q = s.raw_to_sim(raw, cfg).tolist()
    log = {"schema": 1, "synthetic": True, "complete": True, "config": cfg,
           "joint": "shoulder_lift", "p": 16, "i": 0, "d": 0,
           "initial_target_raw": raw, "initial_voltage_v": [12.] * 6,
           "effective_settings": settings,
           "commands": [], "samples": [{"t": 0., "q_rad": initial_q,
                                         "voltage_v": [12.] * 6}]}
    return cfg, params, log


class CoordinateAndHardwareChecks(unittest.TestCase):
    def test_raw_round_trip_including_reversed_gripper(self):
        cfg, _, _ = fixture()
        cfg["mapping"]["gripper"] = {"sign": -1, "offset_rad": .71}
        cfg["mapping"]["elbow_flex"] = {"sign": -1, "offset_rad": -.23}
        raw = np.array([700, 1300, 2200, 3100, 2011, 1789])
        np.testing.assert_array_equal(s.sim_to_raw(s.raw_to_sim(raw, cfg), cfg), raw)
        moved = raw.copy()
        moved[5] += 10
        self.assertAlmostEqual(s.raw_to_sim(moved, cfg)[5] - s.raw_to_sim(raw, cfg)[5],
                               -10 * s.RAD_PER_TICK)

    def test_invalid_motor_id_and_calibration_are_rejected(self):
        cfg, _, _ = fixture()
        cfg["calibration"]["gripper"]["id"] = 1
        with self.assertRaises(ValueError):
            s.validate_calibration(cfg["calibration"])
        cfg["calibration"]["gripper"]["id"] = 6
        cfg["calibration"]["gripper"]["range_max"] = 5000
        with self.assertRaises(ValueError):
            s.validate_calibration(cfg["calibration"])

    def test_limits_and_unexpected_motion_abort(self):
        cfg, _, _ = fixture()
        start = np.array([2048] * 6)
        s.check_motion_bounds(start, start, cfg, 5)
        for value in (512, 3583, 2500):
            bad = start.copy()
            bad[2] = value
            with self.assertRaises(RuntimeError):
                s.check_motion_bounds(bad, start, cfg, 5)

    def test_waveforms_have_no_start_jump_and_stay_bounded(self):
        for kind in ("sine", "ramp"):
            values = np.array([s.waveform(t, 16, math.radians(5), .25, kind)
                               for t in np.linspace(0, 18, 1801)])
            self.assertEqual(values[0], 0)
            self.assertEqual(values[-1], 0)
            self.assertLessEqual(max(abs(values)), math.radians(5) + 1e-12)
            self.assertLess(abs(values[1]), math.radians(.01))

    def test_restore_disables_before_writes_and_never_reenables(self):
        _, _, log = fixture()
        events = []
        bus = object.__new__(s.Bus)
        bus.write = lambda *args: events.append(args)
        bus.restore(log["effective_settings"])
        self.assertEqual(events[:6], [(i, "torque_enable", 0) for i in s.IDS])
        self.assertFalse(any(key == "torque_enable" and value for _, key, value in events))
        self.assertTrue(all(key in set(s.CHANGED) | {"lock", "torque_enable"}
                            for _, key, _ in events))

    def test_record_saves_backup_then_seeds_goals_before_enable_and_restores_on_error(self):
        cfg, _, log = fixture()
        events = []
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "config.json"
            output = Path(td) / "trial.json"
            s.save_json(config, cfg)

            class FakeBus:
                def verify(self, _): pass
                def snapshot(self): return copy.deepcopy(log["effective_settings"])
                def feedback(self):
                    return {"raw_position": [2048]*6, "status": [0]*6,
                            "temperature_c": [25]*6, "voltage_v": [12.]*6}
                def write(self, id_, key, value):
                    assert output.with_suffix(".settings.json").exists()
                    events.append((id_, key, value))
                    if key == "torque_enable" and value == 1:
                        raise RuntimeError("Injected connection loss")
                def goals(self, raw): events.append(("goals", list(raw)))
                def restore(self, snapshot): events.append(("restore", snapshot))

            @contextmanager
            def fake_open(_): yield FakeBus()

            args = SimpleNamespace(config=str(config), out=str(output), port="FAKE",
                                   joint="shoulder_lift", trajectory="sine", amplitude=5,
                                   duration=16, rate=50, p=16, frequency=.25, max_speed=15)
            with patch.object(s, "open_bus", fake_open), patch("builtins.input", return_value=""):
                with self.assertRaisesRegex(RuntimeError, "Injected connection loss"):
                    s.cmd_record(args)
            goal_index = next(i for i,e in enumerate(events) if e[0] == "goals")
            enable_index = next(i for i,e in enumerate(events) if len(e) == 3 and e[1] == "torque_enable")
            self.assertLess(goal_index, enable_index)
            self.assertEqual(events[-1][0], "restore")
            failed = json.loads(output.read_text())
            self.assertFalse(failed["complete"])
            self.assertTrue(failed["settings_restored"])

    def test_coordinated_record_completes_with_bounded_quantized_targets(self):
        cfg, _, log = fixture()
        class Clock:
            now = 0.
            def perf_counter(self):
                self.now += .0001
                return self.now
            def sleep(self, seconds): self.now += seconds
        clock = Clock()
        class FakeBus:
            def __init__(self):
                self.settings = copy.deepcopy(log["effective_settings"])
            def verify(self, _): pass
            def snapshot(self): return copy.deepcopy(self.settings)
            def write(self, id_, key, value): self.settings[s.JOINTS[id_-1]][key] = value
            def goals(self, raw): pass
            def restore(self, snapshot): self.settings = copy.deepcopy(snapshot)
            def feedback(self):
                t = clock.perf_counter()
                return {"raw_position": [2048]*6, "status": [0]*6,
                        "temperature_c": [25]*6, "voltage_v": [12.]*6,
                        "t": t, "read_start": t-.0001, "read_end": t+.0001}
        @contextmanager
        def fake_open(_): yield FakeBus()
        with tempfile.TemporaryDirectory() as td:
            config, output = Path(td)/"config.json", Path(td)/"trial.json"
            s.save_json(config, cfg)
            args = SimpleNamespace(config=str(config), out=str(output), port="FAKE",
                                   joint="all", trajectory="sine", amplitude=2,
                                   duration=4, rate=10, p=16, frequency=.25, max_speed=5)
            with patch.object(s,"open_bus",fake_open), patch.object(s,"time",clock), \
                 patch.object(s,"wait_supported"), patch("builtins.input",return_value=""):
                s.cmd_record(args)
            trial = s.load_trial(output,cfg)
            self.assertTrue(trial["complete"])
            self.assertTrue(trial["settings_restored"])
            raw = np.array([c["raw_target"] for c in trial["commands"]])
            self.assertTrue(np.all(np.max(abs(raw-2048),axis=0)*s.RAD_PER_TICK <=
                                   abs(s.COORDINATED_WEIGHTS)*math.radians(2) + s.RAD_PER_TICK))
            self.assertTrue(np.all(np.ptp(raw,axis=0) > 0))
            self.assertTrue(all(v["d"] == 0 for v in trial["effective_settings"].values()))


@unittest.skipUnless(os.environ.get("SO101_TEST_XML"), "Set SO101_TEST_XML for MuJoCo tests")
class MuJoCoChecks(unittest.TestCase):
    def test_torque_actuators_and_replay_reset_are_deterministic(self):
        cfg, params, log = fixture()
        xml = Path(os.environ["SO101_TEST_XML"])
        sim = s.Simulation(xml, cfg, params, log)
        np.testing.assert_array_equal(sim.data.qpos[sim.qadr], log["samples"][0]["q_rad"])
        nonzero_log = copy.deepcopy(log)
        nonzero_log["samples"][0]["q_rad"] = [.2, -.4, .3, -.2, .1, .6]
        nonzero_log["initial_velocity_rad_s"] = [.01, -.02, .03, 0., 0., 0.]
        nonzero_sim = s.Simulation(xml, cfg, params, nonzero_log)
        np.testing.assert_array_equal(nonzero_sim.data.qpos[nonzero_sim.qadr], nonzero_log["samples"][0]["q_rad"])
        np.testing.assert_array_equal(nonzero_sim.data.qvel[nonzero_sim.vadr], nonzero_log["initial_velocity_rad_s"])
        self.assertEqual(sim.model.nu, 6)
        self.assertFalse(np.any(sim.model.actuator_ctrllimited))
        self.assertFalse(np.any(sim.model.actuator_forcelimited))
        self.assertTrue(np.all(sim.model.actuator_gear[:, 0] == 1))
        self.assertTrue(np.all(sim.model.geom_contype == 0))
        self.assertTrue(all(c.model.actuator.vin == 12. for c in sim.controllers))
        self.assertTrue(all(c.model.actuator.kp == 16 for c in sim.controllers))
        for t in np.arange(0, 1.01, .02):
            q = s.raw_to_sim(log["initial_target_raw"], cfg)
            q[1] += .06 * math.sin(2 * math.pi * .5 * t)
            log["commands"].append({"t": float(t), "q_target_rad": q.tolist()})
        log["samples"] = [{"t": float(t), "q_rad": log["samples"][0]["q_rad"],
                           "voltage_v": [12.] * 6} for t in np.arange(0, 1.01, .02)]
        first = s.rollout(xml, cfg, params, log)
        second = s.rollout(xml, cfg, params, log)
        np.testing.assert_array_equal(first, second)
        delayed = copy.deepcopy(params)
        delayed["joints"]["shoulder_lift"]["command_delay_s"] = .06
        third = s.rollout(xml, cfg, delayed, log)
        self.assertGreater(np.max(abs(first[:,1] - third[:,1])), 1e-5)


if __name__ == "__main__":
    unittest.main()
