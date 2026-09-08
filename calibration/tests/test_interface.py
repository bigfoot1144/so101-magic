"""Offline behavioral/numerical tests. No serial port is opened."""
import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import so101_sysid as s
import so101_calibration_core as c


def config():
    return {"schema": 1, "xml": "unused.xml", "mapping_reviewed": True, "free_space": True,
            "calibration": {j: {"id": k + 1, "drive_mode": 0, "homing_offset": 0,
                                  "range_min": 512, "range_max": 3583} for k, j in enumerate(s.JOINTS)},
            "mapping": {j: {"sign": 1, "offset_rad": 0.} for j in s.JOINTS}}


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now
    def advance(self, dt=.02): self.now += dt


class FakeMechanics:
    limits = np.array([[-2.5, 2.5]] * 6)
    def check_q(self, q, margin=True):
        if np.any(abs(q) > 2.4): raise ValueError("model limit")
    def gravity(self, q): return np.sin(q)


class SpyBus(c.DemoBus):
    def __init__(self, cfg, clock):
        super().__init__(cfg, FakeMechanics(), clock)
        self.events = []
        self.temperature = [30] * 6
        self.individual_temperature = [30] * 6
    def write(self, *args):
        self.events.append(("write", *args))
        super().write(*args)
    def goals(self, raw):
        self.events.append(("goals", vector_list(raw)))
        super().goals(raw)
    def read_temperatures(self): return self.individual_temperature.copy()
    def feedback(self):
        row = super().feedback()
        row["temperature_c"] = self.temperature.copy()
        return row


def vector_list(value): return np.asarray(value).tolist()


class WorkerBehavior(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = config()
        self.clock = Clock()
        self.bus = SpyBus(self.cfg, self.clock)
        self.engine = c.Engine(self.bus, self.cfg, FakeMechanics(), self.root, True, self.clock)
        self.engine.tick()

    def tearDown(self):
        self.engine.close()
        self.tmp.cleanup()

    def step(self, n=1):
        for _ in range(n):
            self.clock.advance()
            self.engine.last_heartbeat = self.clock()
            self.engine.tick()

    def test_connect_and_reference_edits_do_not_write_to_motors(self):
        self.step(40)
        self.engine.freeze()
        reference = np.array(self.engine.frozen["mapped_q_rad"]) + .01
        self.engine.capture(reference, "Test reference")
        self.assertEqual(self.bus.events, [])
        pairs = s.read_json(self.root / "pose_pairs.json")["poses"]
        np.testing.assert_allclose(pairs[0]["gravity_reference_nm"], np.sin(reference))

    def test_enable_persists_settings_and_seeds_all_goals_before_any_enable(self):
        self.engine.arm()
        self.assertTrue(list(self.root.glob("settings-*.json")))
        seeded = next(i for i, e in enumerate(self.bus.events) if e[0] == "goals")
        enabled = [i for i, e in enumerate(self.bus.events) if e[0] == "write" and e[2:] == ("torque_enable", 1)]
        self.assertEqual(len(enabled), 6)
        self.assertLess(seeded, min(enabled))
        self.assertTrue(all(row["d"] == 0 for row in self.bus.settings.values()))
        self.engine.disable()
        self.assertTrue(all(row["d"] == 32 and row["torque_enable"] == 0 for row in self.bus.settings.values()))

    def test_partial_enable_failure_restores_without_reenabling(self):
        original_write = self.bus.write
        def failing_write(id_, key, value):
            if id_ == 3 and key == "torque_enable" and value == 1:
                raise RuntimeError("injected failed third motor enable")
            original_write(id_, key, value)
        self.bus.write = failing_write
        with self.assertRaises(RuntimeError) as ctx: self.engine.arm()
        self.engine.fault(ctx.exception)
        self.assertFalse(self.engine.armed)
        self.assertTrue(all(row["torque_enable"] == 0 and row["d"] == 32 for row in self.bus.settings.values()))

    def test_near_stop_and_nonfinite_targets_rejected(self):
        self.bus.raw[2] = self.cfg["calibration"]["elbow_flex"]["range_max"] - 7
        with self.assertRaisesRegex(ValueError, "travel margin"): self.engine.arm()
        self.assertFalse(any(e[0] == "write" for e in self.bus.events))
        with self.assertRaises(ValueError): c.vector([0, 0, 0, 0, float("nan"), 0])

    def test_hot_spike_warns_but_recording_continues_and_banner_persists(self):
        self.engine.arm()
        self.step(40)
        self.engine.start_suite()
        self.bus.temperature[4] = 58
        self.step()
        self.assertTrue(self.engine.armed)
        self.assertFalse(self.engine.faulted)
        self.assertEqual(self.engine.mode, "RECORD")
        self.assertIn("wrist_roll (ID 5): 58", self.engine.state()["temperature_warning"])
        self.assertEqual(self.engine.trial["samples"][-1]["temperature_c"][4], 58)
        self.assertTrue(self.engine.trial["commands"])
        self.bus.temperature[4] = 31
        self.step(100)
        self.assertIn("58", self.engine.state()["temperature_warning"])
        self.step(160)
        self.assertIsNone(self.engine.state()["temperature_warning"])
        self.assertEqual(self.engine.mode, "RECORD")
        rows = [json.loads(line) for line in (self.root / "telemetry.jsonl").read_text().splitlines()]
        self.assertTrue(any(r["kind"] == "temperature_warning" for r in rows))

    def test_temperature_warning_threshold_and_no_temperature_abort(self):
        row = self.bus.feedback()
        for value in (49, 50, 58, 85):
            row["temperature_c"][4] = value
            self.assertEqual(bool(s.temperature_warning(row)), value >= 50)
            s.telemetry_ok(row, warn_temperature=False)

    def test_status_and_voltage_still_abort_with_hot_temperature(self):
        row = self.bus.feedback()
        row["temperature_c"][4] = 58
        row["status"][4] = 1
        with self.assertRaisesRegex(RuntimeError, "status alarm"):
            s.telemetry_ok(row, warn_temperature=False)
        row["status"][4] = 0
        row["voltage_v"][4] = 8
        with self.assertRaisesRegex(RuntimeError, "Voltage"):
            s.telemetry_ok(row, warn_temperature=False)

    def test_stale_gui_and_long_sample_gaps_stop_the_worker(self):
        self.engine.arm()
        self.engine.last_heartbeat = self.clock() - 2
        with self.assertRaisesRegex(RuntimeError, "heartbeat"): self.engine.tick()
        self.engine.last_heartbeat = self.clock()
        self.clock.advance(.3)
        with self.assertRaisesRegex(RuntimeError, "stalled"): self.engine.tick()

    def test_move_is_bounded_coordinated_and_cancel_holds_last_goal(self):
        self.engine.arm()
        self.step(40)
        q = s.raw_to_sim(self.engine.target, self.cfg)
        with self.assertRaisesRegex(ValueError, "10 degrees"): self.engine.move(q + math.radians(11))
        self.engine.move(q + math.radians(3))
        self.step(5)
        target = self.engine.target.copy()
        self.engine.cancel_sweep()
        self.step(20)
        np.testing.assert_allclose(self.engine.target, target)
        sent = np.array([e[1] for e in self.bus.events if e[0] == "goals"])
        # Quantization adds at most one encoder count to per-tick changes.
        self.assertLessEqual(np.max(abs(np.diff(sent, axis=0))) * s.RAD_PER_TICK,
                             math.radians(c.SPEED_DEG_S / c.RATE) + s.RAD_PER_TICK)

    def test_complete_suite_has_separate_heldout_log_and_all_axes_excited(self):
        self.engine.arm()
        self.step(45)
        self.engine.start_suite()
        self.step(2450)
        manifest = s.read_json(self.engine.run_dir / "manifest.json")
        self.assertTrue(manifest["complete"])
        self.assertEqual([r["role"] for r in manifest["logs"]], ["train", "train", "validation"])
        self.assertEqual(self.engine.mode, "HOLD")
        for entry in manifest["logs"]:
            log = s.load_trial(self.engine.run_dir / entry["file"], self.cfg)
            goals = np.array([r["raw_target"] for r in log["commands"]])
            self.assertTrue(np.all(np.ptp(goals, axis=0) * s.RAD_PER_TICK > math.radians(1)))
            self.assertTrue(np.all(np.max(abs(goals - self.engine.center), axis=0) * s.RAD_PER_TICK
                                   <= np.deg2rad(c.AMPLITUDES_DEG) + s.RAD_PER_TICK))
            self.assertEqual(log["i"], 0)
            self.assertEqual(log["d"], 0)

    def test_pose_cannot_be_captured_if_arm_moved_after_freeze(self):
        self.step(40)
        self.engine.freeze()
        self.bus.raw[1] += 30
        self.step(40)
        with self.assertRaisesRegex(ValueError, "moved since"): self.engine.capture([0] * 6, "Test")

    def test_fitting_lock_rejects_enable(self):
        self.engine.fit_locked = True
        with self.assertRaises(ValueError): self.engine.arm()
        self.assertEqual(self.bus.events, [])


class PoseMath(unittest.TestCase):
    def test_recovers_known_offsets_without_double_counting_homing(self):
        cfg = config()
        cfg["mapping"]["elbow_flex"]["sign"] = -1
        cfg["calibration"]["elbow_flex"]["homing_offset"] = -900
        expected = np.array([.01, -.02, .03, .04, -.05, .06])
        pairs = []
        for delta in (-150, 0, 170):
            raw = np.array([2048 + delta] * 6)
            q = s.raw_to_sim(raw, cfg) + expected
            pairs.append({"raw_position": raw.tolist(), "reference_q_rad": q.tolist()})
        fitted, report = c.fit_offsets(cfg, pairs)
        np.testing.assert_allclose([fitted["mapping"][j]["offset_rad"] for j in s.JOINTS], expected, atol=1e-12)
        self.assertEqual(fitted["mapping"]["elbow_flex"]["sign"], -1)
        self.assertLess(report["elbow_flex"]["reference_residual_rmse_deg"], 1e-10)

    def test_opposite_direction_is_flagged_not_silently_changed(self):
        cfg = config()
        pairs = []
        for delta in (-150, 0, 170):
            raw = np.array([2048 + delta] * 6)
            q = -s.raw_to_sim(raw, cfg)
            pairs.append({"raw_position": raw.tolist(), "reference_q_rad": q.tolist()})
        fitted, report = c.fit_offsets(cfg, pairs)
        self.assertEqual(fitted["mapping"]["shoulder_lift"]["sign"], 1)
        self.assertFalse(fitted["mapping_reviewed"])
        self.assertTrue(any("opposite sign" in w for w in report["shoulder_lift"]["warnings"]))


class Physics(unittest.TestCase):
    def test_gravity_matches_derivative_of_potential_energy(self):
        import mujoco
        # Analytic pendulum plus five independent hinges. No SO-101 assets needed.
        bodies = ''.join(f'<body pos="{k * 3} 0 0"><joint name="{j}" type="hinge" axis="0 1 0" range="-3 3"/>'
                         '<geom type="sphere" size=".05" pos="1 0 0" mass="2"/></body>' for k, j in enumerate(s.JOINTS))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.xml"
            path.write_text('<mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/><worldbody>' + bodies + '</worldbody></mujoco>')
            mechanics = c.Mechanics(path)
            q = np.array([.1, .2, -.3, .4, -.5, .6])
            torque = mechanics.gravity(q)
            np.testing.assert_allclose(torque, -2 * 9.81 * np.cos(q), atol=1e-10)
            eps = 1e-6
            energy_derivative = []
            for k in range(6):
                energies = []
                for sign in (1, -1):
                    shifted = q.copy(); shifted[k] += sign * eps
                    mechanics.gravity(shifted)
                    mujoco.mj_energyPos(mechanics.model, mechanics.data)
                    energies.append(mechanics.data.energy[0])
                energy_derivative.append((energies[0] - energies[1]) / (2 * eps))
            np.testing.assert_allclose(torque, energy_derivative, rtol=1e-7)

    @unittest.skipUnless(os.environ.get("SO101_TEST_XML"), "Set SO101_TEST_XML for stock mesh snapshot test")
    def test_model_snapshot_preserves_kinematics_and_gravity(self):
        cfg = config()
        xml = Path(os.environ["SO101_TEST_XML"])
        original = c.Mechanics(xml)
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = c.snapshot_model(cfg, xml, Path(tmp))
            frozen = c.Mechanics(Path(tmp) / snapshot["xml"])
            q = [.1, -.3, .4, -.2, .1, .6]
            np.testing.assert_allclose(original.gravity(q), frozen.gravity(q), atol=1e-12)
            np.testing.assert_allclose(original.data.xpos, frozen.data.xpos, atol=1e-12)
            self.assertEqual(c.model_digest(xml), c.model_digest(Path(tmp) / snapshot["xml"]))


if __name__ == "__main__": unittest.main()
