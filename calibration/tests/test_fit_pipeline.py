"""Exercise the optimizer and held-out exports against MuJoCo-generated data."""
import copy
import os
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import so101_sysid as s
import so101_calibration_core as c
import so101_calibration_fit as f
from test_interface import config


@unittest.skipUnless(os.environ.get("SO101_TEST_XML"), "Set SO101_TEST_XML for fitting integration test")
class FitPipeline(unittest.TestCase):
    def test_real_optimizer_saves_candidate_before_heldout_rollouts_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "session"
            cfg = c.snapshot_model(config(), Path(os.environ["SO101_TEST_XML"]), session)
            xml = session / cfg["xml"]
            params = {"schema": 1, "joints": {j: {"bam": copy.deepcopy(s.SEED), "command_delay_s": 0.} for j in s.JOINTS}}
            s.save_json(root / "seed.json", params)
            settings = {j: {key: 0 for key in s.REG} for j in s.JOINTS}
            for row in settings.values(): row.update(p=16, max_torque=1000, torque_limit=1000)
            q0 = np.array([.1, -.2, .2, -.1, .1, .6])
            raw0 = s.sim_to_raw(q0, cfg)
            q0 = s.raw_to_sim(raw0, cfg)
            times = np.arange(0., .301, .02)
            run = session / "runs" / "test"
            run.mkdir(parents=True)
            entries = []
            for index, role in enumerate(("train", "train", "validation")):
                log = {"schema": 1, "complete": True, "synthetic": True, "config": cfg,
                       "joint": "all", "role": role, "p": 16, "i": 0, "d": 0,
                       "effective_settings": settings, "initial_target_raw": raw0.tolist(),
                       "initial_voltage_v": [12.] * 6, "samples": [], "commands": []}
                for t in times:
                    target = q0 + .02 * np.sin((3 + index) * np.pi * t + np.arange(6) * .2)
                    log["commands"].append({"t": float(t), "q_target_rad": target.tolist()})
                    log["samples"].append({"t": float(t), "q_rad": q0.tolist(), "voltage_v": [12.] * 6})
                truth = copy.deepcopy(params)
                truth["joints"]["shoulder_lift"]["bam"]["R"] *= 1.1
                measured = s.rollout(xml, cfg, truth, log)
                for row, q in zip(log["samples"], measured): row["q_rad"] = q.tolist()
                name = f"{index}.json"
                s.save_json(run / name, log)
                entries.append({"file": name, "role": role, "complete": True})
            s.save_json(run / "manifest.json", {"complete": True, "synthetic": True, "logs": entries})
            with self.assertRaisesRegex(ValueError, "Demo data"):
                f.collect_sessions([session])
            out = root / "fit"
            original = s.rollout
            validation_calls = []
            def checked_rollout(xml, cfg, params, log, dt=.002):
                if log["role"] == "validation":
                    self.assertTrue((out / "params.json").exists(), "validation used before candidate was frozen")
                    validation_calls.append(dt)
                return original(xml, cfg, params, log, dt)
            with patch.object(s, "rollout", checked_rollout):
                f.fit_campaign([session], root / "seed.json", out, max_evals=8, allow_synthetic=True)
            self.assertEqual(validation_calls, [.002, .002])
            provenance = s.read_json(out / "params.json")["calibration_interface_fit"]
            self.assertEqual(provenance["training_recording_sha256"], {
                str(run / f"{i}.json"): hashlib.sha256((run / f"{i}.json").read_bytes()).hexdigest()
                for i in (0, 1)})
            report = s.read_json(out / "report.json")
            self.assertTrue(report["synthetic"])
            self.assertFalse(report["sim_to_real_certified"])
            self.assertLessEqual(report["training_final_rmse_deg"], report["training_initial_rmse_deg"] + 1e-12)
            self.assertEqual(len(report["joint_errors"]), 6)
            self.assertTrue((out / "validation-01.png").stat().st_size > 1000)
            self.assertTrue((out / "validation-01.csv").stat().st_size > 1000)
            with self.assertRaisesRegex(ValueError, "training data"):
                f.evaluate_campaign([session], out / "params.json", root / "seed.json", allow_synthetic=True)



class EvaluationProvenance(unittest.TestCase):
    def evaluate(self, root, provenance, recording, *, validation=False):
        cfg = {"mapping": {}, "calibration": {}}
        params = root / "params.json"
        s.save_json(params, {"calibration_interface_fit": {
            "mapping": {}, "calibration": {}, "model_sha256": "model", **provenance}})
        before = params.read_bytes()
        entries = [(str(recording), {})]
        with patch.object(f, "collect_sessions", return_value=(
                cfg, "model.xml", [] if validation else entries, entries if validation else [])), \
             patch.object(f, "model_digest", return_value="model"), \
             patch.object(f, "export_validation", return_value={"joint_errors": {}}) as export:
            try:
                result = f.evaluate_campaign([], params, params, root / "evaluation")
            finally:
                self.assertEqual(params.read_bytes(), before)
            export.assert_called_once()
            return result

    def test_relocated_legacy_session_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = root / "session-1/runs/run-1/train_a.json"
            with self.assertRaisesRegex(ValueError, "training data"):
                self.evaluate(root, {"training_logs": [
                    "/original/archive/session-1/runs/run-1/train_a.json"]}, recording)
            self.assertFalse((root / "evaluation").exists())

    def test_renamed_matching_recording_is_rejected_in_either_split(self):
        for validation in (False, True):
            with self.subTest(validation=validation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                recording = root / "renamed.json"
                recording.write_bytes(b'{"samples": [1, 2, 3]}')
                provenance = {"training_logs": ["/old/session/runs/run/train.json"],
                    "training_recording_sha256": {"/old/session/runs/run/train.json":
                        hashlib.sha256(recording.read_bytes()).hexdigest()}}
                with self.assertRaisesRegex(ValueError, "training data"):
                    self.evaluate(root, provenance, recording, validation=validation)
                self.assertFalse((root / "evaluation").exists())

    def test_separate_session_with_reused_run_and_filename_is_accepted(self):
        for with_hash in (False, True):
            with self.subTest(with_hash=with_hash), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                recording = root / "session-new/runs/run-1/train_a.json"
                recording.parent.mkdir(parents=True)
                recording.write_bytes(b'{"samples": [4, 5, 6]}')
                provenance = {"training_logs": ["/old/session-old/runs/run-1/train_a.json"]}
                if with_hash:
                    provenance["training_recording_sha256"] = {
                        provenance["training_logs"][0]: hashlib.sha256(b"different recording").hexdigest()}
                result = self.evaluate(root, provenance, recording)
                self.assertTrue((result / "report.json").is_file())
