#!/usr/bin/env python3
"""Export a finished interface fit and CPU reference replay. Never opens a port.

Run with the CALIBRATION Python environment, so the reference uses the same
MuJoCo and BAM that fitted the arm. Source sessions and fitted files are read-only.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import sys
import shutil
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ppo" / "src"))
import so101_sysid as s
from so101_calibration_core import model_digest, snapshot_model
from so101_calibration_fit import collect_sessions
from so101_ppo.calibration import BAM_REVISION, Calibration, read_json, sha256, write_json
from so101_ppo.contract import JOINTS, PHYSICS_DT


def export(sessions, params_path, output, allow_synthetic=False, all_replays=False):
    cfg, xml, training, validation = collect_sessions(sessions, allow_synthetic)
    params_path, output = Path(params_path).resolve(), Path(output).resolve()
    params = read_json(params_path)
    provenance = params.get("calibration_interface_fit", {})
    if not provenance:
        raise ValueError("Expected the finished fit-*/params.json, not seed_params.json")
    synthetic = bool(provenance.get("synthetic") or cfg.get("synthetic") or
                     any(log.get("synthetic") for _, log in training + validation))
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic data cannot represent a physical calibration")
    if not cfg.get("mapping_reviewed") and not synthetic:
        raise ValueError("Use the session recorded with your reviewed coordinate mapping")
    if (provenance.get("mapping") != cfg["mapping"] or
            provenance.get("calibration") != cfg["calibration"] or
            provenance.get("model_sha256") != model_digest(xml)):
        raise ValueError("The fit and session use different mappings or model assets")
    if cfg.get("bam_commit", BAM_REVISION) != BAM_REVISION:
        raise ValueError("Session uses a different BAM revision")
    if tuple(provenance.get(k) for k in ("p", "i", "d")) != (16, 0, 0):
        raise ValueError("Expected a P=16, I=D=0 fit")
    if not cfg.get("free_space", True):
        raise ValueError("This baseline imports the interface's free-space calibration")
    settings = training[0][1]["effective_settings"]
    relevant = ("p", "i", "d", "max_torque", "torque_limit", "acceleration", "max_acceleration", "goal_velocity")
    voltages = []
    for _, log in training + validation:
        if (log["p"], log["i"], log["d"]) != (16, 0, 0) or log.get("command_rate_hz", 50) != 50:
            raise ValueError("Session controller or command rate differs")
        for j in JOINTS:
            if any(log["effective_settings"][j][k] != settings[j][k] for k in relevant):
                raise ValueError(f"{j}: effective motor settings differ across recordings")
            if (settings[j]["p"], settings[j]["i"], settings[j]["d"]) != (16, 0, 0):
                raise ValueError(f"{j}: recorded register gains disagree with the fit")
        voltages.append(log["initial_voltage_v"])
    nominal_voltage = np.median(np.asarray(voltages, float), axis=0)
    if nominal_voltage.shape != (6,) or not np.isfinite(nominal_voltage).all() or np.any(nominal_voltage <= 0):
        raise ValueError("Invalid voltage measurements")
    max_pwm, delays, flat = [], [], []
    for j in JOINTS:
        motor = copy.deepcopy(params["joints"][j]["bam"])
        if motor.get("model") != "m1":
            raise ValueError("This interface fit uses M1; other friction models need a separate validation")
        for key in ("kt", "R", "error_gain_ratio", "armature", "max_velocity", "friction_base", "friction_viscous"):
            value = float(motor[key])
            if not np.isfinite(value) or value < 0 or (key in ("kt", "R", "error_gain_ratio", "max_velocity") and value == 0):
                raise ValueError(f"Invalid {j} parameter {key}")
        # Exactly the calibration simulator's treatment of bench zero and 12 V registration.
        motor.update(actuator="sts321512v", q_offset=0.0)
        ratio = min(settings[j]["max_torque"], settings[j]["torque_limit"]) / 1000
        if not 0 < ratio <= 1:
            raise ValueError(f"Invalid recorded torque cap for {j}")
        max_pwm.append(.97 * ratio)
        delays.append(float(params["joints"][j].get("command_delay_s", 0.)))
        flat.append(motor)
    if not np.isfinite(delays).all() or np.any((np.array(delays) < 0) | (np.array(delays) > .08)):
        raise ValueError("Fitted delay is outside 0..80 ms")
    if output.exists():
        raise FileExistsError(f"Choose a new bundle directory: {output}")
    output.mkdir(parents=True)
    frozen = snapshot_model(cfg, xml, output)
    shutil.copy2(params_path, output / "fitted_params.json")
    files = []
    for j, motor in zip(JOINTS, flat):
        name = f"motors/{j}.json"
        write_json(output / name, motor)
        files.append(name)
    diagnostics = [params_path.parent / name for name in ("REPORT.md", "report.json", "fit_progress.json")]
    diagnostics.extend(sorted(params_path.parent.glob("validation-*.json")))
    for source in diagnostics:
        if source.is_file():
            (output / "source_fit").mkdir(exist_ok=True)
            shutil.copy2(source, output / "source_fit" / source.name)
    replays = []
    selected = validation if all_replays else validation[:1]
    for index, (source, log) in enumerate(selected):
        stem = f"replays/{index:02d}"
        print(f"Exporting reference {index + 1}/{len(selected)}: {Path(source).name}", flush=True)
        prediction = s.rollout(xml, cfg, params, log, PHYSICS_DT)
        write_json(output / f"{stem}.json", log)
        np.savez_compressed(output / f"{stem}.npz", time_s=[r["t"] for r in log["samples"]], q_rad=prediction)
        replays.append({"log": f"{stem}.json", "reference": f"{stem}.npz"})
    document = {
        "schema": 1, "synthetic": synthetic, "hardware_ready": False,
        "joint_order": list(JOINTS), "bam_revision": BAM_REVISION,
        "config_file": "config.json", "model_xml": frozen["xml"], "motor_files": files,
        "physics_dt_s": PHYSICS_DT,
        "controller": {"p": 16, "i": 0, "d": 0, "control_hz": 50,
            "command_slew_deg_s": 8.0, "voltage_v": nominal_voltage.tolist(),
            "voltage_initial_min_v": np.min(voltages, axis=0).tolist(),
            "voltage_initial_max_v": np.max(voltages, axis=0).tolist(),
            "max_pwm": max_pwm, "command_delay_s": delays,
            "effective_settings": settings},
        "source": {"params_sha256": sha256(params_path), "model_digest": model_digest(xml),
            "versions": {k: importlib.metadata.version(k) for k in ("mujoco", "better-actuator-models", "numpy")}},
        "replays": replays,
        "scope": "Recorded free-space P-only response. Torque/current protection transients, contact and hardware velocity observations are not identified.",
    }
    document["files_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "bundle.json", document)
    Calibration(output, allow_synthetic=allow_synthetic)
    print(f"Bundle: {output}\nNo source sessions or motor registers were changed.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", nargs="+", type=Path, required=True, help="Recorded session directories, each containing config.json and runs/")
    parser.add_argument("--params", type=Path, required=True, help="Finished fit-*/params.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all-replays", action="store_true", help="Export every held-out waveform instead of one")
    parser.add_argument("--allow-synthetic", action="store_true", help="Software testing only")
    args = parser.parse_args()
    export(args.session, args.params, args.out, args.allow_synthetic, args.all_replays)


if __name__ == "__main__":
    main()
