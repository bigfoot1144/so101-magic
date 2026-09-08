"""Offline coordinated-motion fit. This module never opens a serial port."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import so101_sysid as s
from so101_calibration_core import atomic_json, model_digest, stamp


def recording_sha256(path):
    """Hash the original recording bytes without modifying the source."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recording_id(path):
    """Portable legacy identity: session/runs/run/filename, excluding parents."""
    parts = Path(path).parts
    if len(parts) >= 4 and parts[-3] == "runs":
        return parts[-4:]
    return None


def includes_training_recording(provenance, entries):
    paths = set(provenance.get("training_logs", []))
    identities = {recording_id(path) for path in paths} - {None}
    hashes = set(provenance.get("training_recording_sha256", {}).values())
    for path, _ in entries:
        if path in paths or recording_id(path) in identities:
            return True
        if hashes and recording_sha256(path) in hashes:
            return True
    return False


def collect_sessions(paths, allow_synthetic=False):
    cfg = xml = fingerprint = None
    training, validation = [], []
    for path in paths:
        root = Path(path).resolve()
        this_cfg, this_xml = s.load_config(root / "config.json")
        this_digest = model_digest(this_xml)
        if cfg is None:
            cfg, xml, fingerprint = this_cfg, this_xml, this_digest
        if (cfg["calibration"] != this_cfg["calibration"] or cfg["mapping"] != this_cfg["mapping"]
                or cfg.get("free_space", True) != this_cfg.get("free_space", True) or fingerprint != this_digest):
            raise ValueError("Sessions have different mappings, geometry, or physics. Fit them separately.")
        for manifest_path in sorted((root / "runs").glob("*/manifest.json")):
            manifest = s.read_json(manifest_path)
            if not manifest.get("complete"):
                print(f"Skipping incomplete run: {manifest_path.parent}", flush=True)
                continue
            if manifest.get("synthetic") and not allow_synthetic:
                raise ValueError("Demo data cannot identify real motors. --allow-synthetic is for software checks only.")
            for item in manifest["logs"]:
                log_path = manifest_path.parent / item["file"]
                log = s.load_trial(log_path, cfg)
                if log.get("synthetic") and not allow_synthetic:
                    raise ValueError("Synthetic log in hardware fit")
                if item["role"] != log.get("role"):
                    raise ValueError("Manifest/log role differs")
                entry = (str(log_path), log)
                if item["role"] == "train": training.append(entry)
                elif item["role"] == "validation": validation.append(entry)
                else: raise ValueError("Unexpected split role")
    if not training or not validation:
        raise ValueError("Need at least one complete coordinated run with training and held-out logs.")
    return cfg, xml, training, validation


def score(xml, cfg, params, logs, dt):
    squared = []
    for _, log in logs:
        prediction = s.rollout(xml, cfg, params, log, dt)
        actual = np.array([r["q_rad"] for r in log["samples"]])
        squared.append(np.mean(np.rad2deg(prediction - actual)**2))
    # Equal trial weighting prevents a slower serial trial from dominating.
    return float(np.sqrt(np.mean(squared)))


def export_validation(root, xml, cfg, baseline, candidate, entries, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    all_actual, all_base, all_candidate = [], [], []
    trials = []
    for index, (path, log) in enumerate(entries, 1):
        actual = np.array([r["q_rad"] for r in log["samples"]])
        predicted0 = s.rollout(xml, cfg, baseline, log, dt)
        predicted1 = s.rollout(xml, cfg, candidate, log, dt)
        all_actual.append(actual)
        all_base.append(predicted0)
        all_candidate.append(predicted1)
        times = np.array([r["t"] for r in log["samples"]])
        trial = {"log": path, "baseline": s.metrics(log, predicted0), "fitted": s.metrics(log, predicted1)}
        trials.append(trial)
        output = root / f"validation-{index:02d}"
        atomic_json(output.with_suffix(".json"), trial)
        with output.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s"] + [f"{j}_{kind}_deg" for kind in ("measured", "baseline", "fitted") for j in s.JOINTS])
            writer.writerows(np.column_stack([times, np.rad2deg(actual), np.rad2deg(predicted0), np.rad2deg(predicted1)]))
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        ct = [r["t"] for r in log["commands"]]
        cq = np.rad2deg([r["q_target_rad"] for r in log["commands"]])
        for k, ax in enumerate(axes.flat):
            ax.step(ct, cq[:, k], where="post", color=".7", lw=.9, label="Sent target")
            ax.plot(times, np.rad2deg(actual[:, k]), lw=1.8, label="Encoder measurement")
            ax.plot(times, np.rad2deg(predicted0[:, k]), lw=1., label="Initial model")
            ax.plot(times, np.rad2deg(predicted1[:, k]), lw=1.3, label="Fitted model")
            ax.set(title=s.JOINTS[k], ylabel="Angle (deg)")
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]: ax.set_xlabel("Time (s)")
        fig.suptitle("Held-out motion: free rollout initialized once; no encoder state resets")
        fig.tight_layout()
        fig.savefig(output.with_suffix(".png"), dpi=140)
        plt.close(fig)
    measured = np.concatenate(all_actual)
    errors0 = np.rad2deg(np.concatenate(all_base) - measured)
    errors1 = np.rad2deg(np.concatenate(all_candidate) - measured)
    by_joint = {}
    for k, j in enumerate(s.JOINTS):
        by_joint[j] = {"baseline_rmse_deg": float(np.sqrt(np.mean(errors0[:, k]**2))),
                       "fitted_rmse_deg": float(np.sqrt(np.mean(errors1[:, k]**2))),
                       "fitted_p95_abs_deg": float(np.percentile(abs(errors1[:, k]), 95)),
                       "fitted_max_abs_deg": float(np.max(abs(errors1[:, k])))}
    return {"trials": trials, "joint_errors": by_joint,
            "baseline_overall_rmse_deg": float(np.sqrt(np.mean(errors0**2))),
            "fitted_overall_rmse_deg": float(np.sqrt(np.mean(errors1**2)))}


def fit_campaign(sessions, params_path, output=None, max_evals=48, passes=1, dt=.004,
                 allow_synthetic=False):
    """Bounded coordinate descent over 4 parameters per joint, full-arm loss.

    The finite evaluation budget produces an effective local fit, not unique
    physical constants. Validation is read ONLY after freezing the candidate.
    """
    from scipy.optimize import minimize
    if not 8 <= max_evals <= 1000 or not 1 <= passes <= 5 or not .001 <= dt <= .004:
        raise ValueError("Fit envelope: 8..1000 evaluations per joint, 1..5 passes, dt .001..004.")
    cfg, xml, training, validation = collect_sessions(sessions, allow_synthetic)
    baseline = s.read_json(params_path)
    candidate = copy.deepcopy(baseline)
    root = Path(output) if output else Path(sessions[0]) / ("fit-" + stamp())
    root.mkdir(parents=True, exist_ok=False)
    print(f"Output: {root.resolve()}", flush=True)
    print(f"Training: {len(training)} logs. Held out: {len(validation)} logs. No hardware access.", flush=True)
    names = ["R", "friction_base", "friction_viscous", "command_delay_s"]
    lower, upper = np.array([.5, 0., 0., 0.]), np.array([8., .6, .3, .08])
    history = []
    baseline_score = score(xml, cfg, baseline, training, dt)
    if not math.isfinite(baseline_score):
        raise ValueError("Initial model has a non-finite training error; inspect the model and data.")
    best_score = baseline_score
    print(f"Initial training RMSE: {baseline_score:.4f} deg", flush=True)
    for pass_index in range(passes):
        for joint in s.JOINTS:
            base = copy.deepcopy(candidate)
            values = base["joints"][joint]
            x0 = np.array([values["bam"][n] for n in names[:3]] + [values.get(names[3], 0.)])
            if np.any(x0 < lower) or np.any(x0 > upper):
                raise ValueError(f"{joint}: initial parameters outside fitting bounds")
            best = {"score": best_score, "params": base, "x": x0.copy(), "evaluations": 0}

            def objective(z):
                x = lower + np.asarray(z) * (upper - lower)
                trial_params = copy.deepcopy(base)
                for n, value in zip(names[:3], x[:3]): trial_params["joints"][joint]["bam"][n] = float(value)
                trial_params["joints"][joint][names[3]] = float(x[3])
                best["evaluations"] += 1
                try:
                    value = score(xml, cfg, trial_params, training, dt)
                    if not math.isfinite(value): value = 1e6
                except (RuntimeError, ValueError, FloatingPointError):
                    value = 1e6
                if value < best["score"]:
                    best.update(score=value, params=trial_params, x=x.copy())
                    atomic_json(root / "training_checkpoint.json", trial_params)
                if best["evaluations"] == 1 or best["evaluations"] % 8 == 0:
                    print(f"Pass {pass_index + 1}, {joint}, evaluation {best['evaluations']}/{max_evals}: "
                          f"best full-arm training RMSE {best['score']:.4f} deg", flush=True)
                return value

            result = minimize(objective, (x0 - lower) / (upper - lower), method="Powell",
                              bounds=[(0., 1.)] * 4,
                              options={"maxfev": max_evals, "xtol": .02, "ftol": .005})
            candidate, best_score = best["params"], best["score"]
            normalized = (best["x"] - lower) / (upper - lower)
            history.append({"pass": pass_index + 1, "joint": joint,
                            "training_rmse_deg": best_score, "evaluations": best["evaluations"],
                            "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
                            "effective_parameters": dict(zip(names, best["x"].tolist())),
                            "near_bound": [n for n, z in zip(names, normalized) if z < .02 or z > .98]})
            atomic_json(root / "fit_progress.json", history)
    candidate["calibration_interface_fit"] = {"training_logs": [p for p, _ in training],
                    "training_recording_sha256": {p: recording_sha256(p) for p, _ in training},
                    "model_sha256": model_digest(xml), "training_dt_s": dt, "p": 16, "i": 0, "d": 0,
                    "mapping": cfg["mapping"], "calibration": cfg["calibration"],
                    "parameter_interpretation": "effective local closed-loop fit; Kt, mass, inertia and armature fixed",
                    "synthetic": any(log.get("synthetic") for _, log in training), "history": history}
    # Freeze before evaluating any held-out prediction. Never auto-install it.
    atomic_json(root / "params.json", candidate)
    print("Candidate frozen. Evaluating unseen motions at the deployment timestep of 0.002 s...", flush=True)
    report = export_validation(root, xml, cfg, baseline, candidate, validation, .002)
    report.update({"training_initial_rmse_deg": baseline_score, "training_final_rmse_deg": best_score,
                   "validation_dt_s": .002, "history": history,
                   "synthetic": candidate["calibration_interface_fit"]["synthetic"],
                   "scope": "local free-space position response, P=16 I=D=0, recorded payload and voltage",
                   "sim_to_real_certified": False,
                   "unidentified": ["true torque constant", "mass/inertia", "load-dependent/directional friction",
                                    "backlash/compliance", "I/D controller dynamics", "contact and grip forces",
                                    "camera calibration and observations"]})
    centers = np.array([log["initial_target_raw"] for _, log in training])
    report["training_center_span_deg"] = dict(zip(s.JOINTS, (np.ptp(centers, axis=0) * s.RAD_PER_TICK * 180 / np.pi).tolist()))
    atomic_json(root / "report.json", report)
    lines = ["# Calibration fit report", "", f"Scope: {report['scope']}.", "",
             "Torques remain model estimates. This report does not certify sim-to-real accuracy.", "",
             f"Training RMSE: {baseline_score:.3f} -> {best_score:.3f} degrees.",
             f"Held-out RMSE: {report['baseline_overall_rmse_deg']:.3f} -> {report['fitted_overall_rmse_deg']:.3f} degrees.", "",
             "| Joint | Initial RMSE (deg) | Fitted RMSE (deg) | Fitted p95 (deg) | Fitted max (deg) |", "|---|---:|---:|---:|---:|"]
    for joint, m in report["joint_errors"].items():
        lines.append(f"| {joint} | {m['baseline_rmse_deg']:.3f} | {m['fitted_rmse_deg']:.3f} | {m['fitted_p95_abs_deg']:.3f} | {m['fitted_max_abs_deg']:.3f} |")
    lines.extend(["", "The split holds out a different waveform at each captured center. It does not hold out an entire pose or payload.",
                  "Validate a new session at different poses and speeds before using these parameters for deployment.",
                  "If held-out error worsens, keep the initial parameters and diagnose the model/data; do not tune on that validation trace.",
                  "Check fit_progress.json for optimizer budget exhaustion and parameter-bound warnings.",
                  "Same-pose visual matching is not an independent end-effector accuracy test.", ""])
    if report["synthetic"]: lines.insert(2, "**SYNTHETIC DEMO DATA: these parameters must not be used for a physical robot.**\n")
    (root / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(report["joint_errors"], indent=2), flush=True)
    print(f"Finished: {root.resolve() / 'REPORT.md'}", flush=True)
    return root


def evaluate_campaign(sessions, params_path, baseline_path, output=None, allow_synthetic=False):
    """Evaluate every motion from NEW sessions, without fitting anything."""
    cfg, xml, training, validation = collect_sessions(sessions, allow_synthetic)
    params, baseline = s.read_json(params_path), s.read_json(baseline_path)
    provenance = params.get("calibration_interface_fit", {})
    if provenance:
        if (provenance.get("mapping") != cfg["mapping"] or provenance.get("calibration") != cfg["calibration"]
                or provenance["model_sha256"] != model_digest(xml)):
            raise ValueError("Fitted parameters belong to a different mapping or model.")
        if includes_training_recording(provenance, training + validation):
            raise ValueError("This evaluation includes training data. Record a NEW session for an independent pose test.")
    if provenance.get("synthetic") and not allow_synthetic:
        raise ValueError("Synthetic parameters cannot validate real hardware")
    root = Path(output) if output else Path(sessions[0]) / ("evaluation-" + stamp())
    root.mkdir(parents=True, exist_ok=False)
    report = export_validation(root, xml, cfg, baseline, params, training + validation, .002)
    report.update({"kind": "new-session evaluation; no fitting", "sim_to_real_certified": False})
    atomic_json(root / "report.json", report)
    print(json.dumps(report["joint_errors"], indent=2), flush=True)
    print(f"Evaluation saved to {root.resolve()}", flush=True)
    return root
