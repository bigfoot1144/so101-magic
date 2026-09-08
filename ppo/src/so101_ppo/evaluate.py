"""Evaluate exported policies in CPU MuJoCo, optionally with a native viewer."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .contract import (
    ACTION_SCALE, CONTROL_DT, EPISODE_STEPS, HOLD_STEPS, HOME, SUCCESS_DISTANCE,
    SUCCESS_SPEED, policy_manifest,
)
from .cpu import CpuArm
from .model import target_bank
from .calibration import Calibration


def load_policy(path, allow_synthetic=False):
    path = Path(path)
    manifest = json.loads((path.parent / "policy_manifest.json").read_text())
    for key, value in policy_manifest().items():
        if manifest.get(key) != value:
            raise ValueError(f"Policy/runtime contract mismatch: {key}")
    calibration = Calibration(path.parent / manifest["calibration"]["bundle_path"], allow_synthetic)
    if calibration.digest != manifest["calibration"]["bundle_sha256"]:
        raise ValueError("Policy/runtime calibration mismatch")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 21] or session.get_outputs()[0].shape != [1, 3]:
        raise ValueError("Expected a [1,21] -> [1,3] policy")
    name = session.get_inputs()[0].name

    def policy(obs):
        return session.run(None, {name: np.asarray(obs, np.float32)[None]})[0][0]

    return policy, manifest, calibration


def evaluate(policy_path=None, mode=None, episodes=50, seed=2026, viewer=False,
             baseline="zero", output=None, bank_seed=54321, calibration_path=None, allow_synthetic=False):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if policy_path:
        policy, manifest, calibration = load_policy(policy_path, allow_synthetic)
        mode = mode or manifest["task_mode"]
        robot = CpuArm(calibration)
    else:
        mode = mode or "fixed"
        if calibration_path is None:
            raise ValueError("A baseline requires --calibration")
        calibration = Calibration(calibration_path, allow_synthetic)
        robot = CpuArm(calibration)
        policy = None
    # Different bank seed is a held-out set for random-goal evaluation.
    goals, solutions = target_bank(mode, seed=bank_seed)
    rng = np.random.default_rng(seed)
    rows = []
    view = None
    if viewer:
        import mujoco.viewer
        view = mujoco.viewer.launch_passive(robot.model, robot.data)
        view.cam.lookat[:] = [0.20, 0, 0.16]
        view.cam.distance, view.cam.elevation, view.cam.azimuth = 0.75, -25, 135
    try:
        for episode in range(episodes):
            index = int(rng.integers(len(goals)))
            q = HOME.copy()
            q[:3] += rng.uniform(-0.02, 0.02, 3)
            q[:3] = np.clip(q[:3], *robot.bounds)
            obs = robot.reset(goals[index], q)
            hold, max_hold, episode_return, failed = 0, 0, 0.0, False
            distances = []
            for step in range(EPISODE_STEPS):
                start = time.monotonic()
                if policy is not None:
                    action = policy(obs)
                elif baseline == "oracle":
                    # Known FK solution, only for actuator/task feasibility checks.
                    action = (solutions[index, :3] - HOME[:3]) / ACTION_SCALE
                elif baseline == "random":
                    action = rng.uniform(-1, 1, 3)
                else:
                    action = np.zeros(3)
                obs, reward = robot.step(action)
                distance = float(np.linalg.norm(robot.goal - robot.tool))
                distances.append(distance)
                episode_return += reward
                stable = distance < SUCCESS_DISTANCE and np.max(np.abs(robot.dq)) < SUCCESS_SPEED
                hold = hold + 1 if stable else 0
                max_hold = max(max_hold, hold)
                failed = robot.tool[2] < 0.06 or np.max(np.abs(robot.dq)) > 20
                if view is not None:
                    if not view.is_running():
                        return {"viewer_closed": True, "completed_episodes": len(rows)}
                    import mujoco
                    with view.lock():
                        view.user_scn.ngeom = 1
                        mujoco.mjv_initGeom(view.user_scn.geoms[0],
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([SUCCESS_DISTANCE] * 3), robot.goal.astype(float),
                            np.eye(3).ravel(), np.array([0.2, 0.85, 0.3, 0.6]))
                    view.sync()
                    time.sleep(max(0, CONTROL_DT - (time.monotonic() - start)))
                if failed:
                    break
            rows.append({
                "episode": episode, "success": int(max_hold >= HOLD_STEPS and not failed),
                "failed": int(failed), "steps": step + 1, "return": episode_return,
                "final_distance_m": distances[-1], "mean_distance_m": float(np.mean(distances)),
                "max_hold_s": max_hold * CONTROL_DT,
                "target_x": float(robot.goal[0]), "target_y": float(robot.goal[1]),
                "target_z": float(robot.goal[2]),
            })
    finally:
        if view is not None:
            view.close()
    result = {
        "backend": "CPU MuJoCo + BAM; imported nominal calibration",
        "synthetic_calibration": calibration.document["synthetic"],
        "calibration_sha256": calibration.digest,
        "hardware_evaluation": False,
        "policy": str(policy_path) if policy_path else baseline,
        "task_mode": mode, "episodes": episodes, "seed": seed, "bank_seed": bank_seed,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "failure_rate": float(np.mean([r["failed"] for r in rows])),
        "mean_final_distance_m": float(np.mean([r["final_distance_m"] for r in rows])),
        "p95_final_distance_m": float(np.quantile([r["final_distance_m"] for r in rows], 0.95)),
        "mean_return": float(np.mean([r["return"] for r in rows])),
    }
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        with output.with_suffix(".csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--calibration", type=Path, help="Required for baseline evaluation; policies carry their own bundle")
    parser.add_argument("--allow-synthetic", action="store_true", help="Software testing only")
    parser.add_argument("--mode", choices=["fixed", "random"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--baseline", choices=["zero", "random", "oracle"], default="zero")
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    args = parser.parse_args()
    print(json.dumps(evaluate(policy_path=args.policy, mode=args.mode, episodes=args.episodes,
        seed=args.seed, viewer=args.viewer, baseline=args.baseline, output=args.output,
        calibration_path=args.calibration, allow_synthetic=args.allow_synthetic), indent=2))


if __name__ == "__main__":
    main()
