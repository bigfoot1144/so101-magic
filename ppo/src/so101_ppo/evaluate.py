"""Evaluate exported policies in CPU MuJoCo, optionally with a native viewer."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .contract import (
    CONTROL_DT, HOLD_STEPS, SUCCESS_DISTANCE,
    SUCCESS_SPEED, policy_manifest,
)
from .cpu import CpuArm
from .workspace import resolve_workspace, workspace_config, task_bank
from .calibration import Calibration
from .starts import EVAL_START_SEED, home_start, resolve_start_mode, sample_start, start_bank, start_metadata


def load_policy(path, allow_synthetic=False):
    path = Path(path)
    manifest = json.loads((path.parent / "policy_manifest.json").read_text())
    calibration = Calibration(path.parent / manifest["calibration"]["bundle_path"], allow_synthetic)
    if manifest.get("task", "reach") == "pick-place":
        from .pick_place import manifest as pick_manifest
        contract = pick_manifest(calibration)
    elif manifest.get("task", "reach") == "reach":
        region = workspace_config(calibration, resolve_workspace(manifest=manifest))
        contract = policy_manifest(region)
    else:
        raise ValueError("Unknown policy task")
    for key, value in contract.items():
        if manifest.get(key) != value:
            raise ValueError(f"Policy/runtime contract mismatch: {key}")
    if calibration.digest != manifest["calibration"]["bundle_sha256"]:
        raise ValueError("Policy/runtime calibration mismatch")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if (session.get_inputs()[0].shape != contract["observation_shape"]
            or session.get_outputs()[0].shape != contract["action_shape"]):
        raise ValueError("ONNX shape does not match the task policy contract")
    name = session.get_inputs()[0].name

    def policy(obs):
        return session.run(None, {name: np.asarray(obs, np.float32)[None]})[0][0]

    return policy, manifest, calibration


def run_episode(robot, policy, goal, q, on_step=None):
    """Run the shared evaluation/preview episode; callbacks may stop playback."""
    from .pick_place_cpu import CpuPickPlace, run_episode as run_pick_episode
    if isinstance(robot, CpuPickPlace):
        return run_pick_episode(robot, policy, goal, q, on_step)
    obs = robot.reset(goal, q)
    hold = max_hold = 0
    episode_return = 0.0
    distances = []
    for step in range(robot.episode_steps):
        start = time.monotonic()
        obs, reward = robot.step(policy(obs))
        distance = float(np.linalg.norm(robot.goal - robot.tool))
        distances.append(distance)
        episode_return += reward
        stable = distance < SUCCESS_DISTANCE and np.max(np.abs(robot.dq)) < SUCCESS_SPEED
        hold = hold + 1 if stable else 0
        max_hold = max(max_hold, hold)
        failed = bool(robot.tool[2] < 0.06 or np.max(np.abs(robot.dq)) > 20)
        state = {"step": step + 1, "time_s": (step + 1) * CONTROL_DT,
                 "distance_m": distance, "success": bool(max_hold >= HOLD_STEPS and not failed),
                 "failed": failed, "wall_step_start": start, "episode_steps": robot.episode_steps}
        if on_step is not None and on_step(robot, state) is False:
            return None
        if failed:
            break
    return {"success": int(max_hold >= HOLD_STEPS and not failed),
            "failed": int(failed), "steps": step + 1, "return": episode_return,
            "final_distance_m": distances[-1], "mean_distance_m": float(np.mean(distances)),
            "max_hold_s": max_hold * CONTROL_DT,
            "target_x": float(goal[0]), "target_y": float(goal[1]), "target_z": float(goal[2])}


def evaluate(policy_path=None, mode=None, episodes=50, seed=2026, viewer=False,
             baseline="zero", output=None, bank_seed=54321, calibration_path=None, allow_synthetic=False,
             start_mode=None, workspace=None, task=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    manifest = None
    if policy_path:
        policy, manifest, calibration = load_policy(policy_path, allow_synthetic)
        mode = mode or manifest["task_mode"]
    else:
        mode = mode or "fixed"
        if calibration_path is None:
            raise ValueError("A baseline requires --calibration")
        calibration = Calibration(calibration_path, allow_synthetic)
        policy = None
    saved_task = (manifest or {}).get("task", "reach")
    task = task or saved_task
    if manifest is not None and task != saved_task:
        raise ValueError("Policy task cannot be overridden")
    if task == "pick-place":
        from .pick_place import validate_options
        from .pick_place_evaluate import evaluate as evaluate_pick_place
        validate_options(mode, start_mode or "home", workspace)
        return evaluate_pick_place(calibration, policy, episodes, seed, viewer, baseline, output,
                                   policy_path=policy_path)
    if task != "reach":
        raise ValueError("Unknown task")
    selected_workspace = resolve_workspace(workspace, mode, manifest)
    if manifest is not None and selected_workspace != resolve_workspace(manifest=manifest):
        raise ValueError("Policy workspace cannot be overridden: its action mapping is frozen; train a new policy")
    region = workspace_config(calibration, selected_workspace)
    robot = CpuArm(calibration, workspace=selected_workspace)
    start_mode = resolve_start_mode(start_mode, manifest)
    starts = start_bank(calibration, EVAL_START_SEED, selected_workspace) if start_mode == "random" else None
    start_rng = np.random.default_rng(seed)
    # Different bank seed is a held-out set for random-goal evaluation.
    goals, solutions = task_bank(calibration, mode, seed=bank_seed, workspace=selected_workspace)
    rng = np.random.default_rng(seed)
    rows = []
    view = None
    if viewer:
        import mujoco.viewer
        view = mujoco.viewer.launch_passive(robot.model, robot.data)
        view.cam.lookat[:] = [0.20, 0, 0.16]
        view.cam.distance, view.cam.elevation, view.cam.azimuth = 0.75, -25, 135
        if selected_workspace == "wide":
            view.cam.lookat[:] = [0., 0., .22]
            view.cam.distance = 1.25
    try:
        for episode in range(episodes):
            index = int(rng.integers(len(goals)))
            q = home_start(rng, robot.bounds)
            start_index = None
            if starts is not None:
                start_index, q = sample_start(starts, start_rng)
            def act(obs):
                if policy is not None:
                    return policy(obs)
                if baseline == "oracle":
                    return (solutions[index, :3] - robot.center) / robot.scale
                if baseline == "random":
                    return rng.uniform(-1, 1, 3)
                return np.zeros(3)

            def display(robot, state):
                if not view.is_running():
                    return False
                import mujoco
                with view.lock():
                    view.user_scn.ngeom = 1
                    mujoco.mjv_initGeom(view.user_scn.geoms[0],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([SUCCESS_DISTANCE] * 3), robot.goal.astype(float),
                        np.eye(3).ravel(), np.array([0.2, 0.85, 0.3, 0.6]))
                view.sync()
                time.sleep(max(0, CONTROL_DT - (time.monotonic() - state["wall_step_start"])))

            result = run_episode(robot, act, goals[index], q, display if view else None)
            if result is None:
                return {"viewer_closed": True, "completed_episodes": len(rows)}
            rows.append({"episode": episode, "start_mode": start_mode,
                         "start_index": start_index, "initial_q_rad": q.tolist(), **result})
    finally:
        if view is not None:
            view.close()
    result = {
        "backend": "CPU MuJoCo + BAM; imported nominal calibration",
        "synthetic_calibration": calibration.document["synthetic"],
        "calibration_sha256": calibration.digest,
        "hardware_evaluation": False,
        "policy": str(policy_path) if policy_path else baseline,
        "task_mode": mode, **start_metadata(start_mode), **region.metadata(),
        "start_bank_size": len(starts) if starts is not None else None, "episodes": episodes, "seed": seed, "bank_seed": bank_seed,
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
    parser.add_argument("--task", choices=["reach", "pick-place"], help="Inherited from policy; reach for baselines")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--calibration", type=Path, help="Required for baseline evaluation; policies carry their own bundle")
    parser.add_argument("--allow-synthetic", action="store_true", help="Software testing only")
    parser.add_argument("--mode", choices=["fixed", "random"])
    parser.add_argument("--workspace", choices=["near", "wide"],
                        help="Baseline workspace; policies must use their saved workspace")
    parser.add_argument("--start-mode", choices=["home", "random"],
                        help="Override saved start mode; default home for baselines and legacy policies")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--baseline", choices=["zero", "random", "oracle"], default="zero")
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    args = parser.parse_args()
    print(json.dumps(evaluate(policy_path=args.policy, mode=args.mode, episodes=args.episodes,
        seed=args.seed, viewer=args.viewer, baseline=args.baseline, output=args.output,
        calibration_path=args.calibration, allow_synthetic=args.allow_synthetic,
        start_mode=args.start_mode, workspace=args.workspace, task=args.task), indent=2))


if __name__ == "__main__":
    main()
