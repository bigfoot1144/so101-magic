"""Train PPO, save checkpoints, export ONNX, and evaluate in CPU MuJoCo."""

import argparse
import importlib.metadata
import json
import math
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from .contract import policy_manifest
from .model import BAM_REVISION, ROBOT_REVISION, validate_geometry
from .calibration import Calibration
from .task import make_env_cfg, runner_cfg
from .starts import resolve_start_mode, start_metadata
from .workspace import resolve_workspace, workspace_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["reach", "pick-place"], help="Default reach; inherited on resume")
    parser.add_argument("--mode", choices=["fixed", "random"],
                        help="Layout mode: fixed for new runs; random currently supports reaching only")
    parser.add_argument("--workspace", choices=["near", "wide"],
                        help="Default wide for new random tasks, near for fixed; inherited on resume")
    parser.add_argument("--start-mode", choices=["home", "random"],
                        help="Episode starts: default home for new runs; inherit on resume")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--calibration", type=Path, required=True, help="Directory exported by calibration/export_for_ppo.py")
    parser.add_argument("--allow-synthetic", action="store_true", help="Software testing only")
    parser.add_argument("--robust", action="store_true",
                        help="Optional heuristic voltage +/-5%%, gain/friction +/-10%%; retain fitted delays")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--visualize", choices=["off", "show", "save", "both"], default="off",
                        help="Periodically preview the current policy in a separate simulation")
    parser.add_argument("--visualize-every", type=int, default=25,
                        help="Preview every N completed PPO rollout/update iterations (default: 25)")
    parser.add_argument("--visualize-targets", choices=["auto", "fixed", "rotate"], default="auto",
                        help="Preview targets: auto rotates in random mode; fixed repeats one target")
    parser.add_argument("--timelapse-speed", type=float, default=4.0,
                        help="Saved compilation playback multiplier; episode clips stay real time")
    args = parser.parse_args()
    if args.visualize_every < 1:
        parser.error("visualize-every must be positive")
    if not math.isfinite(args.timelapse_speed) or args.timelapse_speed <= 0:
        parser.error("timelapse-speed must be finite and positive")
    if args.num_envs < 1 or args.iterations < 1:
        parser.error("Environment count and iterations must be positive")
    if args.eval_episodes < 0:
        parser.error("eval-episodes must be nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use an NVIDIA GPU for training, or run a slow "
                     "smoke test with --device cpu --num-envs 4 --iterations 2")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    calibration = Calibration(args.calibration, allow_synthetic=args.allow_synthetic)
    validate_geometry(calibration)
    calibration.task_bounds()
    old = None
    if args.resume:
        old = json.loads((args.resume.resolve().parent / "policy_manifest.json").read_text())
        if old.get("calibration", {}).get("bundle_sha256") != calibration.digest:
            raise ValueError("Resume requires the same calibration bundle; retrain when motor fits change")
    args.task = args.task or (old or {}).get("task", "reach")
    if old is not None and args.task != old.get("task", "reach"):
        raise ValueError("Cannot resume a checkpoint from a different task")
    args.mode = args.mode or (old or {}).get("task_mode", "fixed")
    if args.task == "pick-place":
        from .pick_place import validate_options
        validate_options(args.mode, args.start_mode or "home", args.workspace, args.robust, args.visualize_targets)
    args.workspace = resolve_workspace(args.workspace, args.mode, old)
    region = workspace_config(calibration, args.workspace)
    if args.task == "pick-place":
        from .pick_place import manifest as pick_manifest
        expected_contract = pick_manifest(calibration)
    else:
        expected_contract = policy_manifest(region)
    if old is not None and any(old.get(k) != v for k, v in expected_contract.items()):
        raise ValueError("Checkpoint has an incompatible task, policy/control contract, or workspace; start a new run")
    args.start_mode = resolve_start_mode(args.start_mode, old)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = (args.run or Path("runs") / f"{args.task}_{args.mode}_{timestamp}").resolve()
    run.mkdir(parents=True, exist_ok=False)
    shutil.copytree(calibration.root, run / "calibration")
    calibration = Calibration(run / "calibration", allow_synthetic=args.allow_synthetic)
    print("Using calibration bundle: P=16, 50 Hz control, 8 deg/s command slew, per-joint fits/caps/delays.")
    if calibration.document["synthetic"]:
        print("SYNTHETIC software test. These parameters are not your physical arm's calibration.")
    manifest = expected_contract.copy()
    manifest.update({
        "task": args.task, "task_mode": args.mode, "seed": args.seed, **start_metadata(args.start_mode),
        **(region.metadata() if args.task == "reach" else {"episode_seconds": expected_contract["pick_place"]["episode_seconds"]}),
        "robot_revision": ROBOT_REVISION, "bam_revision": BAM_REVISION,
        "calibration": {"bundle_path": "calibration", "bundle_sha256": calibration.digest,
            "synthetic": calibration.document["synthetic"]},
        "actuator": {"status": "synthetic_test" if calibration.document["synthetic"] else "imported_calibration",
            "kp_fw": 16, "voltage_v": calibration.voltage.tolist(),
            "max_pwm": calibration.max_pwm.tolist(), "delay_s": calibration.delays.tolist(),
            "delay_steps": calibration.lags.tolist(), "stiff_frictionloss": False,
            "robust_training": args.robust},
        "command_bounds_rad": (expected_contract["command_bounds_rad"] if args.task == "pick-place"
                               else [region.lower.tolist(), region.upper.tolist()]),
        "versions": {name: importlib.metadata.version(name) for name in
            ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib",
             "better-actuator-models", "onnxruntime")},
    })
    (run / "policy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "run.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    if args.task == "pick-place":
        from .pick_place_task import make_env_cfg as make_pick_env_cfg
        cfg = make_pick_env_cfg(calibration, num_envs=args.num_envs, seed=args.seed)
    else:
        cfg = make_env_cfg(calibration, mode=args.mode, num_envs=args.num_envs, seed=args.seed, robust=args.robust,
                           start_mode=args.start_mode, workspace=args.workspace)
    print(f"Workspace: {args.workspace}; starts: {args.start_mode}; episode: {cfg.episode_length_s:g} s")
    agent = runner_cfg()
    if args.task == "pick-place":
        agent.experiment_name = "so101_pick_place"
    agent.seed = args.seed
    agent.max_iterations = args.iterations
    (run / "agent.json").write_text(json.dumps(asdict(agent), indent=2) + "\n")
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device))
    visualization = None
    try:
        runner = MjlabOnPolicyRunner(env, asdict(agent), str(run), args.device)
        if args.resume:
            runner.load(str(args.resume.resolve()), map_location=args.device)
        if args.task == "pick-place":
            from .training_logging import RolloutLogger
            runner.logger = RolloutLogger(runner.logger)
        if args.visualize != "off":
            from .visualization import PreviewLogger, TrainingVisualization
            visualization = TrainingVisualization(calibration, run, args.visualize,
                every=args.visualize_every, speed=args.timelapse_speed,
                task_mode=args.mode, source_checkpoint=args.resume, target_selection=args.visualize_targets,
                start_mode=args.start_mode, workspace=args.workspace, task=args.task)
            visualization.capture(runner)
            runner.logger = PreviewLogger(runner.logger, lambda: visualization.iteration(runner))
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        if visualization is not None:
            visualization.capture(runner)
        runner.save(str(run / "checkpoint.pt"))
        runner.export_policy_to_onnx(str(run), "policy.onnx")
        # Compare raw-observation inference through both paths, including normalization.
        runner.get_inference_policy(device=args.device)
        exported = runner.alg.get_policy().as_onnx(verbose=False).to("cpu").eval()
        session = ort.InferenceSession(str(run / "policy.onnx"), providers=["CPUExecutionProvider"])
        obs = env.get_observations()["actor"].detach().cpu().numpy()
        errors = []
        for row in obs[:32]:
            with torch.inference_mode():
                expected = exported(torch.from_numpy(row[None])).numpy()
            actual = session.run(None, {session.get_inputs()[0].name: row[None]})[0]
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
            errors.append(float(np.max(np.abs(actual - expected))))
        (run / "export_check.json").write_text(json.dumps({"max_abs_error": max(errors)}, indent=2))
    finally:
        try:
            if visualization is not None:
                visualization.close()
        finally:
            env.close()
    if args.eval_episodes:
        from .evaluate import evaluate
        result = evaluate(run / "policy.onnx", episodes=args.eval_episodes,
                          output=run / "evaluation.json", allow_synthetic=args.allow_synthetic)
        print(json.dumps(result, indent=2))
    print(f"Run: {run}\nPolicy: {run / 'policy.onnx'}\nCheckpoint: {run / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
