"""Compare CPU and mjlab predictions with the exported calibration reference."""
import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import Calibration, read_json, write_json
from .contract import JOINTS, PHYSICS_DT
from .cpu import CpuArm


def targets_at(t, log, calibration):
    origin = log["samples"][0]["t"]
    times = np.asarray([r["t"] for r in log["commands"]]) - origin
    values = np.asarray([r["q_target_rad"] for r in log["commands"]])
    index = np.searchsorted(times, t - calibration.delays, side="right") - 1
    target = calibration.raw_to_sim(log["initial_target_raw"])
    valid = index >= 0
    target[valid] = values[index[valid], np.arange(6)[valid]]
    return target


def interpolate(times, positions, log):
    samples = np.asarray([r["t"] for r in log["samples"]])
    samples -= samples[0]
    q = np.asarray(positions)
    return np.column_stack([np.interp(samples, times, q[:, i]) for i in range(6)])


def cpu_replay(calibration, log):
    robot = CpuArm(calibration, voltage=log["initial_voltage_v"], check_task=False)
    robot.reset(np.zeros(3), q=log["samples"][0]["q_rad"],
                dq=log.get("initial_velocity_rad_s", [0.] * 6),
                initial_target=calibration.raw_to_sim(log["initial_target_raw"]))
    times, positions = [0.], [robot.q]
    end = log["samples"][-1]["t"] - log["samples"][0]["t"]
    while robot.data.time < end:
        q = robot.step_targets(targets_at(robot.data.time, log, calibration), apply_delay=False)
        times.append(float(robot.data.time))
        positions.append(q)
    return interpolate(times, positions, log)


def mjlab_replay(calibration, log, device):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from .task import make_env_cfg

    torch.set_num_threads(1)
    env = ManagerBasedRlEnv(make_env_cfg(calibration, num_envs=1, replay=True,
                             voltage=log["initial_voltage_v"]), device=device)
    try:
        env.reset()
        robot = env.scene["robot"]
        ids, _ = robot.find_joints(JOINTS, preserve_order=True)
        q = torch.tensor([log["samples"][0]["q_rad"]], dtype=torch.float32, device=device)
        dq = torch.tensor([log.get("initial_velocity_rad_s", [0.] * 6)], dtype=torch.float32, device=device)
        robot.write_joint_state_to_sim(q, dq, joint_ids=ids)
        initial = torch.tensor(calibration.raw_to_sim(log["initial_target_raw"])[None], dtype=torch.float32, device=device)
        robot.set_joint_position_target(initial, joint_ids=ids)
        for i, actuator in enumerate(robot.actuators):
            actuator.reset_target_history(slice(None), initial[:, i:i+1])
        env.sim.forward()
        # Forwarding does not integrate. A scene update here would commit the
        # stale actuator state computed at env.reset(), replacing the recorded
        # initial target history with the task home pose.
        times, positions = [0.], [q[0].cpu().numpy().copy()]
        end = log["samples"][-1]["t"] - log["samples"][0]["t"]
        t = 0.
        while t < end:
            target = torch.tensor(targets_at(t, log, calibration)[None], dtype=torch.float32, device=device)
            robot.set_joint_position_target(target, joint_ids=ids)
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(PHYSICS_DT)
            t += PHYSICS_DT
            times.append(t)
            positions.append(robot.data.joint_pos[0, ids].cpu().numpy().copy())
        return interpolate(times, positions, log)
    finally:
        env.close()


def errors(a, b):
    delta = np.rad2deg(np.asarray(a) - np.asarray(b))
    return {j: {"rmse": float(np.sqrt(np.mean(delta[:, i] ** 2))),
                "p95_abs": float(np.percentile(abs(delta[:, i]), 95)),
                "max_abs": float(np.max(abs(delta[:, i])))} for i, j in enumerate(JOINTS)}


def replay(calibration, device=None, output=None, rmse_limit=.05, max_limit=.25):
    rows = []
    for entry in calibration.document["replays"]:
        log = read_json(calibration.path(entry["log"]))
        with np.load(calibration.path(entry["reference"]), allow_pickle=False) as archive:
            reference = archive["q_rad"]
        measured = np.array([r["q_rad"] for r in log["samples"]])
        cpu = cpu_replay(calibration, log)
        row = {"log": entry["log"], "cpu_vs_reference_deg": errors(cpu, reference),
               "cpu_vs_measured_deg": errors(cpu, measured)}
        comparisons = [row["cpu_vs_reference_deg"]]
        if device:
            gpu = mjlab_replay(calibration, log, device)
            row["mjlab_vs_reference_deg"] = errors(gpu, reference)
            row["mjlab_vs_measured_deg"] = errors(gpu, measured)
            comparisons.append(row["mjlab_vs_reference_deg"])
        row["implementation_parity_passed"] = all(
            m["rmse"] <= rmse_limit and m["max_abs"] <= max_limit
            for comparison in comparisons for m in comparison.values())
        rows.append(row)
    if not rows:
        raise ValueError("Bundle has no reference recordings")
    result = {"calibration_sha256": calibration.digest, "synthetic": calibration.document["synthetic"],
              "scope": "Simulator implementation parity, not physical task accuracy",
              "rmse_limit_deg": rmse_limit, "max_limit_deg": max_limit,
              "mjlab_device": device, "passed": all(r["implementation_parity_passed"] for r in rows), "trials": rows}
    if output:
        write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--mjlab", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("replay_report.json"))
    args = parser.parse_args()
    result = replay(Calibration(args.calibration, args.allow_synthetic),
                    device=args.device if args.mjlab else None, output=args.output)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Replay drift exceeded the software-parity tolerances; inspect the report before trusting transfer.")


if __name__ == "__main__":
    main()
