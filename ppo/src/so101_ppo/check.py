"""Check physical feasibility and, optionally, the actual mjlab rollout path."""

import argparse
import json

import numpy as np

from .contract import ACTION_SCALE, HOME, JOINTS, OBS_DIM, SUCCESS_DISTANCE
from .cpu import CpuArm
from .calibration import Calibration
from .model import target_bank


def check_cpu(calibration):
    robot = CpuArm(calibration)
    goals, solutions = target_bank("fixed")
    obs = robot.reset(goals[0])
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
    action = (solutions[0, :3] - HOME[:3]) / ACTION_SCALE
    for _ in range(200):
        robot.step(action)
    distance = float(np.linalg.norm(robot.goal - robot.tool))
    return {"cpu_oracle_error_m": distance, "cpu_oracle_max_speed_rad_s":
            float(np.max(np.abs(robot.dq))),
            "known_fk_command_within_tolerance": bool(distance < SUCCESS_DISTANCE),
            "known_fk_command_settled": bool(np.max(np.abs(robot.dq)) < .15),
            "note": "The FK command does not compensate servo tracking error; PPO can learn a different command.",
            "random_targets": len(target_bank("random")[0]),
            "fixed_target_m": goals[0].tolist()}


def check_mjlab(calibration, device="cpu"):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from .task import actor_observation, arm, joint_state, make_env_cfg, tool_position
    from .contract import next_command, observation

    torch.set_num_threads(1)
    cfg = make_env_cfg(calibration, num_envs=2)
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        env.reset()
        q, dq = joint_state(env)
        expected = observation(q.cpu().numpy(), dq.cpu().numpy(),
            env.command_manager.get_command("goal").cpu().numpy(),
            tool_position(env).cpu().numpy(), arm(env).command.cpu().numpy())
        np.testing.assert_allclose(actor_observation(env).cpu().numpy(), expected, atol=1e-6)
        mirrors = [CpuArm(calibration) for _ in range(2)]
        for i, mirror in enumerate(mirrors):
            mirror.reset(env.command_manager.get_command("goal")[i].cpu().numpy(), q[i].cpu().numpy())
        trajectory_error = 0.
        previous = arm(env).command.cpu().numpy().copy()
        actions = torch.tensor([[0.8, -0.6, -0.7], [-0.8, 0.6, 0.7]], device=device)
        for _ in range(8):
            obs, reward, _, _, _ = env.step(actions)
            expected = next_command(actions.cpu().numpy(), previous, calibration.task_bounds())
            np.testing.assert_allclose(arm(env).command.cpu().numpy(), expected, atol=1e-6)
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
            actual_q, _ = joint_state(env)
            for i, mirror in enumerate(mirrors):
                mirror.step(actions[i].cpu().numpy())
                error = float(np.max(np.abs(mirror.q - actual_q[i].cpu().numpy())))
                trajectory_error = max(trajectory_error, error)
                assert error < 5e-5, f"CPU/mjlab policy rollout differs by {error} rad"
            previous = expected
        other_command = arm(env).command[1].clone()
        other_history = [a._bam_model.actuator.q_target_smooth[1].clone()
                         for a in env.scene["robot"].actuators]
        other_delay = [a._history[:, 1].clone() for a in env.scene["robot"].actuators]
        env.reset(env_ids=torch.tensor([0], device=device))
        torch.testing.assert_close(arm(env).command[1], other_command)
        for i, actuator in enumerate(env.scene["robot"].actuators):
            smooth = actuator._bam_model.actuator.q_target_smooth
            torch.testing.assert_close(smooth[1], other_history[i])
            torch.testing.assert_close(actuator._history[:, 1], other_delay[i])
            torch.testing.assert_close(smooth[0], arm(env).quantized_command()[0, i:i+1])
        return {"mjlab_device": device, "mjlab_observation_dim": actor_observation(env).shape[-1],
                "joint_order": list(JOINTS), "partial_reset": "passed",
                "cpu_mjlab_policy_rollout_max_abs_rad": trajectory_error,
                "numpy_torch_observation_and_action_parity": "passed"}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=str, required=True)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--mjlab", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    calibration = Calibration(args.calibration, args.allow_synthetic)
    result = check_cpu(calibration)
    if args.mjlab:
        result.update(check_mjlab(calibration, args.device))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
