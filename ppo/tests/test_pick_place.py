"""Behavioral checks for manipulation milestones, physics, and policy compatibility."""
import json

import mujoco
import numpy as np
import pytest
import torch

from so101_ppo.contract import CONTROL_DT
from so101_ppo.pick_place import CONFIG, TaskState, action_bounds, manifest, next_command, validate_options
from so101_ppo.pick_place_cpu import CpuPickPlace


def transition(state, *, pos=None, quat=(1., 0., 0., 0.), contacts=(False, False, False, False), velocity=None):
    count = len(state.reward)
    def batch(value):
        return torch.tensor(value, dtype=torch.float32).repeat(count, 1)
    pos = CONFIG.brick_position if pos is None else pos
    return state.advance(batch(CONFIG.start_q), batch([0.]*6), batch(pos), batch(quat),
        batch([0.]*6 if velocity is None else velocity), batch([.2, 0., .04]),
        batch(CONFIG.target_position), torch.tensor(contacts).repeat(count, 1), batch([0.]*6))


def pickup(state):
    for _ in range(round(CONFIG.lift_hold_seconds/CONTROL_DT)):
        transition(state, pos=(.22, 0., .03), contacts=(True, True, True, False))


def test_touch_and_pickup_are_once_per_episode():
    state = TaskState(1)
    transition(state, contacts=(False, False, True, True))
    assert not state.milestones.any()  # Arm contact is not fingertip contact.
    transition(state, contacts=(True, False, True, True))
    assert state.reward.item() > .9
    transition(state, contacts=(True, False, True, True))
    assert state.reward.item() < .1
    pickup(state)
    assert state.milestones.tolist() == [[True, True, False]]
    assert state.reward.item() > 9.
    transition(state)  # Drop and regrasp must not pay the pickup bonus again.
    pickup(state)
    assert state.reward.item() < .1


def test_pickup_requires_both_fingers_clearance_and_continuous_hold():
    state = TaskState(1)
    for _ in range(20):
        transition(state, pos=(.22, 0., .04), contacts=(True, False, True, False))
    assert not state.milestones[0, 1]
    for _ in range(9):
        transition(state, pos=(.22, 0., .03), contacts=(True, True, True, False))
    transition(state, pos=(.22, 0., .006), contacts=(True, True, True, True))
    assert state.holds[0, 0] == 0
    assert not state.milestones[0, 1]
    pickup(state)
    assert state.milestones[0, 1]


def test_place_requires_prior_lift_release_full_footprint_and_settling():
    state = TaskState(1)
    for _ in range(30):
        transition(state, pos=CONFIG.target_position, contacts=(False, False, False, True))
    assert not state.milestones[0, 2]  # Pushing to target is not pick and place.
    pickup(state)
    bad = [
        dict(pos=CONFIG.target_position, contacts=(True, False, True, True)),
        dict(pos=CONFIG.target_position, contacts=(False, False, False, False)),
        dict(pos=(.25, .09, .005), contacts=(False, False, False, True)),
        dict(pos=CONFIG.target_position, contacts=(False, False, False, True), velocity=(.03, 0., 0., 0., 0., 0.)),
        dict(pos=CONFIG.target_position, contacts=(False, False, False, True), velocity=(0., 0., 0., 0., 0., .3)),
    ]
    for kwargs in bad:
        for _ in range(30):
            transition(state, **kwargs)
        assert not state.milestones[0, 2]
    for _ in range(24):
        transition(state, pos=CONFIG.target_position, contacts=(False, False, False, True))
    assert not state.milestones[0, 2]
    transition(state, pos=CONFIG.target_position, contacts=(False, False, False, True))
    assert state.milestones[0, 2]
    assert state.reward.item() > 99.


def test_rotated_footprint_and_partial_reset():
    state = TaskState(2)
    pickup(state)
    # At 90 degrees, the long side extends beyond the square in y.
    for _ in range(25):
        transition(state, pos=(.22, .12, .005), quat=(2**-.5, 0., 0., 2**-.5), contacts=(False, False, False, True))
    assert not state.milestones[:, 2].any()
    before = state.milestones[1].clone()
    state.reset(torch.tensor([0]))
    assert not state.milestones[0].any()
    assert torch.equal(state.milestones[1], before)


def test_contract_and_option_validation(calibration):
    doc = manifest(calibration)
    assert json.loads(json.dumps(doc)) == doc
    assert doc['action_shape'] == [1, 6]
    assert doc['observation_shape'] == [1, 46]
    for kwargs in ({'mode': 'random'}, {'start_mode': 'random'}, {'robust': True}, {'workspace': 'wide'}, {'targets': 'rotate'}):
        with pytest.raises(ValueError):
            validate_options(**kwargs)
    lower, upper = action_bounds(calibration)
    previous = np.array(CONFIG.start_q)
    actual = next_command(np.ones(6), previous, lower, upper)
    assert np.all(actual >= lower) and np.all(actual <= upper)
    assert np.all(actual > previous)
    with pytest.raises(ValueError):
        next_command(np.zeros(3), previous, lower, upper)


def test_cpu_physical_brick_and_fixed_reset(calibration):
    robot = CpuPickPlace(calibration)
    initial = robot.reset()
    assert initial.shape == (46,)
    assert robot.model.nq == 13 and robot.model.nv == 12
    assert robot.model.body('brick').mass == pytest.approx(CONFIG.brick_mass)
    for _ in range(10):
        obs, reward = robot.step((robot.home-robot.center)/robot.scale)
        assert np.isfinite(obs).all() and np.isfinite(reward)
    assert robot.inputs()[-1][0, 3]
    np.testing.assert_allclose(robot.data.xpos[robot.brick_id, 2], .005, atol=.001)
    np.testing.assert_array_equal(robot.reset(), initial)
    assert not robot.state.milestones.any()


def test_gpu_reset_and_cpu_contract_parity(calibration):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from mjlab.envs import ManagerBasedRlEnv
    from so101_ppo.pick_place_task import make_env_cfg
    cfg = make_env_cfg(calibration, num_envs=2)
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        obs, _ = env.reset()
        cpu = CpuPickPlace(calibration)
        np.testing.assert_allclose(obs['actor'][0].cpu(), cpu.reset(), atol=1e-5)
        np.testing.assert_allclose(obs['actor'][0].cpu(), obs['actor'][1].cpu(), atol=1e-5)
        action = torch.tensor((cpu.home-cpu.center)/cpu.scale, device=env.device).repeat(2, 1)
        for _ in range(5):
            obs, reward, _, _, _ = env.step(action)
            cpu_obs, cpu_reward = cpu.step(action[0].cpu().numpy())
            np.testing.assert_allclose(obs['actor'][0].cpu(), cpu_obs, atol=2e-3)
            assert reward[0].item() == pytest.approx(cpu_reward, abs=2e-4)
        state = env.command_manager.get_term('goal').state
        state.milestones[1, :2] = True
        command = env.action_manager.get_term('arm').command[1].clone()
        brick = env.scene['brick'].data.root_link_pose_w[1].clone()
        env.reset(env_ids=torch.tensor([0], device=env.device))
        assert not state.milestones[0].any() and state.milestones[1, :2].all()
        torch.testing.assert_close(env.action_manager.get_term('arm').command[1], command)
        torch.testing.assert_close(env.scene['brick'].data.root_link_pose_w[1], brick)
    finally:
        env.close()


def test_failure_state_has_finite_reward_and_resets():
    state = TaskState(1)
    transition(state, pos=(.8, 0., .005))
    assert state.failed.item()
    state.reset()
    transition(state, velocity=(float('nan'), 0., 0., 0., 0., 0.))
    assert state.failed.item() and torch.isfinite(state.reward).all()
    state.reset()
    assert not state.failed.any() and not state.reward_components.any()


def test_pick_place_preview_status():
    from so101_ppo.visualization import overlay_text
    text = overlay_text(2, dict(step=1, time_s=.02, episode_steps=3000,
        failed=False, success=False, touched=True, picked=True, placed=False, distance_m=.05))
    assert 'Pickup: True' in text and 'Placed: False' in text
    assert 'PICK AND PLACE' in text


def test_scripted_physical_pick_and_place(calibration):
    from so101_ppo.pick_place_cpu import run_episode
    from so101_ppo.pick_place_scripted import ScriptedPolicy
    robot = CpuPickPlace(calibration)
    result = run_episode(robot, ScriptedPolicy(robot))
    assert result['touched'] == result['picked'] == result['placed'] == result['success'] == 1
    assert result['failed'] == 0
    assert result['return'] > 90
    assert result['steps'] < robot.episode_steps
    contacts = robot.inputs()[-1][0]
    assert contacts[3] and not contacts[2]  # Released and supported by the floor.


def distance_reward(distance, *, moving=False, prior_pickup=False):
    state = TaskState(1)
    state.milestones[0, 1] = prior_pickup
    pos = torch.tensor([[.22, 0., .005]])
    tool = pos + torch.tensor([[0., 0., distance]])
    zeros = torch.zeros(1, 6)
    from so101_ppo.contract import MAX_COMMAND_SPEED
    delta = torch.full((1, 6), CONTROL_DT*MAX_COMMAND_SPEED) if moving else zeros
    state.advance(torch.tensor([CONFIG.start_q]), zeros, pos,
        torch.tensor([[1., 0., 0., 0.]]), zeros, tool, torch.tensor([CONFIG.target_position]),
        torch.tensor([[False, False, False, True]]), delta)
    return state.reward.item()


def test_approach_is_positive_and_strictly_increases_as_hand_gets_closer():
    rewards = [distance_reward(d) for d in (.1, .05, .02, .005)]
    assert 0 < rewards[0] < rewards[1] < rewards[2] < rewards[3]
    # Even all six commands moving at their slew limit must not outweigh
    # proximity guidance at the initial 55 mm distance.
    assert distance_reward(.055, moving=True) > 0
    # A prior pickup must not remove approach guidance after dropping the brick.
    assert distance_reward(.02, prior_pickup=True) > distance_reward(.1, prior_pickup=True)


def test_staying_close_receives_positive_reward_without_potential_history():
    state = TaskState(1)
    transition(state)
    first = state.reward.item()
    transition(state)
    assert first > 0 and state.reward.item() == pytest.approx(first)


def test_dense_grasp_and_lift_rewards_precede_pickup_bonus():
    state = TaskState(1)
    state.milestones[0, 0] = True  # Compare guidance without first-touch bonus.
    transition(state, contacts=(False, False, False, True))
    approach = state.reward_components[0, 3].item()
    transition(state, contacts=(True, True, True, True))
    grasp = state.reward_components[0, 3].item()
    transition(state, pos=(.22, 0., .015), contacts=(True, True, True, False))
    lift = state.reward_components[0, 3].item()
    assert approach < grasp < lift
    assert not state.milestones[0, 1]  # Guidance exists before clearing the lift threshold.


def test_carry_guidance_requires_current_grasp_and_lift():
    state = TaskState(1)
    state.milestones[0, 1] = True
    tool = torch.tensor([[.2, .0, .04]])
    bottom = torch.tensor([.025])
    pos = torch.tensor([[.22, 0., .03]])
    far = torch.tensor([[.22, .09, .005]])
    near = torch.tensor([[.22, .01, .005]])
    held = torch.tensor([[True, True, True, False]])
    dropped = torch.tensor([[False, False, False, True]])
    assert state.dense_reward(pos, tool, near, bottom, held) > state.dense_reward(pos, tool, far, bottom, held)
    assert state.dense_reward(pos, tool, near, bottom, dropped) == state.dense_reward(pos, tool, far, bottom, dropped)
    assert state.dense_reward(pos, tool, near, torch.zeros(1), held) == state.dense_reward(pos, tool, far, torch.zeros(1), held)
    max_dense_rate = sum((CONFIG.approach_reward_rate, CONFIG.grasp_reward_rate,
                          CONFIG.lift_reward_rate, CONFIG.carry_reward_rate))
    assert max_dense_rate*CONFIG.episode_seconds < CONFIG.placement_bonus
