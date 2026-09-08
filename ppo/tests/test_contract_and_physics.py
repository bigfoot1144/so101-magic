import json
import shutil

import numpy as np
import pytest

from so101_ppo.calibration import Calibration, RAD_PER_TICK, delay_steps
from so101_ppo.check import check_cpu
from so101_ppo.contract import ACTION_SCALE, CONTROL_DT, HOME, MAX_COMMAND_SPEED, next_command
from so101_ppo.cpu import CpuArm
from so101_ppo.model import target_bank
from so101_ppo.replay import replay


def test_command_bounds_and_held_servos(calibration):
    rng = np.random.default_rng(9)
    command = HOME.copy()
    for _ in range(100):
        previous = command.copy()
        command = next_command(rng.normal(size=3) * 5, command, calibration.task_bounds())
        assert np.max(np.abs(command[:3] - previous[:3])) <= MAX_COMMAND_SPEED * CONTROL_DT + 1e-6
        assert np.all(np.abs(command[:3] - HOME[:3]) <= ACTION_SCALE + 1e-6)
        np.testing.assert_array_equal(command[3:], HOME[3:])


def test_known_goal_is_physically_holdable(calibration):
    assert check_cpu(calibration)['cpu_oracle_error_m'] < .015


def test_reset_reproduces_delayed_trajectory(calibration):
    robot = CpuArm(calibration)
    goal = target_bank()[0][0]
    robot.reset(goal)
    action = np.array([.4, -.2, -.5])
    for _ in range(10):
        robot.step(action)
    expected = robot.q
    for _ in range(30):
        robot.step(-action)
    robot.reset(goal)
    for _ in range(10):
        robot.step(action)
    np.testing.assert_allclose(robot.q, expected, atol=1e-9)


def test_mapping_signs_offsets_and_encoder_rounding(calibration):
    raw = np.array([2030, 1891, 1792, 2160, 2301, 2500])
    expected = (raw - calibration.mid) * RAD_PER_TICK * calibration.sign + calibration.offset
    np.testing.assert_allclose(calibration.raw_to_sim(raw), expected)
    np.testing.assert_array_equal(calibration.sim_to_raw(expected), raw)
    q = np.linspace(-.3, .3, 6)
    assert np.max(abs(calibration.quantize(q) - q)) <= RAD_PER_TICK / 2 + 1e-12


def test_delay_uses_first_available_tick(calibration):
    np.testing.assert_array_equal(delay_steps([0, .002, .007, .08]), [0, 1, 4, 40])
    robot = CpuArm(calibration)
    initial = calibration.quantize(HOME)
    robot.reset(np.zeros(3), initial_target=initial)
    target = initial + .01
    for tick in range(42):
        robot.step_targets(target)
        effective = np.array([ctrl.q_target[0] for ctrl in robot.controllers])
        expected = np.where(tick >= calibration.lags, target, initial)
        np.testing.assert_allclose(effective, expected)


def test_cpu_reproduces_export_from_calibration_mujoco(calibration):
    result = replay(calibration)
    assert result['passed']
    assert max(x['rmse'] for x in result['trials'][0]['cpu_vs_reference_deg'].values()) < .01


def test_synthetic_and_mutated_bundles_are_detected(calibration, tmp_path):
    with pytest.raises(ValueError, match='Synthetic'):
        Calibration(calibration.root)
    copy = tmp_path / 'bundle'
    shutil.copytree(calibration.root, copy)
    (copy / 'motors' / 'shoulder_pan.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        Calibration(copy, allow_synthetic=True)


def test_fixed_goal_is_original_baseline():
    np.testing.assert_allclose(target_bank()[0][0], [.3145965338, -.0332624428, .2507610917], atol=1e-9)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_replay_initializes_recorded_pose_away_from_task_home(calibration, device):
    import torch
    from so101_ppo.replay import cpu_replay, mjlab_replay

    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    # Exercise a recording where every held servo differs from task HOME.
    # Committing env.reset() actuator state before the first integration used
    # to overwrite this target history and produce a large startup transient.
    q = calibration.quantize(np.array([.2, -.2, .7, -.1, .3, .4]))
    log = {
        'initial_target_raw': calibration.sim_to_raw(q).tolist(),
        'initial_voltage_v': calibration.voltage.tolist(),
        'samples': [{'t': float(t), 'q_rad': q.tolist()} for t in np.arange(0., .201, .02)],
        'commands': [{'t': 0., 'q_target_rad': q.tolist()}],
    }
    cpu = cpu_replay(calibration, log)
    mjlab = mjlab_replay(calibration, log, device)
    delta = np.rad2deg(cpu - mjlab)
    assert np.max(np.sqrt(np.mean(delta ** 2, axis=0))) <= .05
    assert np.max(np.abs(delta)) <= .25
