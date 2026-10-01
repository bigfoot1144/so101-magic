import csv
import json

import mujoco
import numpy as np
import pytest
import torch

from so101_ppo.contract import ACTION_SCALE, HOME, JOINTS, SITE
from so101_ppo.cpu import CpuArm
from so101_ppo.model import cpu_spec, indices, target_bank
from so101_ppo.starts import (
    EVAL_START_SEED, TRAIN_START_SEED, home_start, resolve_start_mode, sample_start, start_bank,
)


@pytest.mark.parametrize('seed', [TRAIN_START_SEED, EVAL_START_SEED])
def test_start_bank_geometry_and_determinism(calibration, seed):
    bank = start_bank(calibration, seed)
    np.testing.assert_array_equal(bank, start_bank(calibration, seed))
    assert len(bank) == 512
    assert not bank.flags.writeable
    assert np.all(bank >= calibration.lower) and np.all(bank <= calibration.upper)
    assert np.all(np.abs(bank[:, :3] - HOME[:3]) <= ACTION_SCALE)
    np.testing.assert_array_equal(bank[:, 3:], np.tile(HOME[3:], (len(bank), 1)))
    assert np.ptp(bank[:, :3], axis=0).min() > .2
    model = cpu_spec().compile()
    data = mujoco.MjData(model)
    qi, _ = indices(model)
    joint_ids = [model.joint(name).id for name in JOINTS]
    assert np.all(bank >= model.jnt_range[joint_ids, 0])
    assert np.all(bank <= model.jnt_range[joint_ids, 1])
    for q in bank:
        for fraction in np.linspace(0, 1, 9):
            data.qpos[qi] = HOME + fraction * (q - HOME)
            mujoco.mj_forward(model, data)
            assert data.ncon == 0
            assert data.site(SITE).xpos[2] >= .08
    other_seed = EVAL_START_SEED if seed == TRAIN_START_SEED else TRAIN_START_SEED
    assert not np.array_equal(bank, start_bank(calibration, other_seed))


def test_rejects_out_of_bounds_without_clipping(calibration, monkeypatch):
    from so101_ppo import starts
    candidates = start_bank(calibration)[:1].copy()
    invalid = candidates.copy()
    invalid[0, 0] = 99
    monkeypatch.setattr(starts, 'target_bank', lambda *a, **kw: (None, np.concatenate([candidates, invalid])))
    np.testing.assert_array_equal(start_bank(calibration), candidates)
    monkeypatch.setattr(starts, 'target_bank', lambda *a, **kw: (None, invalid))
    with pytest.raises(ValueError, match='No reachable starting poses'):
        start_bank(calibration)


@pytest.mark.parametrize('requested,manifest,expected', [
    (None, None, 'home'), (None, {}, 'home'),
    (None, {'start_mode': 'random'}, 'random'),
    ('home', {'start_mode': 'random'}, 'home'),
    ('random', {'start_mode': 'home'}, 'random'),
])
def test_start_mode_inheritance_and_override(requested, manifest, expected):
    assert resolve_start_mode(requested, manifest) == expected


def test_home_start_preserves_legacy_rng_and_pose(calibration):
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    expected = HOME.copy()
    expected[:3] += a.uniform(-.02, .02, 3)
    expected[:3] = np.clip(expected[:3], *calibration.task_bounds())
    np.testing.assert_array_equal(home_start(b, calibration.task_bounds()), expected)
    assert a.random() == b.random()


def test_sample_start_excludes_previous_and_handles_singleton(calibration):
    bank = start_bank(calibration)
    rng = np.random.default_rng(12)
    previous = None
    for _ in range(100):
        index, q = sample_start(bank, rng, previous)
        assert index != previous
        np.testing.assert_array_equal(q, bank[index])
        previous = index
    assert sample_start(bank[:1], rng, 0)[0] == 0


def test_cpu_random_reset_state_and_history(calibration):
    robot = CpuArm(calibration)
    goal = target_bank()[0][0]
    for q in start_bank(calibration)[:4]:
        robot.reset(goal, q)
        robot.step(np.zeros(3))
        obs = robot.reset(goal, q)
        np.testing.assert_array_equal(robot.q, q)
        np.testing.assert_array_equal(robot.dq, np.zeros(6))
        np.testing.assert_array_equal(robot.command, q)
        np.testing.assert_allclose(obs[:6], q - HOME, atol=1e-8)
        expected = calibration.quantize(q)
        np.testing.assert_allclose(robot.history, np.tile(expected, (len(robot.history), 1)))
        for i, controller in enumerate(robot.controllers):
            np.testing.assert_allclose(controller.actuator_state[0], [expected[i]])


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_mjlab_random_resets_and_partial_reset(calibration, device):
    from mjlab.envs import ManagerBasedRlEnv
    from so101_ppo.task import make_env_cfg
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    env = ManagerBasedRlEnv(make_env_cfg(calibration, mode='random', start_mode='random', num_envs=4), device=device)
    try:
        env.reset()
        action = env.action_manager.get_term('arm')
        robot = env.scene['robot']
        ids, _ = robot.find_joints(JOINTS, preserve_order=True)
        bank = env._start_pose_bank
        q = robot.data.joint_pos[:, ids].clone()
        assert torch.unique(q, dim=0).shape[0] > 1
        assert torch.all(torch.any(torch.all(q[:, None] == bank[None], dim=-1), dim=1))
        torch.testing.assert_close(action.command, q, rtol=0, atol=0)
        torch.testing.assert_close(robot.data.joint_vel[:, ids], torch.zeros_like(q))
        # Reset after actual integration, so untouched histories contain real data.
        env.step(torch.zeros(4, 3, device=device))
        before_q = robot.data.joint_pos[:, ids].clone()
        before_command = action.command.clone()
        before_history = [a._history.clone() for a in robot.actuators]
        reset_ids = torch.tensor([0, 2], device=device)
        keep = torch.tensor([1, 3], device=device)
        env._reset_idx(reset_ids)
        q = robot.data.joint_pos[:, ids]
        torch.testing.assert_close(q[keep], before_q[keep], rtol=0, atol=0)
        torch.testing.assert_close(action.command[keep], before_command[keep], rtol=0, atol=0)
        torch.testing.assert_close(action.command[reset_ids], q[reset_ids], rtol=0, atol=0)
        torch.testing.assert_close(robot.data.joint_vel[:, ids][reset_ids], torch.zeros(2, 6, device=device))
        targets = action.quantized_command()
        for i, actuator in enumerate(robot.actuators):
            torch.testing.assert_close(actuator._history[:, keep], before_history[i][:, keep], rtol=0, atol=0)
            expected = targets[reset_ids, i:i+1].expand(len(actuator._history), -1, -1)
            torch.testing.assert_close(actuator._history[:, reset_ids], expected, rtol=0, atol=0)
        assert env._start_pose_bank is bank
    finally:
        env.close()


@pytest.mark.parametrize('saved,override,expected', [('random', None, 'random'), ('random', 'home', 'home'), (None, None, 'home')])
def test_evaluation_inherits_and_records_starts(calibration, tmp_path, monkeypatch, saved, override, expected):
    import so101_ppo.evaluate as evaluation
    manifest = {'task_mode': 'random'}
    if saved:
        manifest['start_mode'] = saved
    monkeypatch.setattr(evaluation, 'load_policy', lambda *a: (lambda obs: np.zeros(3), manifest, calibration))
    output = tmp_path / 'evaluation.json'
    result = evaluation.evaluate('unused.onnx', episodes=2, start_mode=override, output=output)
    assert result['start_mode'] == expected
    rows = list(csv.DictReader(output.with_suffix('.csv').open()))
    bank = start_bank(calibration, EVAL_START_SEED)
    for row in rows:
        q = json.loads(row['initial_q_rad'])
        assert row['start_mode'] == expected
        if expected == 'random':
            np.testing.assert_array_equal(q, bank[int(row['start_index'])])
        else:
            assert not row['start_index']
            assert np.max(np.abs(np.asarray(q)[:3] - HOME[:3])) <= .020001
