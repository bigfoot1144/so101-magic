import copy
import json
import random
import subprocess
import sys
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from so101_ppo.contract import HOME
from so101_ppo.cpu import CpuArm
from so101_ppo.evaluate import run_episode
from so101_ppo.model import target_bank
from so101_ppo.visualization import (
    PreviewLogger, TrainingVisualization, compile_timelapse, overlay_text,
    preserve_rng, snapshot_policy, video_writer,
)


@pytest.fixture
def actor():
    observations = TensorDict({'actor': torch.randn(4, 21)}, batch_size=[4])
    model = MLPModel(observations, {'actor': ['actor']}, 'actor', 3,
                     hidden_dims=[8, 8], obs_normalization=True,
                     distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': .5})
    model.train()
    model.update_normalization(observations)
    # Populate distribution caches just as a training update does.
    model(observations, stochastic_output=True)
    return model


@pytest.fixture
def preview(calibration, tmp_path, monkeypatch):
    def prepare(self, calibration, task_mode):
        self.robot = CpuArm(calibration)
        self._prepare_targets(task_mode)
        self._prepare_starts(calibration)
    monkeypatch.setattr(TrainingVisualization, '_prepare', prepare)
    return TrainingVisualization(calibration, tmp_path, 'save', every=2)


def runner_for(actor):
    return SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor),
                           env=SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=64)))


def test_snapshot_preserves_actor_and_normalizer(actor):
    before = copy.deepcopy(actor.state_dict())
    modes = [m.training for m in actor.modules()]
    policy = snapshot_policy(actor)
    obs = np.ones(21, np.float32)
    expected = actor.as_onnx(verbose=False).eval()(torch.from_numpy(obs[None])).detach().numpy()[0]
    np.testing.assert_allclose(policy(obs), expected)
    np.testing.assert_array_equal(policy(obs), policy(obs))
    for key, value in before.items():
        torch.testing.assert_close(actor.state_dict()[key], value, rtol=0, atol=0)
    assert [m.training for m in actor.modules()] == modes
    with torch.no_grad():
        next(actor.parameters()).add_(1)
    np.testing.assert_array_equal(policy(obs), expected)


def test_rng_restored_even_on_failure():
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    if torch.cuda.is_available():
        torch.cuda.init()
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    python, numpy = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError), preserve_rng():
        random.random()
        np.random.rand()
        torch.rand(4)
        if cuda:
            torch.rand(4, device='cuda:0')
        raise RuntimeError('test failure')
    assert random.getstate() == python
    np.testing.assert_array_equal(np.random.get_state()[1], numpy[1])
    assert np.random.get_state()[2:] == numpy[2:]
    torch.testing.assert_close(torch.get_rng_state(), cpu)
    for actual, expected in zip(torch.cuda.get_rng_state_all() if cuda else [], cuda):
        torch.testing.assert_close(actual, expected)


def test_schedule_and_resume_are_local_and_final_is_not_duplicated(preview, actor, monkeypatch):
    captures = []
    monkeypatch.setattr(preview, '_episode', lambda policy, steps: captures.append((preview.completed, steps)))
    runner = runner_for(actor)  # Simulates a checkpoint with existing environment steps.
    preview.capture(runner)
    for _ in range(4):
        preview.iteration(runner)
    preview.capture(runner)
    assert captures == [(0, 64), (2, 64), (4, 64)]
    preview.iteration(runner)
    preview.capture(runner)
    assert captures[-1] == (5, 64)


def test_logger_forwards_before_callback():
    calls = []
    original = SimpleNamespace(writer='writer', log=lambda **kw: calls.append(kw) or 17)
    logger = PreviewLogger(original, lambda: calls.append('preview'))
    assert logger.writer == 'writer'
    assert logger.log(it=8) == 17
    assert calls == [{'it': 8}, 'preview']


def test_render_failure_disables_previews_but_not_training(preview, actor, monkeypatch, capsys):
    def fail(*args):
        raise RuntimeError('renderer lost')
    monkeypatch.setattr(preview, '_episode', fail)
    preview.capture(runner_for(actor))
    assert preview.disabled
    assert 'training continues' in capsys.readouterr().err
    assert 'renderer lost' in json.loads((preview.root / 'manifest.json').read_text())['errors'][0]
    preview.iteration(runner_for(actor))
    assert preview.completed == 1


def test_closed_window_keeps_recording(preview, monkeypatch):
    # Use tiny fake rendering, but real MuJoCo episode dynamics and MP4 encoding.
    from PIL import ImageFont
    preview.font = ImageFont.load_default()
    preview.show = True
    closed = []
    preview.view = SimpleNamespace(is_running=lambda: False, close=lambda: closed.append(True))
    preview.renderer = SimpleNamespace(update_scene=lambda *a, **kw: None,
        render=lambda: np.zeros((32, 32, 3), dtype=np.uint8), scene=None)
    preview.camera = None
    monkeypatch.setattr('so101_ppo.visualization.add_target', lambda *a: None)
    preview._episode(lambda obs: np.zeros(3), 64)
    assert closed == [True] and not preview.show and preview.view is None
    assert preview.records[0]['frames'] == 100
    assert (preview.root / preview.records[0]['clip']).is_file()


@pytest.mark.parametrize('speed,expected_frames', [(4., 5), (2., 10), (.5, 40)])
def test_compilation_speed_order_and_duration(tmp_path, speed, expected_frames):
    clips = []
    for i, color in enumerate((30, 200)):
        path = tmp_path / f'{i}.mp4'
        with video_writer(path) as writer:
            for _ in range(10):
                writer.append_data(np.full((32, 32, 3), color, np.uint8))
        clips.append(path)
    output = tmp_path / 'timelapse.mp4'
    assert compile_timelapse(clips, output, speed) == expected_frames
    with imageio.get_reader(output, format='FFMPEG') as reader:
        frames = list(reader.iter_data())
        assert reader.get_meta_data()['fps'] == 25
    assert len(frames) == expected_frames
    assert frames[0].mean() < 40 and frames[-1].mean() > 190
    assert not list(tmp_path.glob('*.partial.mp4'))


def test_compilation_failure_retains_clips_and_does_not_raise(preview, monkeypatch):
    clip = preview.root / 'episode_000000.mp4'
    clip.write_bytes(b'kept')
    preview.records = [{'clip': clip.name}]
    monkeypatch.setattr('so101_ppo.visualization.compile_timelapse',
                        lambda *args: (_ for _ in ()).throw(RuntimeError('encoder failed')))
    preview.close()
    assert clip.read_bytes() == b'kept'
    assert 'compilation failed' in preview.errors[-1]


def test_preview_rollout_deterministic_and_other_arm_unchanged(calibration):
    training = CpuArm(calibration)
    goal = target_bank()[0][0]
    training.reset(goal)
    training.step(np.array([.5, -.2, .1]))
    before = [training.q, training.dq, training.history.copy(), training.command.copy()]
    preview = CpuArm(calibration)
    policy = lambda obs: np.zeros(3)
    first = run_episode(preview, policy, goal, HOME)
    second = run_episode(preview, policy, goal, HOME)
    assert first == second
    for a, b in zip(before, [training.q, training.dq, training.history, training.command]):
        np.testing.assert_array_equal(a, b)


def test_overlay_contains_progress_and_outcome():
    text = overlay_text(50, {'step': 200, 'time_s': 4., 'distance_m': .022,
                            'success': False, 'failed': False})
    assert '50' in text and '4.00 s' in text and '22.0 mm' in text
    assert 'TIMEOUT' in text


@pytest.mark.parametrize('option,value', [('--visualize-every', '0'), ('--visualize-every', '-2'),
                                         ('--timelapse-speed', '0'), ('--timelapse-speed', 'nan')])
def test_cli_rejects_invalid_visualization_values(option, value):
    result = subprocess.run([sys.executable, '-m', 'so101_ppo.train', '--calibration', 'unused',
                             option, value], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'must be' in result.stderr


def test_interruption_discards_partial_clip_and_compiles_completed_clips(preview, monkeypatch):
    clip = preview.root / 'episode_000000.mp4'
    with video_writer(clip) as writer:
        for _ in range(8):
            writer.append_data(np.full((32, 32, 3), 90, np.uint8))
    preview.records = [{'clip': clip.name}]
    preview.completed = 1
    monkeypatch.setattr('so101_ppo.visualization.run_episode',
                        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        try:
            preview._episode(lambda obs: np.zeros(3), 32)
        finally:
            preview.close()
    assert clip.exists()
    assert (preview.root / 'timelapse.mp4').exists()
    assert not list(preview.root.glob('*.partial.mp4'))
    assert not (preview.root / 'episode_000001.mp4').exists()


def test_preflight_failure_raises_and_cleans_up(calibration, tmp_path, monkeypatch):
    closed = []
    def prepare(self, *args):
        self.renderer = SimpleNamespace(close=lambda: closed.append(True))
        raise RuntimeError('OpenGL unavailable')
    monkeypatch.setattr(TrainingVisualization, '_prepare', prepare)
    with pytest.raises(RuntimeError, match='OpenGL unavailable'):
        TrainingVisualization(calibration, tmp_path, 'save')
    assert closed == [True]


def test_encoder_cleanup_error_does_not_mask_interrupt(preview, monkeypatch):
    def fail_close():
        raise RuntimeError('encoder close failed')
    monkeypatch.setattr('so101_ppo.visualization.video_writer',
                        lambda path: SimpleNamespace(close=fail_close))
    monkeypatch.setattr('so101_ppo.visualization.run_episode',
                        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        preview._episode(lambda obs: np.zeros(3), 0)
    assert 'cleanup failed' in preview.errors[-1]


def test_cli_help_lists_visualization_options():
    result = subprocess.run([sys.executable, '-m', 'so101_ppo.train', '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for option in ['--visualize', '--visualize-every', '--visualize-targets', '--timelapse-speed', '--start-mode', '--workspace']:
        assert option in result.stdout
    assert '+/-5%' in result.stdout


@pytest.fixture
def target_preview(calibration, tmp_path, monkeypatch):
    def prepare(self, calibration, task_mode):
        self.robot = CpuArm(calibration)
        self._prepare_targets(task_mode)
        self._prepare_starts(calibration)
    monkeypatch.setattr(TrainingVisualization, '_prepare', prepare)
    def make(name, task_mode='random', selection='auto', checkpoint=None, start_mode='home'):
        return TrainingVisualization(calibration, tmp_path / name, 'save', every=1,
                                     task_mode=task_mode, target_selection=selection,
                                     source_checkpoint=checkpoint, start_mode=start_mode)
    return make


@pytest.mark.parametrize('task_mode,selection,effective', [
    ('random', 'auto', 'rotate'), ('random', 'rotate', 'rotate'), ('random', 'fixed', 'fixed'),
    ('fixed', 'auto', 'fixed'), ('fixed', 'rotate', 'fixed'), ('fixed', 'fixed', 'fixed'),
])
def test_target_selection_modes_and_legacy_first_pose(target_preview, task_mode, selection, effective):
    preview = target_preview('modes', task_mode, selection)
    assert preview.target_selection == effective
    # Reproduce the original preview initialization, before rotation existed.
    rng = np.random.default_rng(2026)
    goals, _ = target_bank(task_mode, seed=54321)
    index = int(rng.integers(len(goals)))
    q = HOME.copy()
    q[:3] += rng.uniform(-.02, .02, 3)
    q[:3] = np.clip(q[:3], *preview.robot.bounds)
    np.testing.assert_array_equal(preview.goal, goals[index])
    np.testing.assert_array_equal(preview.q, q)
    assert preview.metadata['initial_target_index'] == index
    assert ('target_m' in preview.metadata) == (effective == 'fixed')
    initial = preview.goal.copy()
    for _ in range(5):
        before = preview.goal.copy()
        preview._select_next_target()
        np.testing.assert_array_equal(preview.goal, goals[preview.target_index])
        np.testing.assert_array_equal(preview.q, q)
        if effective == 'rotate':
            assert np.linalg.norm(preview.goal.astype(float) - before) >= .03
        else:
            np.testing.assert_array_equal(preview.goal, initial)


def test_rotating_targets_repeat_on_new_and_resumed_invocations(target_preview):
    first = target_preview('first')
    repeated = target_preview('repeated')
    resumed = target_preview('resumed', checkpoint='old/checkpoint.pt')
    for _ in range(25):
        for other in (repeated, resumed):
            assert first.target_index == other.target_index
            np.testing.assert_array_equal(first.goal, other.goal)
            np.testing.assert_array_equal(first.q, other.q)
        for p in (first, repeated, resumed):
            p._select_next_target()
    assert resumed.metadata['source_checkpoint'] == 'old/checkpoint.pt'


def test_target_rotation_falls_back_to_farthest_when_bank_is_too_small(target_preview):
    preview = target_preview('small')
    preview._goals = np.array([[0., 0., .2], [.01, 0., .2], [.02, 0., .2]])
    preview.goal = preview._goals[0].copy()
    preview._select_next_target()
    assert preview.target_index == 2
    preview._select_next_target()
    assert preview.target_index == 0


@pytest.mark.parametrize("start_mode", ["home", "random"])
def test_rotation_happens_only_on_new_captures_and_preserves_training_state(target_preview, actor, monkeypatch, start_mode):
    preview = target_preview("captures", start_mode=start_mode)
    captures = []
    def episode(policy, steps):
        captures.append(preview.target_index)
        obs = preview.robot.reset(preview.goal, preview.q)
        np.testing.assert_array_equal(obs[12:15], preview.goal)
        policy(obs)
    monkeypatch.setattr(preview, '_episode', episode)
    runner = runner_for(actor)
    original_actor = copy.deepcopy(actor.state_dict())
    modes = [m.training for m in actor.modules()]
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    preview.capture(runner)
    preview.capture(runner)  # Duplicate initial/final call must not advance selection.
    preview.iteration(runner)
    preview.iteration(runner)
    preview.capture(runner)
    assert len(captures) == 3
    assert captures[0] != captures[1] and captures[1] != captures[2]
    assert runner.env.unwrapped.common_step_counter == 64
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    for actual, expected in zip(torch.cuda.get_rng_state_all() if cuda_state else [], cuda_state):
        torch.testing.assert_close(actual, expected)
    for key, value in original_actor.items():
        torch.testing.assert_close(actor.state_dict()[key], value, rtol=0, atol=0)
    assert [m.training for m in actor.modules()] == modes


def test_overlay_identifies_episode_target():
    text = overlay_text(25, {'step': 2, 'time_s': .04, 'distance_m': .05,
                            'success': False, 'failed': False}, 42, [.3, -.04, .2])
    assert 'Target #42' in text and '(0.300, -0.040, 0.200) m' in text


@pytest.mark.parametrize('task_mode,selection,rotates', [
    ('random', 'auto', True), ('random', 'fixed', False), ('random', 'rotate', True),
    ('fixed', 'auto', False), ('fixed', 'rotate', False),
])
def test_random_start_preview_sequence(target_preview, task_mode, selection, rotates):
    home = target_preview('home', task_mode, selection)
    random = target_preview('random', task_mode, selection, start_mode='random')
    resumed = target_preview('resumed', task_mode, selection, checkpoint='old/checkpoint.pt', start_mode='random')
    initial = random.q.copy()
    for _ in range(10):
        assert random.target_index == home.target_index
        np.testing.assert_array_equal(random.q, random._starts[random.start_index])
        np.testing.assert_array_equal(random.q, resumed.q)
        before = random.start_index
        for p in (home, random, resumed):
            p._select_next_target()
            p._select_next_start()
        assert (random.start_index != before) == rotates
        if not rotates:
            np.testing.assert_array_equal(random.q, initial)
    assert random.metadata['start_mode'] == 'random'
    assert random.metadata['initial_q_rad'] == initial.tolist()


def test_overlay_identifies_random_start():
    text = overlay_text(0, {'step': 1, 'time_s': .02, 'distance_m': .05,
                            'success': False, 'failed': False}, 42, [.3, -.04, .2], 19)
    assert 'Start #19' in text
