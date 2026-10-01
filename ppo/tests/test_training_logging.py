"""Regressions for fresh rollout statistics during long fixed episodes."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from rsl_rl.utils.logger import Logger

from so101_ppo.training_logging import RolloutLogger
from so101_ppo.visualization import PreviewLogger


class Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, step):
        self.values[name, step] = float(value)


def logger():
    base = Logger(None, {'algorithm': {'rnd_cfg': None}, 'num_steps_per_env': 2}, {},
        num_envs=2, is_distributed=False, gpu_world_size=1, gpu_global_rank=0, device='cpu')
    base.writer, base.logger_type = Writer(), 'tensorboard'
    return RolloutLogger(base)


def log_update(logger, iteration):
    logger.log(it=iteration, start_it=0, total_it=4, collect_time=1., learn_time=.1,
        loss_dict={}, learning_rate=.001, action_std=torch.ones(1), rnd_weight=None)


def test_live_reward_changes_before_and_between_completed_episodes(capsys):
    live = logger()
    # A live metric exists before any of the long episodes finish.
    live.process_env_step(torch.tensor([1., 3.]), torch.zeros(2), {})
    log_update(live, 0)
    assert live.writer.values['Rollout/mean_step_reward', 0] == 2.
    assert ('Train/mean_reward', 0) not in live.writer.values
    # Completed episode returns retain the upstream definition: 6 and 10.
    live.process_env_step(torch.tensor([5., 7.]), torch.ones(2), {})
    log_update(live, 1)
    assert live.writer.values['Train/mean_reward', 1] == 8.
    assert live.writer.values['Rollout/completed_episodes', 1] == 2
    # No completions: old mean remains 8, new rollout mean changes to .75.
    live.process_env_step(torch.tensor([.25, .75]), torch.zeros(2), {})
    live.process_env_step(torch.tensor([.5, 1.5]), torch.zeros(2), {})
    log_update(live, 2)
    assert live.writer.values['Train/mean_reward', 2] == 8.
    assert live.writer.values['Rollout/mean_step_reward', 2] == .75
    assert live.writer.values['Rollout/completed_episodes', 2] == 0
    assert 'no new episodes finished' in capsys.readouterr().out


def test_components_are_snapshots_and_reset_each_rollout():
    live = logger()
    components = torch.tensor([1., 10., 100., .01, -.02, .04])
    rewards = torch.tensor([2., 4.])
    live.process_env_step(rewards, torch.ones(2), {'pick_place_step': components})
    components.zero_()
    rewards.zero_()
    log_update(live, 0)
    assert live.writer.values['Rollout/mean_step_reward', 0] == 3.
    assert live.writer.values['Rollout/pickup_bonus_per_step', 0] == 10.
    live.process_env_step(torch.zeros(2), torch.zeros(2), {'pick_place_step': components})
    log_update(live, 1)
    assert live.writer.values['Rollout/pickup_bonus_per_step', 1] == 0.


def test_preview_wrapper_forwards_live_logging():
    live = logger()
    callback = Mock()
    wrapped = PreviewLogger(live, callback)
    wrapped.process_env_step(torch.ones(2), torch.zeros(2), {})
    log_update(wrapped, 0)
    callback.assert_called_once_with()
    assert live.writer.values['Rollout/mean_step_reward', 0] == 1.


def test_disabled_writer_still_forwards_steps():
    base = SimpleNamespace(writer=None, process_env_step=Mock(), log=Mock())
    live = RolloutLogger(base)
    live.process_env_step(torch.ones(2), torch.zeros(2), {})
    live.log(it=0)
    base.process_env_step.assert_called_once()
    base.log.assert_called_once_with(it=0)
    assert live._samples == 0


def test_terminal_diagnostics_survive_automatic_reset(calibration):
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from mjlab.envs import ManagerBasedRlEnv
    from so101_ppo.contract import CONTROL_DT
    from so101_ppo.pick_place_task import make_env_cfg
    cfg = make_env_cfg(calibration, num_envs=2)
    cfg.episode_length_s = CONTROL_DT
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        env.reset()
        _, reward, _, timeout, extras = env.step(torch.zeros(2, 6, device=env.device))
        assert timeout.all()
        assert not env.command_manager.get_term('goal').state.reward_components.any()
        snapshot = extras['pick_place_step']
        saved = snapshot.clone()
        assert snapshot[4] < 0  # The terminal movement penalty was not erased.
        torch.testing.assert_close(snapshot[:5].sum(), reward.mean())
        env.step(torch.ones(2, 6, device=env.device))
        torch.testing.assert_close(snapshot, saved)
    finally:
        env.close()
