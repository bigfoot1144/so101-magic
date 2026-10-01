"""Fresh rollout diagnostics alongside RSL-RL's completed-episode statistics."""

PICK_PLACE_METRICS = (
    'touch_bonus_per_step', 'pickup_bonus_per_step', 'placement_bonus_per_step',
    'shaping_reward_per_step', 'motion_penalty_per_step', 'tool_to_brick_distance_m',
)


class RolloutLogger:
    """Delegate episode accounting unchanged; report new samples every PPO update.

    With 3,000-step fixed episodes and 32-step rollouts, completed-episode means
    can stay unchanged for 94 updates. Averaging current transitions provides a
    separate live measurement, including before the first episode finishes.
    """
    def __init__(self, logger):
        self._logger = logger
        self._reward_sum = self._done_count = self._diagnostics = None
        self._samples = self._diagnostic_steps = 0

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def process_env_step(self, rewards, dones, extras, intrinsic_rewards=None):
        self._logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
        if self._logger.writer is None:
            return
        reward_sum = rewards.detach().sum()
        done_count = (dones.detach() > 0).sum()
        if self._reward_sum is None:
            self._reward_sum, self._done_count = reward_sum, done_count
        else:
            self._reward_sum += reward_sum
            self._done_count += done_count
        self._samples += rewards.numel()
        diagnostics = extras.get('pick_place_step')
        if diagnostics is not None:
            # Do not retain a view of mutable environment buffers.
            if self._diagnostics is None:
                self._diagnostics = diagnostics.detach().clone()
            else:
                self._diagnostics += diagnostics.detach()
            self._diagnostic_steps += 1

    def log(self, *args, **kwargs):
        result = self._logger.log(*args, **kwargs)
        if not self._samples:
            return result
        iteration = kwargs['it'] if 'it' in kwargs else args[0]
        writer = self._logger.writer
        mean_reward = (self._reward_sum/self._samples).item()
        completed = int(self._done_count.item())
        writer.add_scalar('Rollout/mean_step_reward', mean_reward, iteration)
        writer.add_scalar('Rollout/completed_episodes', completed, iteration)
        print(f"{'Rollout mean step reward:':>40} {mean_reward:.6f}")
        print(f"{'Completed episodes this rollout:':>40} {completed}")
        if completed == 0:
            print('  Mean reward above uses completed episodes; no new episodes finished this rollout.')
        if self._diagnostics is not None:
            values = (self._diagnostics/self._diagnostic_steps).cpu().tolist()
            for name, value in zip(PICK_PLACE_METRICS, values, strict=True):
                writer.add_scalar(f'Rollout/{name}', value, iteration)
            touch, pickup, place, shaping, penalty, distance = values
            print(f"  Reward per step: touch={touch:.6f}, pickup={pickup:.6f}, "
                  f"place={place:.6f}, shaping={shaping:.6f}, motion={penalty:.6f}")
            print(f"{'Rollout mean tool-to-brick distance:':>40} {distance*1000:.2f} mm")
        self._reward_sum = self._done_count = self._diagnostics = None
        self._samples = self._diagnostic_steps = 0
        return result
