"""Batched mjlab pick-and-place task with real contact and six servo actions."""
from dataclasses import dataclass
from functools import partial

import torch
from mjlab.entity import EntityCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from .contract import CONTROL_DT, JOINTS, MAX_COMMAND_SPEED, SITE
from .pick_place import CONFIG, TaskState, action_bounds, brick_spec, robot_spec
from .task import ArmAction, ArmActionCfg, make_env_cfg as reach_env_cfg


class ManipulationAction(ArmAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.home = torch.tensor(CONFIG.start_q, device=self.device)
        self.command[:] = self.home
        self._raw = torch.zeros(self.num_envs, 6, device=self.device)
        self.delta = torch.zeros_like(self._raw)

    @property
    def action_dim(self):
        return 6

    def process_actions(self, actions):
        self._raw[:] = actions
        desired = self.center + self.scale*actions.clamp(-1, 1)
        maximum = MAX_COMMAND_SPEED*CONTROL_DT
        self.delta[:] = (desired-self.command).clamp(-maximum, maximum)
        previous = self.command.clone()
        self.command[:] = (self.command+self.delta).clamp(self.lower, self.upper)
        self.delta[:] = self.command-previous

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.delta[ids] = 0
        self.command[ids] = self._entity.data.joint_pos[ids][:, self.ids]
        target = self.quantized_command()
        for i, actuator in enumerate(self._entity.actuators):
            actuator.reset_target_history(ids, target[:, i:i+1])
        self.apply_actions()


@dataclass(kw_only=True)
class ManipulationActionCfg(ArmActionCfg):
    def build(self, env):
        return ManipulationAction(self, env)


class PlacementCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.goal = torch.tensor(CONFIG.target_position, device=self.device).repeat(self.num_envs, 1)
        self.site_id = env.scene['robot'].find_sites((SITE,))[0][0]
        self.state = TaskState(self.num_envs, self.device)
        self.last_step = -1
        for name in ('touch_rate', 'pickup_rate', 'episode_success', 'distance_m', 'settle_s'):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self.goal

    def _resample_command(self, env_ids):
        self.state.reset(env_ids)

    def _update_metrics(self):
        pass  # Capture terminal metrics before auto-reset in task_done.

    def _update_command(self):
        pass


@dataclass(kw_only=True)
class PlacementCommandCfg(CommandTermCfg):
    def build(self, env):
        return PlacementCommand(self, env)


def inputs(env):
    action = env.action_manager.get_term('arm')
    command = env.command_manager.get_term('goal')
    robot = env.scene['robot'].data
    brick = env.scene['brick'].data
    base = robot.root_link_pos_w
    # Fixed base orientation is identity in every world; translations may differ.
    contacts = torch.stack([
        ((env.scene[name].data.found > 0) & (env.scene[name].data.dist < 0)).any(-1)
        for name in ('fixed_contact', 'moving_contact', 'robot_contact', 'floor_contact')
    ], -1)
    return (robot.joint_pos[:, action.ids], robot.joint_vel[:, action.ids],
            brick.root_link_pos_w-base, brick.root_link_quat_w,
            torch.cat((brick.root_link_lin_vel_w, brick.root_link_ang_vel_w), -1),
            robot.site_pos_w[:, command.site_id]-base, command.goal, contacts)


def observation(env):
    q, dq, pos, quat, vel, tool, target, contacts = inputs(env)
    action = env.action_manager.get_term('arm')
    return env.command_manager.get_term('goal').state.observation(q, dq, action.command,
        action.lower, action.upper, pos, quat, vel, tool, target, contacts)


def task_done(env):
    command = env.command_manager.get_term('goal')
    state = command.state
    if command.last_step != env.common_step_counter:
        state.advance(*inputs(env), env.action_manager.get_term('arm').delta)
        command.last_step = env.common_step_counter
        for i, name in enumerate(('touch_rate', 'pickup_rate', 'episode_success')):
            command.metrics[name][:] = state.milestones[:, i].float()
        pos = env.scene['brick'].data.root_link_pos_w-env.scene['robot'].data.root_link_pos_w
        command.metrics['distance_m'][:] = torch.linalg.vector_norm(pos[:, :2]-command.goal[:, :2], dim=-1)
        command.metrics['settle_s'][:] = state.holds[:, 1]*CONTROL_DT
    return state.failed | state.milestones[:, 2]


def reward(env):
    command = env.command_manager.get_term('goal')
    robot, brick = env.scene['robot'].data, env.scene['brick'].data
    distance = torch.linalg.vector_norm(
        brick.root_link_pos_w-robot.site_pos_w[:, command.site_id], dim=-1).mean()
    # Snapshot before automatic resets erase terminal rewards and milestone state.
    env.extras['pick_place_step'] = torch.cat((command.state.reward_components.mean(0), distance[None])).detach()
    return command.state.reward


def reset_scene(env, env_ids):
    robot = env.scene['robot']
    ids, _ = robot.find_joints(JOINTS, preserve_order=True)
    q = torch.tensor(CONFIG.start_q, device=env.device).repeat(len(env_ids), 1)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)
    pose = torch.tensor((*CONFIG.brick_position, *CONFIG.brick_quaternion), device=env.device).repeat(len(env_ids), 1)
    pose[:, :3] += env.scene.env_origins[env_ids]
    env.scene['brick'].write_root_link_pose_to_sim(pose, env_ids=env_ids)
    env.scene['brick'].write_root_link_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def make_env_cfg(calibration, num_envs=1024, seed=42):
    cfg = reach_env_cfg(calibration, num_envs=num_envs, seed=seed)
    lower, upper = action_bounds(calibration)
    robot = cfg.scene.entities['robot']
    robot.spec_fn = partial(robot_spec, calibration)
    robot.init_state = EntityCfg.InitialStateCfg(joint_pos=dict(zip(JOINTS, CONFIG.start_q)), joint_vel={'.*': 0.})
    cfg.scene.entities['brick'] = EntityCfg(spec_fn=brick_spec,
        init_state=EntityCfg.InitialStateCfg(pos=CONFIG.brick_position, rot=CONFIG.brick_quaternion))
    cfg.scene.sensors = tuple(ContactSensorCfg(name=name,
        primary=ContactMatch(mode=mode, pattern=pattern, entity='robot'),
        secondary=ContactMatch(mode='geom', pattern='brick_geom', entity='brick'),
        fields=('found', 'dist'), reduce='mindist', secondary_policy='error') for name, mode, pattern in (
            ('fixed_contact', 'geom', 'fixed_finger_.*'), ('moving_contact', 'geom', 'moving_finger_.*'),
            ('robot_contact', 'subtree', 'base'), ('floor_contact', 'geom', 'floor')))
    cfg.actions = {'arm': ManipulationActionCfg(entity_name='robot',
        lower=tuple(lower), upper=tuple(upper), center=tuple((lower+upper)/2), scale=tuple((upper-lower)/2),
        mid=tuple(calibration.mid), sign=tuple(calibration.sign), offset=tuple(calibration.offset))}
    cfg.commands = {'goal': PlacementCommandCfg(resampling_time_range=(1e9, 1e9))}
    cfg.events['reset_robot'] = EventTermCfg(func=reset_scene, mode='reset')
    terms = {'state': ObservationTermCfg(func=observation)}
    cfg.observations = {name: ObservationGroupCfg(dict(terms), enable_corruption=False) for name in ('actor', 'critic')}
    cfg.terminations['failure'] = TerminationTermCfg(func=task_done)
    cfg.rewards = {'pick_place': RewardTermCfg(func=reward, weight=1.)}
    cfg.scale_rewards_by_dt = False  # Milestones are transition bonuses, not reward rates.
    cfg.episode_length_s = CONFIG.episode_seconds
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.iterations = 100
    cfg.sim.nconmax = 256
    cfg.sim.njmax = 1024
    cfg.viewer.lookat = (.22, .045, .08)
    cfg.viewer.distance = .65
    cfg.viewer.elevation = -35
    return cfg
