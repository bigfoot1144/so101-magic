"""A fixed-base SO-101 reaching MDP for mjlab 1.3.0."""

from dataclasses import dataclass
from functools import partial
import torch
from bam.mjlab import bam_init
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.viewer import ViewerConfig

from .actuators import CalibratedBamActuatorCfg
from .calibration import RAD_PER_TICK, register_12v
from .contract import (
    ACTION_SCALE, CONTROL_DT, DECIMATION, EPISODE_SECONDS, HOLD_STEPS, HOME,
    JOINTS, MAX_COMMAND_SPEED, PHYSICS_DT, SITE, SUCCESS_DISTANCE, SUCCESS_SPEED,
)
from .model import robot_spec, target_bank, validate_geometry


class ArmAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids, names = self._entity.find_joints(JOINTS, preserve_order=True)
        if tuple(names) != JOINTS:
            raise RuntimeError(f"Joint mapping mismatch: {names}")
        self.ids = torch.tensor(ids, device=self.device)
        self.home = torch.tensor(HOME, device=self.device)
        self.scale = torch.tensor(ACTION_SCALE, device=self.device)
        self.lower = torch.tensor(cfg.lower, device=self.device)
        self.upper = torch.tensor(cfg.upper, device=self.device)
        self.mid = torch.tensor(cfg.mid, dtype=torch.float64, device=self.device)
        self.sign = torch.tensor(cfg.sign, dtype=torch.float64, device=self.device)
        self.offset = torch.tensor(cfg.offset, dtype=torch.float64, device=self.device)
        self._raw = torch.zeros(self.num_envs, 3, device=self.device)
        self.command = self.home.repeat(self.num_envs, 1)
        self.delta = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self):
        return 3

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions
        desired = self.home[:3] + self.scale * actions.clamp(-1, 1)
        maximum = MAX_COMMAND_SPEED * CONTROL_DT
        self.delta[:] = (desired - self.command[:, :3]).clamp(-maximum, maximum)
        self.command[:, :3] += self.delta
        self.command[:, :3].clamp_(self.home[:3] - self.scale, self.home[:3] + self.scale)
        self.command[:, :3].clamp_(self.lower, self.upper)

    def quantized_command(self):
        raw = torch.round((self.command.double() - self.offset) / self.sign / RAD_PER_TICK + self.mid)
        return ((raw - self.mid) * RAD_PER_TICK * self.sign + self.offset).float()

    def apply_actions(self):
        # The three remaining servos hold HOME with finite BAM actuator torque.
        self._entity.set_joint_position_target(self.quantized_command(), joint_ids=self.ids)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw[env_ids] = 0
        self.delta[env_ids] = 0
        self.command[env_ids] = self.home
        self.command[env_ids, :3] = self._entity.data.joint_pos[env_ids][:, self.ids[:3]]
        target = self.quantized_command()
        for i, actuator in enumerate(self._entity.actuators):
            actuator.reset_target_history(env_ids, target[:, i:i+1])
        self.apply_actions()


@dataclass(kw_only=True)
class ArmActionCfg(ActionTermCfg):
    lower: tuple = tuple(HOME[:3] - ACTION_SCALE)
    upper: tuple = tuple(HOME[:3] + ACTION_SCALE)
    mid: tuple = (2047.5,) * 6
    sign: tuple = (1.,) * 6
    offset: tuple = (0.,) * 6

    def build(self, env):
        return ArmAction(self, env)


def arm(env):
    return env.action_manager.get_term("arm")


def joint_state(env):
    action = arm(env)
    data = env.scene["robot"].data
    return data.joint_pos[:, action.ids], data.joint_vel[:, action.ids]


def tool_position(env):
    data = env.scene["robot"].data
    site_id = env.command_manager.get_term("goal").site_id
    # Compute in the actual base frame, including any per-world translation.
    return quat_apply_inverse(
        data.root_link_quat_w,
        data.site_pos_w[:, site_id] - data.root_link_pos_w,
    )


class GoalCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ids, _ = env.scene["robot"].find_sites((SITE,))
        self.site_id = ids[0]
        self.bank = torch.tensor(target_bank(cfg.mode, cfg.bank_seed)[0], device=self.device)
        self.goal = self.bank[:1].repeat(self.num_envs, 1)
        self.hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.succeeded = torch.zeros(self.num_envs, device=self.device)
        for name in ("distance_m", "episode_success", "hold_s"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self.goal

    def _resample_command(self, env_ids):
        sample = torch.randint(len(self.bank), (len(env_ids),), device=self.device)
        self.goal[env_ids] = self.bank[sample]
        self.hold[env_ids] = 0
        self.succeeded[env_ids] = 0

    def _update_metrics(self):
        # Metrics are recorded in task_failure before auto-reset, so the final
        # transition is counted and a partial reset cannot erase its success.
        pass

    def _update_command(self):
        pass

    def _debug_vis_impl(self, visualizer):
        data = self._env.scene["robot"].data
        # This task's fixed base has identity rotation.
        for i in visualizer.get_env_indices(self.num_envs):
            world = self.goal[i] + data.root_link_pos_w[i]
            visualizer.add_sphere(center=world.cpu().numpy(), radius=SUCCESS_DISTANCE,
                                  color=(0.2, 0.85, 0.3, 0.7), label=f"goal_{i}")


@dataclass(kw_only=True)
class GoalCommandCfg(CommandTermCfg):
    mode: str = "fixed"
    bank_seed: int = 12345

    def build(self, env):
        return GoalCommand(self, env)


def reset_robot(env, env_ids, lower, upper, spread=0.02):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(JOINTS, preserve_order=True)
    q = torch.tensor(HOME, device=env.device).repeat(len(env_ids), 1)
    q[:, :3] += torch.empty(len(env_ids), 3, device=env.device).uniform_(-spread, spread)
    q[:, :3].clamp_(torch.tensor(lower, device=env.device), torch.tensor(upper, device=env.device))
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)


def randomize_gains(env, env_ids=None):
    for actuator in env.scene["robot"].actuators:
        scale = torch.empty(env.num_envs, 1, device=env.device).uniform_(0.9, 1.1)
        actuator.kp_scale[:] = scale
        actuator.default_kp_scale[:] = scale


def actor_observation(env):
    q, dq = joint_state(env)
    action = arm(env)
    goal = env.command_manager.get_command("goal")
    return torch.cat((
        q - action.home, dq * 0.1, goal, goal - tool_position(env),
        (action.command[:, :3] - action.home[:3]) / action.scale,
    ), dim=-1)


def reaching_reward(env):
    _, dq = joint_state(env)
    distance = torch.linalg.vector_norm(
        env.command_manager.get_command("goal") - tool_position(env), dim=-1
    )
    rate = arm(env).delta / (CONTROL_DT * MAX_COMMAND_SPEED)
    return (2 * (1 - torch.tanh(distance / 0.10))
            + torch.exp(-0.5 * (distance / 0.02).square())
            - 0.01 * dq.square().sum(-1) - 0.02 * rate.square().sum(-1))


def task_failure(env):
    q, dq = joint_state(env)
    tool = tool_position(env)
    goal = env.command_manager.get_term("goal")
    distance = torch.linalg.vector_norm(goal.goal - tool, dim=-1)
    at_goal = ((distance < SUCCESS_DISTANCE)
               & (dq.abs().amax(-1) < SUCCESS_SPEED))
    goal.hold[:] = torch.where(at_goal, goal.hold + 1, 0)
    goal.succeeded[:] = torch.maximum(goal.succeeded, (goal.hold >= HOLD_STEPS).float())
    goal.metrics["distance_m"][:] = distance
    goal.metrics["episode_success"][:] = goal.succeeded
    goal.metrics["hold_s"][:] = goal.hold * CONTROL_DT
    finite = torch.isfinite(q).all(-1) & torch.isfinite(dq).all(-1)
    return (~finite) | (tool[:, 2] < 0.06) | (dq.abs().amax(-1) > 20)


def make_env_cfg(calibration, mode="fixed", num_envs=1024, seed=42,
                 robust=False, bank_seed=12345, replay=False, voltage=None):
    register_12v()
    source_model = validate_geometry(calibration)
    lower, upper = calibration.task_bounds()
    volts = calibration.voltage if voltage is None else voltage
    actuators = tuple(CalibratedBamActuatorCfg(
        json_path=path, target_names_expr=(name,), vin=float(volts[i]), kp_fw=16,
        max_pwm=float(calibration.max_pwm[i]),
        command_delay_s=0. if replay else float(calibration.delays[i]),
        vin_range=(float(volts[i]) * .95, float(volts[i]) * 1.05) if robust else None,
        friction_scale_range=(.9, 1.1) if robust else None,
    ) for i, (name, path) in enumerate(zip(JOINTS, calibration.paths, strict=True)))
    robot = EntityCfg(
        spec_fn=partial(robot_spec, calibration),
        init_state=EntityCfg.InitialStateCfg(joint_pos=dict(zip(JOINTS, HOME.tolist())),
                                            joint_vel={".*": 0.0}),
        articulation=EntityArticulationInfoCfg(actuators=actuators),
    )
    events = {
        "bam_init": EventTermCfg(func=bam_init, mode="startup"),
        "reset_robot": EventTermCfg(func=reset_robot, mode="reset", params={"lower": lower, "upper": upper}),
    }
    if robust:
        events["gain_randomization"] = EventTermCfg(func=randomize_gains, mode="startup")
    terms = {"state": ObservationTermCfg(func=actor_observation)}
    return ManagerBasedRlEnvCfg(
        seed=seed, decimation=DECIMATION, episode_length_s=EPISODE_SECONDS,
        scene=SceneCfg(num_envs=num_envs, env_spacing=1, entities={"robot": robot}),
        sim=SimulationCfg(nconmax=128, njmax=256, mujoco=MujocoCfg(
            timestep=PHYSICS_DT, integrator="implicitfast", iterations=int(source_model.opt.iterations),
            ls_iterations=int(source_model.opt.ls_iterations), tolerance=float(source_model.opt.tolerance))),
        actions={"arm": ArmActionCfg(entity_name="robot", lower=tuple(lower), upper=tuple(upper),
                    mid=tuple(calibration.mid), sign=tuple(calibration.sign), offset=tuple(calibration.offset))},
        commands={"goal": GoalCommandCfg(mode=mode, bank_seed=bank_seed,
                    resampling_time_range=(1e9, 1e9), debug_vis=True)},
        observations={"actor": ObservationGroupCfg(terms, enable_corruption=False),
                      "critic": ObservationGroupCfg(dict(terms), enable_corruption=False)},
        events=events,
        rewards={"reach": RewardTermCfg(func=reaching_reward, weight=1)},
        terminations={"failure": TerminationTermCfg(func=task_failure),
                      "timeout": TerminationTermCfg(func=time_out, time_out=True)},
        viewer=ViewerConfig(lookat=(0.20, 0, 0.16), distance=0.75, elevation=-25,
                            azimuth=135, origin_type=ViewerConfig.OriginType.WORLD),
    )


def runner_cfg():
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=32, max_iterations=500, save_interval=50,
        experiment_name="so101_reach", logger="tensorboard", upload_model=False,
        actor=RslRlModelCfg(hidden_dims=(128, 128), activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5,
                              "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(128, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(learning_rate=3e-4, num_learning_epochs=5,
            num_mini_batches=4, entropy_coef=0.002, gamma=0.99, lam=0.95,
            clip_param=0.2, desired_kl=0.01),
    )
