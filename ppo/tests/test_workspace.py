import json
import math
import shutil

import mujoco
import numpy as np
import pytest
import torch

from so101_ppo.contract import (
    ACTION_SCALE, CONTROL_DT, HOME, JOINTS, MAX_COMMAND_SPEED, SITE,
    next_command, observation, policy_manifest,
)
from so101_ppo.cpu import CpuArm
from so101_ppo.model import cpu_spec, indices, target_bank
from so101_ppo.starts import start_bank
from so101_ppo.workspace import resolve_workspace, task_bank, workspace_config


@pytest.mark.parametrize('requested,mode,manifest,expected', [
    (None,'random',None,'wide'), (None,'fixed',None,'near'),
    (None,'random',{},'near'), (None,'random',{'workspace':'wide'},'wide'),
    ('near','random',None,'near'), ('wide','fixed',None,'wide'),
])
def test_workspace_defaults_and_legacy_inheritance(requested,mode,manifest,expected):
    assert resolve_workspace(requested,mode,manifest)==expected


def test_wide_bank_spans_workspace_and_is_reproducible(calibration):
    near, _ = target_bank('random')
    goals, q = task_bank(calibration,'random',workspace='wide')
    again = task_bank(calibration,'random',workspace='wide')
    np.testing.assert_array_equal(goals,again[0])
    np.testing.assert_array_equal(q,again[1])
    assert len(goals)==4096
    assert np.all(np.ptp(goals,axis=0) > [0.65,0.70,0.35])
    assert np.all(np.ptp(goals,axis=0) > 2*np.ptp(near,axis=0))
    assert np.all(np.rad2deg(np.ptp(q[:,:3],axis=0)) > 140)
    region=workspace_config(calibration,'wide')
    assert np.all(q[:,:3]>=region.lower) and np.all(q[:,:3]<=region.upper)
    assert np.all(q>=calibration.lower) and np.all(q<=calibration.upper)
    np.testing.assert_array_equal(q[:,3:],np.tile(HOME[3:],(len(q),1)))
    _,counts=np.unique(np.floor(goals/.01).astype(int),axis=0,return_counts=True)
    assert counts.max()<=2
    assert not goals.flags.writeable and not q.flags.writeable
    starts=start_bank(calibration,workspace='wide')
    assert len(starts)==4096
    assert not np.array_equal(q,starts)
    assert np.all(np.rad2deg(np.ptp(starts[:,:3],axis=0)) > 140)


def test_wide_geometry_paths_and_time_budget(calibration):
    goals,poses=task_bank(calibration,'random',workspace='wide')
    region=workspace_config(calibration,'wide')
    model=cpu_spec().compile(); data=mujoco.MjData(model); qi,_=indices(model)
    # Recheck a spread across the bank with twice the generator's resolution.
    for i in np.linspace(0,len(poses)-1,80,dtype=int):
        q=poses[i]
        steps=math.ceil(float(np.max(np.abs(q-HOME)))/np.deg2rad(1))+1
        for fraction in np.linspace(0,1,steps):
            data.qpos[qi]=HOME+fraction*(q-HOME)
            mujoco.mj_forward(model,data)
            assert data.ncon==0
            assert data.site(SITE).xpos[2]>=.08
        np.testing.assert_allclose(data.site(SITE).xpos,goals[i],atol=1e-7)
    maximum_home_time=np.max(np.abs(poses[:,:3]-HOME[:3]))/MAX_COMMAND_SPEED
    assert region.episode_seconds >= 2*maximum_home_time+3
    assert region.episode_steps==round(region.episode_seconds/CONTROL_DT)


def test_action_mapping_uses_full_limits_without_startup_jump(calibration):
    region=workspace_config(calibration,'wide')
    previous=HOME.copy(); previous[:3]=region.upper-.01
    next_q=next_command(-np.ones(3),previous,(region.lower,region.upper),region.center,region.scale)
    np.testing.assert_allclose(next_q[:3],previous[:3]-MAX_COMMAND_SPEED*CONTROL_DT,atol=1e-7)
    assert np.max(abs(next_q[:3]-HOME[:3])) > 1
    np.testing.assert_array_equal(next_q[3:],HOME[3:])
    for _ in range(region.episode_steps):
        previous=next_q
        next_q=next_command(-np.ones(3),previous,(region.lower,region.upper),region.center,region.scale)
        assert np.max(abs(next_q[:3]-previous[:3])) <= MAX_COMMAND_SPEED*CONTROL_DT+1e-7
    np.testing.assert_allclose(next_q[:3],region.lower,atol=1e-7)
    legacy=policy_manifest(); wide=policy_manifest(region)
    assert legacy['schema']==2 and wide['schema']==3
    assert wide['action_center_rad']==region.center.tolist()
    assert 'previous_command_minus_home_divided_by_action_scale' in legacy['observation_layout']
    assert wide['hardware_ready'] is False


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_wide_cpu_mjlab_parity_and_partial_resets(calibration,device):
    from mjlab.envs import ManagerBasedRlEnv
    from so101_ppo.task import make_env_cfg, actor_observation, arm, joint_state, tool_position
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    cfg=make_env_cfg(calibration,mode='random',start_mode='random',workspace='wide',num_envs=2)
    region=workspace_config(calibration,'wide')
    assert cfg.episode_length_s==region.episode_seconds
    env=ManagerBasedRlEnv(cfg,device=device)
    try:
        env.reset(); q,dq=joint_state(env)
        goals=env.command_manager.get_command('goal')
        expected=observation(q.cpu().numpy(),dq.cpu().numpy(),goals.cpu().numpy(),
            tool_position(env).cpu().numpy(),arm(env).command.cpu().numpy(),region.center,region.scale)
        np.testing.assert_allclose(actor_observation(env).cpu().numpy(),expected,atol=1e-6)
        mirrors=[CpuArm(calibration,workspace='wide') for _ in range(2)]
        for i,m in enumerate(mirrors): m.reset(goals[i].cpu().numpy(),q[i].cpu().numpy())
        actions=torch.tensor([[.8,-.6,-.7],[-.8,.6,.7]],device=device)
        for _ in range(8):
            obs,reward,failed,_,_=env.step(actions)
            assert not failed.any()
            q,_=joint_state(env)
            for i,m in enumerate(mirrors):
                m.step(actions[i].cpu().numpy())
                np.testing.assert_allclose(arm(env).command[i].cpu().numpy(),m.command,atol=1e-6)
                np.testing.assert_allclose(q[i].cpu().numpy(),m.q,atol=5e-5)
        keep=arm(env).command[1].clone()
        env.reset(env_ids=torch.tensor([0],device=device))
        torch.testing.assert_close(arm(env).command[1],keep,rtol=0,atol=0)
        assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
    finally: env.close()


def test_load_rejects_inconsistent_wide_contract_before_onnx(calibration,tmp_path):
    from so101_ppo.evaluate import load_policy
    bundle=tmp_path/'calibration'; shutil.copytree(calibration.root,bundle)
    region=workspace_config(calibration,'wide')
    manifest=policy_manifest(region)
    manifest['calibration']={'bundle_path':'calibration','bundle_sha256':calibration.digest}
    manifest['action_scale_rad']=ACTION_SCALE.tolist()
    (tmp_path/'policy_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='contract mismatch: action_scale_rad'):
        load_policy(tmp_path/'policy.onnx',allow_synthetic=True)


def test_wide_preview_changes_hand_and_target_with_fixed_comparison(calibration,tmp_path,monkeypatch):
    from so101_ppo.visualization import TrainingVisualization, overlay_text
    def prepare(self,calibration,task_mode):
        self.robot=CpuArm(calibration,workspace=self.workspace.name)
        self._prepare_targets(task_mode); self._prepare_starts(calibration)
    monkeypatch.setattr(TrainingVisualization,'_prepare',prepare)
    for selection in ('rotate','fixed'):
        p=TrainingVisualization(calibration,tmp_path/selection,'save',task_mode='random',
            start_mode='random',workspace='wide',target_selection=selection)
        for _ in range(5):
            target=p.goal.copy(); start=p._start_goals[p.start_index].copy()
            p._select_next_target(); p._select_next_start()
            if selection=='rotate':
                assert np.linalg.norm(p.goal-target)>=.15
                assert np.linalg.norm(p._start_goals[p.start_index]-start)>=.15
            else:
                np.testing.assert_array_equal(p.goal,target)
                np.testing.assert_array_equal(p._start_goals[p.start_index],start)
        state={'step':200,'episode_steps':p.robot.episode_steps,'time_s':4.,'distance_m':.3,'success':False,'failed':False}
        assert 'TIMEOUT' not in overlay_text(0,state)
        state['step']=p.robot.episode_steps
        assert 'TIMEOUT' in overlay_text(0,state)
        p.close()


def test_wide_recording_header_does_not_occlude_scene():
    from types import SimpleNamespace
    from so101_ppo.visualization import TrainingVisualization, HEIGHT, WIDTH, WIDE_HEADER_HEIGHT
    p=TrainingVisualization.__new__(TrainingVisualization)
    p.workspace=SimpleNamespace(name='wide')
    scene=np.full((HEIGHT-WIDE_HEADER_HEIGHT,WIDTH,3),127,dtype=np.uint8)
    frame=p._recording_frame(scene)
    assert frame.shape==(HEIGHT,WIDTH,3)
    assert not frame[:WIDE_HEADER_HEIGHT].any()
    np.testing.assert_array_equal(frame[WIDE_HEADER_HEIGHT:],scene)
