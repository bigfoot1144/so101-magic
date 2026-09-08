#!/usr/bin/env python3
"""Build explicitly SYNTHETIC records and a bundle for installation tests only.

This does not fit motors and must never be used as a physical calibration.
Run with the calibration Python environment (MuJoCo 3.12).
"""
import argparse
import copy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import so101_sysid as s
from so101_calibration_core import snapshot_model, model_digest
from export_for_ppo import export

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ppo' / 'src'))
from so101_ppo.contract import HOME


def make_demo(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    stock = Path(__file__).resolve().parents[1] / 'ppo/src/so101_ppo/assets/so101/so101.xml'
    tree = ET.parse(stock)
    tree.getroot().find('compiler').set('meshdir', str(stock.parent / 'assets'))
    for default in tree.findall('.//default'):
        for node in list(default):
            if node.tag in ('position', 'velocity', 'general', 'motor', 'intvelocity', 'damper'):
                default.remove(node)
    actuator = tree.getroot().find('actuator')
    actuator.clear()
    for j in s.JOINTS:
        ET.SubElement(actuator, 'motor', name=j, joint=j, gear='1', ctrllimited='false', forcelimited='false')
    option = tree.getroot().find('option')
    if option is None:
        option = ET.SubElement(tree.getroot(), 'option')
    option.set('timestep', '.002')
    option.set('integrator', 'implicitfast')
    tree.write(root / 'source.xml')
    cfg = {'schema': 1, 'synthetic': True, 'mapping_reviewed': False, 'free_space': True,
           'so101_commit': s.SO_SHA, 'bam_commit': s.BAM_SHA,
           'calibration': {j: {'id': i+1, 'drive_mode': 0, 'homing_offset': 0, 'range_min': 0, 'range_max': 4095}
                           for i, j in enumerate(s.JOINTS)},
           'mapping': {j: {'sign': 1 if i % 2 == 0 else -1, 'offset_rad': [.05, -.1, .02, -.03, .07, .1][i]}
                       for i, j in enumerate(s.JOINTS)}}
    session = root / 'session'
    cfg = snapshot_model(cfg, root / 'source.xml', session)
    (root / 'source.xml').unlink()
    xml = session / cfg['xml']
    params = {'schema': 1, 'joints': {}}
    delays = [0., .007, .02, .034, .05, .08]
    for i, j in enumerate(s.JOINTS):
        motor = copy.deepcopy(s.SEED)
        motor['R'] *= 1 + i * .015
        params['joints'][j] = {'bam': motor, 'command_delay_s': delays[i]}
    settings = {j: {k: 0 for k in s.REG} for j in s.JOINTS}
    for i, j in enumerate(s.JOINTS):
        settings[j].update(p=16, i=0, d=0, max_torque=1000, torque_limit=850+i*20,
                           acceleration=254, max_acceleration=254, goal_velocity=0)
    run = session / 'runs' / 'synthetic'
    run.mkdir(parents=True)
    manifest = {'complete': True, 'synthetic': True, 'logs': []}
    raw0 = s.sim_to_raw(HOME, cfg)
    q0 = s.raw_to_sim(raw0, cfg)
    for index, role in enumerate(('train', 'train', 'validation')):
        log = {'schema': 1, 'complete': True, 'synthetic': True, 'config': cfg, 'joint': 'all',
               'role': role, 'command_rate_hz': 50, 'p': 16, 'i': 0, 'd': 0,
               'effective_settings': settings, 'initial_target_raw': raw0.tolist(),
               'initial_voltage_v': [11.8, 12., 12.1, 11.9, 12., 11.7],
               'initial_velocity_rad_s': [0.] * 6, 'samples': [], 'commands': []}
        for t in np.arange(0., .801, .02):
            target = q0 + .012 * np.sin((3 + index) * np.pi * t + np.arange(6) * .2)
            target = s.raw_to_sim(s.sim_to_raw(target, cfg), cfg)
            log['commands'].append({'t': float(t), 'q_target_rad': target.tolist()})
            log['samples'].append({'t': float(t), 'q_rad': q0.tolist(), 'voltage_v': log['initial_voltage_v']})
        measured = s.rollout(xml, cfg, params, log)
        for row, q in zip(log['samples'], measured):
            row['q_rad'] = q.tolist()
        filename = f'{index}.json'
        s.save_json(run / filename, log)
        manifest['logs'].append({'file': filename, 'role': role, 'complete': True})
    s.save_json(run / 'manifest.json', manifest)
    params['calibration_interface_fit'] = {'synthetic': True, 'p': 16, 'i': 0, 'd': 0,
        'mapping': cfg['mapping'], 'calibration': cfg['calibration'], 'model_sha256': model_digest(xml),
        'training_logs': [str(run / '0.json'), str(run / '1.json')],
        'parameter_interpretation': 'UNFITTED synthetic software fixture; not measurements of hardware'}
    s.save_json(root / 'params.json', params)
    return export([session], root / 'params.json', root / 'bundle', allow_synthetic=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(make_demo(parser.parse_args().out))
