"""Evaluation reporting and native viewing for the fixed manipulation task."""
import csv
import json
from pathlib import Path
import time

import numpy as np

from .contract import CONTROL_DT
from .pick_place import CONFIG
from .pick_place_cpu import CpuPickPlace, run_episode


def evaluate(calibration, policy, episodes, seed, viewer, baseline, output, policy_path=None):
    robot = CpuPickPlace(calibration)
    rng = np.random.default_rng(seed)
    rows = []
    view = None
    if viewer:
        import mujoco.viewer
        view = mujoco.viewer.launch_passive(robot.model, robot.data)
        view.cam.lookat[:] = [.22, .045, .08]
        view.cam.distance, view.cam.elevation, view.cam.azimuth = .65, -35, 135
    try:
        for episode in range(episodes):
            if policy is not None:
                act = policy
            elif baseline == 'oracle':
                from .pick_place_scripted import ScriptedPolicy
                act = ScriptedPolicy(robot)
            elif baseline == 'random':
                act = lambda obs: rng.uniform(-1, 1, 6)
            else:
                act = lambda obs: (robot.home-robot.center)/robot.scale

            def display(robot, state):
                if not view.is_running():
                    return False
                view.set_texts((None, None, f"Touch: {state['touched']}  Pickup: {state['picked']}  Placed: {state['placed']}", ""))
                view.sync()
                time.sleep(max(0, CONTROL_DT-(time.monotonic()-state['wall_step_start'])))

            result = run_episode(robot, act, on_step=display if view else None)
            if result is None:
                return {'viewer_closed': True, 'completed_episodes': len(rows)}
            rows.append({'episode': episode, **result})
    finally:
        if view is not None:
            view.close()
    result = dict(backend='CPU MuJoCo + BAM; imported nominal calibration', task='pick-place', task_mode='fixed',
        hardware_evaluation=False, synthetic_calibration=calibration.document['synthetic'],
        calibration_sha256=calibration.digest, policy=str(policy_path) if policy_path else baseline,
        episodes=episodes, seed=seed, brick_position_m=list(CONFIG.brick_position),
        target_position_m=list(CONFIG.target_position),
        mean_return=float(np.mean([r['return'] for r in rows])),
        mean_final_distance_m=float(np.mean([r['final_distance_m'] for r in rows])))
    for key, field in (('touch_rate', 'touched'), ('pickup_rate', 'picked'),
                       ('success_rate', 'success'), ('failure_rate', 'failed')):
        result[key] = float(np.mean([r[field] for r in rows]))
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+'\n')
        with output.with_suffix('.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result
