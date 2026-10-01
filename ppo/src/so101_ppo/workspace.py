"""Versioned reaching regions, including the legacy policy action convention."""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from .contract import ACTION_SCALE, CONTROL_DT, EPISODE_SECONDS, HOME, JOINTS, MAX_COMMAND_SPEED, SITE
from .model import cpu_spec, indices, target_bank

WIDE_BANK_SIZE = 4096


def resolve_workspace(requested=None, mode="fixed", manifest=None):
    # A missing field in a saved policy always denotes the original near task.
    value = requested if requested is not None else (
        manifest.get("workspace", "near") if manifest is not None else
        "wide" if mode == "random" else "near")
    if value not in ("near", "wide"):
        raise ValueError("Expected near or wide workspace")
    return value


@dataclass
class Workspace:
    name: str
    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    episode_seconds: float
    reward_distance_scale: float

    @property
    def episode_steps(self):
        return round(self.episode_seconds / CONTROL_DT)

    def metadata(self):
        return {"workspace": self.name, "workspace_sampler_version": 1,
                "episode_seconds": self.episode_seconds,
                "reward_distance_scale_m": self.reward_distance_scale}


def workspace_config(calibration, name="near"):
    name = resolve_workspace(name)
    if name == "near":
        lower, upper = calibration.task_bounds()
        return Workspace(name, lower, upper, HOME[:3].copy(), ACTION_SCALE.copy(), EPISODE_SECONDS, .10)
    model = cpu_spec().compile()
    ids = [model.joint(j).id for j in JOINTS]
    margin = np.deg2rad(3.)
    lower = np.maximum(calibration.lower, model.jnt_range[ids, 0] + margin)
    upper = np.minimum(calibration.upper, model.jnt_range[ids, 1] - margin)
    if np.any(lower >= upper) or np.any(HOME < lower) or np.any(HOME > upper):
        raise ValueError("Wide workspace must contain HOME within calibrated/model limits and margins")
    lower, upper = lower[:3].astype(np.float32), upper[:3].astype(np.float32)
    # Enough command travel time for any accepted start -> HOME -> goal, plus
    # two seconds to settle and the existing one-second success hold.
    radius = np.max(np.maximum(np.abs(lower - HOME[:3]), np.abs(upper - HOME[:3])))
    seconds = float(math.ceil(2 * radius / MAX_COMMAND_SPEED + 3))
    return Workspace(name, lower, upper, (lower + upper) / 2, (upper - lower) / 2, seconds, .35)


def task_bank(calibration, mode="fixed", seed=12345, workspace="near"):
    workspace = resolve_workspace(workspace)
    if workspace == "near" or mode == "fixed":
        return target_bank(mode, seed=seed)
    if mode != "random":
        raise ValueError(mode)
    cfg = workspace_config(calibration, workspace)
    return wide_bank(tuple(cfg.lower), tuple(cfg.upper), seed)


@lru_cache(maxsize=8)
def wide_bank(lower, upper, seed, size=WIDE_BANK_SIZE):
    """Broad joint-space sampling, spatially thinned to spread tool positions.

    No home-distance shell. Each pose and a home-to-pose route are checked on
    collision-enabled reference geometry at at most two-degree joint intervals.
    The 1 cm voxel cap avoids filling the bank with nearly identical tool XYZs.
    This is sampled geometric validation, not physical deployment certification.
    """
    import mujoco
    model = cpu_spec().compile()
    data = mujoco.MjData(model)
    qi, _ = indices(model)
    rng = np.random.default_rng(seed)
    goals, solutions, occupied = [], [], {}
    for _ in range(size * 200):
        q = HOME.copy()
        q[:3] = rng.uniform(lower, upper)
        data.qpos[qi] = q
        mujoco.mj_forward(model, data)
        goal = data.site(SITE).xpos.copy()
        if data.ncon or goal[2] < .08:
            continue
        voxel = tuple(np.floor(goal / .01).astype(int))
        if occupied.get(voxel, 0) >= 2:
            continue
        steps = max(2, math.ceil(float(np.max(np.abs(q - HOME))) / np.deg2rad(2)) + 1)
        valid = True
        for fraction in np.linspace(0, 1, steps):
            data.qpos[qi] = HOME + fraction * (q - HOME)
            mujoco.mj_forward(model, data)
            if data.ncon or data.site(SITE).xpos[2] < .08:
                valid = False
                break
        if not valid:
            continue
        goals.append(goal)
        solutions.append(q)
        occupied[voxel] = occupied.get(voxel, 0) + 1
        if len(goals) == size:
            result = np.asarray(goals, np.float32), np.asarray(solutions, np.float32)
            for array in result:
                array.setflags(write=False)
            return result
    raise RuntimeError("Could not generate a broad collision-free workspace bank within the calibrated limits")
