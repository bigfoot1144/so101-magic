"""Official SO-101 geometry and a collision-checked reachable target bank."""

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .contract import ACTION_SCALE, HOME, JOINTS, PHYSICS_DT, SITE

ASSETS = Path(__file__).parent / "assets"
XML = ASSETS / "so101" / "so101.xml"
ROBOT_REVISION = "eecbe3e0a9ebb23e25ad7b2759b03884c6660903"
BAM_REVISION = "620a64fe67c1afe94fca81da73b128c7aed17c5f"


def add_floor(spec):
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1, 1, 0.01], rgba=[0.18, 0.20, 0.23, 1],
        friction=[0.8, 0.01, 0.001],
    )


def robot_spec(calibration=None):
    spec = mujoco.MjSpec.from_file(str(calibration.xml if calibration else XML))
    # BAM writes torques, so remove inherited position-controller ctrl limits.
    for actuator in spec.actuators:
        actuator.ctrllimited = False
        actuator.forcelimited = False
    if calibration is not None:
        for geom in spec.geoms:
            geom.contype = geom.conaffinity = 0
    return spec


def cpu_spec(calibration=None):
    spec = robot_spec(calibration)
    if calibration is None:
        add_floor(spec)
    spec.option.timestep = PHYSICS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return spec


def validate_geometry(calibration):
    """Preserve the original goal and FK convention without shifting robot zeros."""
    actual, baseline = cpu_spec(calibration).compile(), cpu_spec().compile()
    actual_data, baseline_data = mujoco.MjData(actual), mujoco.MjData(baseline)
    aq, _ = indices(actual)
    bq, _ = indices(baseline)
    if actual.nq != 6 or actual.nv != 6:
        raise ValueError("Expected a fixed-base six-joint SO-101 snapshot")
    for q in (HOME, target_bank()[1][0], HOME + np.array([.1, .1, -.1, .1, -.1, .1])):
        actual_data.qpos[aq], baseline_data.qpos[bq] = q, q
        mujoco.mj_forward(actual, actual_data)
        mujoco.mj_forward(baseline, baseline_data)
        if not np.allclose(actual_data.site(SITE).xpos, baseline_data.site(SITE).xpos, atol=1e-6, rtol=0):
            raise ValueError("Calibration geometry changes the baseline FK/target; use the original SO-101 model")
    return actual


def indices(model):
    ids = np.array([model.joint(name).id for name in JOINTS])
    return model.jnt_qposadr[ids], model.jnt_dofadr[ids]


@lru_cache(maxsize=8)
def target_bank(mode="fixed", seed=12345, size=512):
    """Sample in joint space, then use FK; hidden joint solutions are never observed.

    Bank generation is deterministic and independent of PPO/reset RNG. The
    random task samples new targets on every reset, not during an episode.
    """
    if mode not in ("fixed", "random"):
        raise ValueError(mode)
    model = cpu_spec().compile()
    data = mujoco.MjData(model)
    qi, _ = indices(model)
    data.qpos[qi] = HOME
    mujoco.mj_forward(model, data)
    home_tool = data.site(SITE).xpos.copy()
    rng = np.random.default_rng(seed)
    goals, solutions = [], []
    count = 1 if mode == "fixed" else size
    for _ in range(max(1000, 100 * count)):
        q = HOME.copy()
        if mode == "fixed":
            q[:3] += [0.12, -0.10, -0.16]
        else:
            q[:3] += rng.uniform(-0.7, 0.7, 3) * ACTION_SCALE
        # Check the direct joint interpolation from home, including endpoints.
        valid = True
        for fraction in np.linspace(0, 1, 9):
            data.qpos[qi] = HOME + fraction * (q - HOME)
            mujoco.mj_forward(model, data)
            if data.ncon or data.site(SITE).xpos[2] < 0.08:
                valid = False
                break
        if not valid:
            if mode == "fixed":
                raise RuntimeError("Fixed goal path collides with bundled model")
            continue
        goal = data.site(SITE).xpos.copy()
        distance = np.linalg.norm(goal - home_tool)
        if not (0.035 < distance < 0.13) or goal[2] < 0.10:
            if mode == "fixed":
                raise RuntimeError(f"Bad fixed goal: distance={distance}, xyz={goal}")
            continue
        goals.append(goal)
        solutions.append(q)
        if len(goals) == count:
            return np.asarray(goals, np.float32), np.asarray(solutions, np.float32)
    raise RuntimeError("Could not generate enough collision-free targets")

