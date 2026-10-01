"""Reachable episode starts shared by training, evaluation and previews."""
import numpy as np

from .contract import HOME, JOINTS
from .model import cpu_spec, target_bank

TRAIN_START_SEED = 24680
EVAL_START_SEED = 86420
START_SAMPLER_VERSION = 1


def resolve_start_mode(requested=None, manifest=None):
    mode = requested if requested is not None else (manifest or {}).get("start_mode", "home")
    if mode not in ("home", "random"):
        raise ValueError("Expected home or random start mode")
    return mode


def start_metadata(mode):
    return {"start_mode": resolve_start_mode(mode),
            "start_sampler_version": START_SAMPLER_VERSION,
            "training_start_bank_seed": TRAIN_START_SEED,
            "evaluation_start_bank_seed": EVAL_START_SEED}


def start_bank(calibration, seed=TRAIN_START_SEED, workspace="near"):
    """Filter collision-checked joint solutions; never clip invalid candidates.

    The reference model has contacts enabled (calibrated runtime models do not).
    Near mode checks nine points on each home-to-pose path and floor clearance.
    Wide mode uses its larger bank and denser path checks.
    This is a simulation reset region, not a physical motion safety guarantee.
    """
    if workspace == "wide":
        from .workspace import task_bank
        return task_bank(calibration, "random", seed, workspace)[1]
    if workspace != "near":
        raise ValueError("Expected near or wide workspace")
    _, candidates = target_bank("random", seed=seed, size=512)
    model = cpu_spec().compile()
    ids = np.array([model.joint(name).id for name in JOINTS])
    limited = model.jnt_limited[ids].astype(bool)
    lower = np.maximum(calibration.lower, np.where(limited, model.jnt_range[ids, 0], -np.inf))
    upper = np.minimum(calibration.upper, np.where(limited, model.jnt_range[ids, 1], np.inf))
    task_lower, task_upper = calibration.task_bounds()
    lower[:3] = np.maximum(lower[:3], task_lower)
    upper[:3] = np.minimum(upper[:3], task_upper)
    valid = np.isfinite(candidates).all(axis=1) & ((candidates >= lower) & (candidates <= upper)).all(axis=1)
    bank = candidates[valid].copy()
    if not len(bank):
        raise ValueError("No reachable starting poses remain within model, calibration and command limits")
    bank.setflags(write=False)
    return bank


def home_start(rng, bounds):
    """Preserve the legacy near-home sampling and random draw order."""
    q = HOME.copy()
    q[:3] += rng.uniform(-0.02, 0.02, 3)
    q[:3] = np.clip(q[:3], *bounds)
    return q


def sample_start(bank, rng, previous=None):
    """Return a bank index and pose, optionally excluding the previous index."""
    if previous is not None and len(bank) > 1:
        index = int(rng.integers(len(bank) - 1))
        index += index >= previous
    else:
        index = int(rng.integers(len(bank)))
    return index, bank[index].copy()
