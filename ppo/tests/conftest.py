from pathlib import Path
import pytest
from so101_ppo.calibration import Calibration


@pytest.fixture(scope='session')
def calibration():
    bundle = Path(__file__).resolve().parents[2] / 'examples' / 'synthetic_calibration'
    return Calibration(bundle, allow_synthetic=True)
