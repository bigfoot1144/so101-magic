# Calibrated SO-101 PPO

Start with [the combined package instructions](../README.md). They cover exporting your calibration, installation, replay checks, training, resuming, evaluation and playback.

From the repository root, the normal training command is:

```bash
uv run --project ppo --locked --extra cuda so101-train --calibration data/calibrations/so101-12v-20260907-174850-482396032 --mode fixed --num-envs 1024 --iterations 300 --run runs/fixed_calibrated
```

The calibration bundle is required. There is no fallback to the earlier 7.4 V motor fit. The task remains the original nearby fixed-point reach with a 15 mm, one-second hold criterion. Controller settings now match the supplied calibration code: P16, 8 degrees/second command slew, 50 Hz control and 2 ms physics.
