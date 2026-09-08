# Validation performed for this package

These are software results on synthetic data, not measurements of the user's arm.
The generated fixture has mixed joint signs/offsets, per-joint voltage and PWM
caps, and delays `[0, 7, 20, 34, 50, 80]` ms. It uses unfitted 12 V seed parameters
with small deliberate per-joint variations.

| Check | Result |
| --- | --- |
| Fresh installation from `uv.lock`, CPU extra | Passed |
| Tests for mapping, delay ticks, parameter integrity, reset repeatability and original goal | 8 passed |
| Calibration reference | MuJoCo 3.12.0, NumPy 2.2.6, BAM revision 620a64fe |
| PPO physics | MuJoCo 3.7.0, mjlab 1.3.0, MuJoCo Warp 3.7.0.1, Warp 1.12.0 |
| CPU versus calibration-reference worst per-joint RMSE | 1.80e-17 degrees |
| mjlab versus calibration-reference worst per-joint RMSE | 4.38e-6 degrees |
| CPU/mjlab policy-command rollout maximum difference | 1.17e-7 radians |
| Partial reset preserving another world's delay and motor state | Passed |
| Original fixed target and known-FK-command diagnostic | Preserved; 11.36 mm endpoint error on fixture |
| PPO training, checkpoint, ONNX export, CPU evaluation | Completed |
| Checkpoint resume, continued learning and re-export | Completed |
| Calibration source and requirements | Unchanged; hashes supplied |
| NVIDIA GPU execution and graphical viewer | Not tested |
| Physical robot or user's fitted parameters | Not available for testing |

## Learning result and its scope

The 64-environment, 100-iteration synthetic run collected 204,800 transitions.
Its exported policy had **0/20 successes** under the unchanged 15 mm / one-second
hold criterion, with **41.2 mm mean final error**. The run verifies the learning,
export and evaluation paths; it does not establish convergence. The much larger
recommended training budget is a starting configuration, not a guaranteed
success rate. The actual fit was not supplied, so convergence with that model
has not been evaluated.

No example policy weights are packaged. Train a fresh policy with your exported
bundle and use the evaluation success rate to judge it. The included synthetic
bundle only supports installation and software checks.

`replay_parity.json` separates differences between implementations from errors
against the synthetic measurements. `mjlab_check.json` includes policy-step
parity through the actual delay buffers. `learning_check_evaluation.json` and
`resume_check_evaluation.json` contain the numerical policy results. Export
checks compare the normalized Torch actor and the ONNX policy on raw inputs.

The first zero-duration motor update, encoder rounding and independent delay
history are matched across CPU and mjlab. State is committed only after actual
integration, so mjlab's extra control computations during reset do not advance
another world's motor history.
