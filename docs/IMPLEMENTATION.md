# Implementation and limits

## Calibration import

`calibration/export_for_ppo.py` uses the supplied interface's session reader, model digest, snapshot packaging and reference rollout. It validates the fit's mapping, LeRobot calibration, model content and P16/I0/D0 provenance against the selected session. All selected logs must have matching effective P/I/D, acceleration/profile and torque-limit settings. Synthetic inputs require an explicit software-test flag.

It exports the original `params.json` intact and also resolves six per-joint BAM M1 files. As in `so101_sysid.Simulation`, these use the registered `sts321512v` motor factory and `q_offset=0`. The robot's zero belongs to the encoder-to-MJCF mapping. Applying the mapping as another joint transform inside MuJoCo would double-count it.

Nominal training voltage is the per-joint median of the supplied logs' initial voltage readings. Reference replay uses each individual log's initial voltage, exactly as calibration does. The hardware torque register percentage becomes a PWM cap using the same approximation in both simulators. Current/overload protection dynamics remain unmodeled.

The full recorded effective-settings snapshot is kept for a future physical runtime. Acceleration and goal-velocity registers are metadata for reproducing that runtime configuration; BAM's fitted `max_velocity` is its internal target-limiter model. The import does not claim to simulate every firmware register separately.

## Dynamics and timing

The training environment uses the session's frozen MJCF, six fitted BAM models, and BAM revision `620a64fe67c1afe94fca81da73b128c7aed17c5f`. The same revision is pinned by the attached calibration requirements. PPO keeps mjlab 1.3.0 with MuJoCo 3.7.0, MuJoCo Warp 3.7.0.1 and Warp 1.12.0. Calibration retains MuJoCo 3.12.0. `ppo/uv.lock` pins the PPO dependency resolution.

Both paths use implicitfast integration at 2 ms. Source solver iteration/tolerance settings are carried into mjlab. The reference fit disabled contacts, so the imported task also runs in free space. This is appropriate for this reaching objective and does not identify collisions, table interaction or gripping. There is no extra stiff-friction override or symmetric torque clip after BAM's voltage/back-EMF computation.

For policy commands on the 2 ms grid, a fitted delay `d` becomes `ceil(d / 0.002)` physics ticks. The extra quantization is less than one physics tick. The per-joint buffers are filled with the initial sent target; they are not initialized with zero-angle commands. Resetting one environment replaces only its histories and internal motor target. CPU and mjlab both reproduce the calibration controller's zero-duration first control update after reset.

Recorded-motion replay evaluates the exact timestamped command schedule at `t - d` in seconds and performs zero-order hold. This preserves irregular serial timing without adding a second delay. Prediction is a free rollout initialized once, never reset to every encoder sample.

The continuous policy command is limited to 8 degrees/second, then rounded through the calibrated raw encoder conversion. The previous-command observation refers to this continuous slew-limited state. The actual BAM target is the rounded sent command. Held wrist/gripper commands undergo the same rounding. This models a finite encoder command resolution without shifting physical joint observations.

## Same reach task

The baseline home is `[0, -0.5, 0.5, 0.3, 0, 0]` radians. The original fixed goal is generated with FK at `home + [0.12, -0.10, -0.16, 0, 0, 0]`. Import checks FK compatibility at several poses, then preserves that goal. The three action scales remain `[0.35, 0.4, 0.4]` radians. The command range intersects these with the calibrated limits plus the same 3-degree margin used by the interface.

Observation size remains 21: six home-relative joint angles, six joint velocities scaled by 0.1, three target coordinates, three target-minus-tool coordinates, and three previous normalized command coordinates. The actor and critic are 128-by-128 ELU networks with observation normalization. PPO still uses RSL-RL. ONNX includes the actor's learned observation normalization.

Reward terms and coefficients are retained: smooth coarse/fine distance reward, joint-velocity penalty and normalized command-change penalty. Each control step is 20 ms. Success remains a 15 mm reach held for one second with all joint speeds under 0.15 rad/s in a four-second episode. Calibration changes the plant and command rate limit, so old policies are not interchangeable even though the task is the same.

## Replay report

`cpu_vs_reference_deg` and `mjlab_vs_reference_deg` measure implementation differences relative to the exported calibration simulator. `cpu_vs_measured_deg` and `mjlab_vs_measured_deg` measure prediction errors against the recording. The former answers whether the simulator port agrees; the latter answers how well the model predicted that motion.

A passed replay on synthetic data validates software behavior. A passed replay on real data still does not measure absolute tool geometry, camera accuracy or physical task success. The default software-parity limits are engineering thresholds and are not a statistical confidence interval.

The checksum manifest detects changed bundle files. It is a consistency check, not a cryptographic signature or proof of physical calibration quality. The physical quality remains that of the supplied recordings, mapping and fit.

## Physical PPO runtime remains future work

This package contains the original hardware calibration interface. The PPO entry points only train, export, replay and evaluate in simulation. No PPO command opens a serial port.

The exported policy carries the joint order, mapping, effective register settings, command bounds, timing and motor model identity needed for a future physical runtime. That runtime must use calibrated feedback in radians, the same quantization/slew/limit logic, P16/I0/D0 and the recorded profile settings. Calibration restores some motor settings on torque-off, so they cannot be assumed to persist afterward.

The actor currently observes simulator joint velocity. Raw velocity registers or a position-difference estimate cannot be substituted without validating units, signs, timestamping, noise and filtering. These observations, physical target placement and real task trials are the next integration work. They are not estimated by importing the fitted JSON, and `hardware_ready` remains false.

## Sources

- [BAM source at the calibration revision](https://github.com/Rhoban/bam/tree/620a64fe67c1afe94fca81da73b128c7aed17c5f)
- [BAM fitting documentation](https://bam.readthedocs.io/en/latest/identification/fitting.html)
- [mjlab](https://github.com/mujocolab/mjlab)
- [Original SO-101 model revision](https://github.com/TheRobotStudio/SO-ARM100/tree/eecbe3e0a9ebb23e25ad7b2759b03884c6660903/Simulation/SO101)

The supplied calibration source remains the authority for the application-specific JSON schema and controller choices.

## Offline evaluation provenance

New fits add `training_recording_sha256`, a mapping from original recording paths to SHA-256 hashes of their file bytes, alongside the existing `training_logs`. Independent evaluation checks both training and validation inputs against those hashes, exact paths, and portable `session/runs/run/filename` identities. This rejects renamed byte-for-byte copies for new fits and relocated sessions for legacy fits. Legacy fitted files are never rewritten; without stored hashes, a legacy recording renamed at every identifying level cannot be recognized. Editing or reserializing a log changes its byte hash, so this is an archive-consistency guard, not an adversarial duplicate detector.

The optimizer, hardware operations, public commands, PPO bundle schema and policy contract are unchanged. Export retains `fitted_params.json` byte-for-byte, includes existing report/progress and per-trial JSON diagnostics under `source_fit/`, and checksums them with the other bundle artifacts. PNG/CSV plot duplicates, raw telemetry and training checkpoints remain in `sessions/`.

## Selecting the Torch backend

The supplied Linux lock selected Torch 2.10.0+cpu even without `--extra cpu`. The project now declares mutually exclusive `cpu` and `cuda` extras, with CUDA 12.8 wheels for Torch 2.10.0 and torchvision 0.25.0. The existing lock was extended without upgrading its prior package versions. Use `uv sync --project ppo --locked --extra cuda` and retain `--extra cuda` on root-level `uv run --project ppo --locked` commands, or substitute `--extra cpu` throughout for CPU. This follows [uv's explicit PyTorch backend configuration](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies).

## Recorded-pose replay initialization

Real recordings exposed a replay-only startup bug: calling `env.scene.update()` after a forward pass committed pending actuator state from the task's reset pose, overwriting the recorded initial targets before any integration. Replay now seeds the recorded state/history and forwards without advancing actuator state; updates follow actual physics steps. Training dynamics, actuator code, reference exports and parity thresholds are unchanged. CPU and CUDA regressions replay a short held pose with all held servos away from task home.
