# Implementation and limits

## Calibration import

`calibration/export_for_ppo.py` uses the supplied interface's session reader, model digest, snapshot packaging and reference rollout. It validates the fit's mapping, LeRobot calibration, model content and P16/I0/D0 provenance against the selected session. All selected logs must have matching effective P/I/D, acceleration/profile and torque-limit settings. Synthetic inputs require an explicit software-test flag.

It exports the original `params.json` intact and also resolves six per-joint BAM M1 files. As in `so101_sysid.Simulation`, these use the registered `sts321512v` motor factory and `q_offset=0`. The robot's zero belongs to the encoder-to-MJCF mapping. Applying the mapping as another joint transform inside MuJoCo would double-count it.

Nominal training voltage is the per-joint median of the supplied logs' initial voltage readings. Reference replay uses each individual log's initial voltage, exactly as calibration does. The hardware torque register percentage becomes a PWM cap using the same approximation in both simulators. Current/overload protection dynamics remain unmodeled.

The full recorded effective-settings snapshot is kept for a future physical runtime. Acceleration and goal-velocity registers are metadata for reproducing that runtime configuration; BAM's fitted `max_velocity` is its internal target-limiter model. The import does not claim to simulate every firmware register separately.

## Dynamics and timing

The training environment uses the session's frozen MJCF, six fitted BAM models, and BAM revision `620a64fe67c1afe94fca81da73b128c7aed17c5f`. The same revision is pinned by the attached calibration requirements. PPO keeps mjlab 1.3.0 with MuJoCo 3.7.0, MuJoCo Warp 3.7.0.1 and Warp 1.12.0. Calibration retains MuJoCo 3.12.0. `ppo/uv.lock` pins the PPO dependency resolution.

Both paths use implicitfast integration at 2 ms. Source solver iteration/tolerance settings are carried into mjlab. The reference fit disabled contacts, so reaching also runs in free space. Pick-place enables robot, floor, and brick contacts with task-specific solver settings; the motor calibration does not identify contact friction or gripping accuracy. There is no extra stiff-friction override or symmetric torque clip after BAM's voltage/back-EMF computation.

For policy commands on the 2 ms grid, a fitted delay `d` becomes `ceil(d / 0.002)` physics ticks. The extra quantization is less than one physics tick. The per-joint buffers are filled with the initial sent target; they are not initialized with zero-angle commands. Resetting one environment replaces only its histories and internal motor target. CPU and mjlab both reproduce the calibration controller's zero-duration first control update after reset.

Recorded-motion replay evaluates the exact timestamped command schedule at `t - d` in seconds and performs zero-order hold. This preserves irregular serial timing without adding a second delay. Prediction is a free rollout initialized once, never reset to every encoder sample.

The continuous policy command is limited to 8 degrees/second, then rounded through the calibrated raw encoder conversion. The previous-command observation refers to this continuous slew-limited state. The actual BAM target is the rounded sent command. Held wrist/gripper commands undergo the same rounding. This models a finite encoder command resolution without shifting physical joint observations.

## Same reach task

The baseline home is `[0, -0.5, 0.5, 0.3, 0, 0]` radians. The original fixed goal is generated with FK at `home + [0.12, -0.10, -0.16, 0, 0, 0]`. Import checks FK compatibility at several poses, then preserves that goal. The three action scales remain `[0.35, 0.4, 0.4]` radians. The command range intersects these with the calibrated limits plus the same 3-degree margin used by the interface.

Observation size remains 21: six home-relative joint angles, six joint velocities scaled by 0.1, three target coordinates, three target-minus-tool coordinates, and three previous normalized command coordinates. The actor and critic are 128-by-128 ELU networks with observation normalization. PPO still uses RSL-RL. ONNX includes the actor's learned observation normalization.

Reward terms and coefficients are retained: smooth coarse/fine distance reward, joint-velocity penalty and normalized command-change penalty. Each control step is 20 ms. Success remains a 15 mm reach held for one second with all joint speeds under 0.15 rad/s in a four-second episode. Calibration changes the plant and command rate limit, so old policies are not interchangeable even though the task is the same.

## Fixed pick-and-place task

`--task pick-place` selects a fixed brick-and-square scene, a fixed open-gripper home pose, and a 60-second episode limit. It requires fixed mode, home starts, nominal physics, and fixed previews. All six calibrated joints receive absolute position targets through the existing 50 Hz command, 8-degree/second slew, and encoder-quantization pipeline. CPU evaluation uses BAM's `MujocoController`; GPU training uses the project's batched `BamActuator` port with the same fitted parameters.

The policy contract uses schema 5, 46 observations, and six actions. Observations include joint state, previous commands, brick pose and velocity, tool-relative brick position, target position, contact flags, and milestone/hold state. This contract is separate from reaching; incompatible task or reward-version checkpoints are rejected on resume. ONNX inference includes observation normalization.

CPU and GPU share `TaskState` for reward version 2 and milestone logic. First finger contact pays once, so scraping can count as touch. Pickup requires bilateral contact and 20 mm floor clearance held for 0.2 seconds. Placement additionally requires release, the full brick footprint inside the square, and 0.5 seconds of settling. Reward combines once-per-episode bonuses of 1/10/100 with per-step proximity, grasp, lift, carry, and motion terms. Success or failure ends an episode; otherwise it times out. See the [task guide](../ppo/README.md#fixed-pick-and-place) for dimensions, settling thresholds, and commands.

Fresh `Rollout/` diagnostics describe each collected batch; completed-episode training statistics can remain unchanged between resets. Previews run a copied deterministic actor on CPU, while PPO collects stochastic actions on GPU. Contact rate and total return do not establish successful grasping. The scripted physical baseline is a feasibility check only and does not supervise PPO or establish learned-policy success.

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

## Periodic policy visualization

`so101-train` optionally wraps the pinned runner logger with a delegating post-update callback, preserving the upstream learning loop and checkpoint frequency. It counts completed iterations locally to each training invocation, captures the starting and final policy, and deduplicates the final capture when it coincides with an interval. Preview rollout count is distinct from the pinned runner's reused checkpoint iteration label.

Preview inference uses a deep-copied actor and frozen observation normalizer on CPU. Python, NumPy and initialized Torch RNG streams are restored around setup and capture; the live training actor is never moved or put into evaluation mode. A separate `CpuArm` runs the shared evaluation episode helper, preserving command timing, success rules and failure termination. Seed 2026 and target-bank seed 54321 preserve the original first target and initial pose. The camera remains fixed, and the initial pose remains fixed with the default home starts; robust runs preview nominal dynamics. `--visualize-targets auto` rotates targets between random-mode previews, while fixed-mode previews retain their target. Explicit `fixed` selection restores the previous single-target comparison behavior.

Saved frames use MuJoCo's offscreen renderer, a target sphere and Pillow text overlays. ImageIO/FFmpeg encodes H.264 MP4 at 1280×720 and 25 FPS. Individual clips play in real time; compilation streams and resamples their frames for the requested speed. Files are finalized from temporary paths, and per-episode metadata is written after completed captures. No video or renderer is created when visualization is off. Policies and calibration bundle schemas are unchanged.

Preview target rotation uses a persistent local NumPy generator and the existing held-out target bank. Subsequent captures sample entries at least 3 cm from the previous target, falling back to the farthest entry if none meet that separation. Targets do not change within an episode, and duplicate final-capture calls do not advance selection. Each resumed invocation starts its own reproducible sequence. Training target sampling, rewards, dynamics, RNG streams and checkpoint formats are unchanged.

Visualization metadata records requested/effective target selection, `initial_target_m`, `initial_target_index`, and per-episode `target_index`/`target_m` plus the existing target coordinate metrics. The old shared `target_m` at the manifest root is retained only for fixed previews. The target sphere, policy goal observation and overlay all use the same selected target.

### Optional random starts

`--start-mode home|random` defaults to the legacy ±0.02-radian home jitter for new runs. Training resume and policy evaluation inherit `start_mode` from the policy manifest, with missing legacy fields treated as `home`; explicit overrides take precedence. Start mode is task metadata and does not change the policy/control contract or bundle schema.

For the near workspace, start sampler version 1 filters 512 joint solutions from the original reachable-region generator, using seed 24680 for training and 86420 for evaluation/previews. Reference geometry retains collision checks, including nine samples on the home-to-pose interpolation and tool clearance above the floor. Candidates must also satisfy model, calibration and policy command bounds; invalid poses are rejected without clipping. An empty bank is an error. This is a checked reset distribution, not certification of physical motion or four-second success for arbitrary start–target pairs.

The accepted training bank is cached on the environment device and sampled per resetting environment. CPU and mjlab reset velocities to zero and seed commands and BAM histories from the sampled pose. Passive joints retain HOME. The existing near-home branch preserves its RNG draw order. Evaluation uses a separate local start RNG with its episode seed; preview starts use a local seed of 2026, preserving the existing preview target sequence and all training RNG streams. Rotating previews exclude the previous start index where possible; fixed previews repeat the initial sampled start. Resume restarts the local preview sequence.

Policy and preview manifests record start mode, sampler version and both bank seeds. Evaluation CSV rows and preview episode records include `initial_q_rad` and `start_index`; overlays show random-start bank indices. Evaluation summaries record their effective start mode and accepted bank size.

### Wide random reaching

New random-task CLI runs default to `--workspace wide`; new fixed tasks default to `near`. Resume inherits task mode and workspace. Missing workspace metadata in existing manifests means `near`, preserving their original action mapping, banks, reward and four-second episodes. Random starts remain explicitly enabled through `--start-mode random`.

`workspace.py` generates 4,096 accepted joint configurations per wide bank, uniformly proposing the three controlled joints across the intersection of calibrated limits and model limits with a three-degree margin. Wrist and gripper remain at HOME. The home-distance shell is removed. Reference-model self/floor collisions and 8 cm tool clearance are checked at endpoints and along home-to-pose interpolation at no more than two-degree joint intervals. A 1 cm spatial voxel holds at most two targets to reduce clustering; the distribution is broad but not exactly uniform in Cartesian volume. Training/evaluation goal seeds remain 12345/54321; start seeds remain 24680/86420. Banks are generated offline at setup, cached, then sampled on device. These checks establish sampled geometric feasibility, not collision avoidance by an arbitrary policy or physical deployment readiness.

Wide actions map [-1,1] to the midpoint and half-range of the usable joint bounds. CPU and mjlab use the same mapping for commands and previous-command observations. Near actions retain the original HOME-centered mapping. The 8°/s slew limit, firmware settings, fitted delays, success thresholds and ONNX tensor dimensions are unchanged. Wide manifests use policy schema 3 with action center, scale, workspace version and episode/reward configuration; near schema 2 remains supported. Loading verifies the selected contract against the frozen bundle, and cross-workspace resume/evaluation is rejected before learning/inference.

Wide episode duration is `ceil(2 * max_joint_distance_from_HOME / MAX_COMMAND_SPEED + 3)` seconds, allowing worst-case command travel via HOME plus two seconds to settle and one second to hold (35 seconds for the selected real bundle). This budget is not a success guarantee. The coarse distance reward uses 35 cm rather than 10 cm to retain a useful proximity signal farther from the goal; the fine reward and PPO optimizer settings remain unchanged. Saved evaluation and previews use the same duration/reward configuration. Wide previews zoom out, reserve a separate header above the rendered scene in recordings, and separate both targets and starting tool positions by at least 15 cm where possible, falling back to the farthest position. Fixed comparisons retain their sampled pose and target.
