# PPO quickstart

Train the SO-101 for position reaching or fixed pick-and-place, watch periodic previews, and evaluate the saved policy. **All commands run from the repository root**, the directory containing `ppo/` and `data/`.

## At a glance

| You want to… | Use |
| --- | --- |
| Reach one particular position | `--mode fixed` |
| Pick up a brick and place it inside a square | `--task pick-place` (fixed layout and home start) |
| Reach widely varied positions across episodes | `--mode random` (new runs use the wide workspace) |
| Keep the original small reaching region | `--workspace near` |
| Start each episode from a varied reachable pose | Add `--start-mode random` |
| Vary simulated motor properties during either reaching mode | Add `--robust` |
| Display periodic policy previews | `--visualize show` |
| Record previews and a timelapse | `--visualize save` |
| Display and record previews | `--visualize both` |
| Choose the preview interval | `--visualize-every 25` |
| Vary random-mode preview targets | `--visualize-targets auto` (default) |
| Repeat one preview target for comparison | `--visualize-targets fixed` |

[Install](#1-install-the-ppo-environment) · [Tasks](#2-understand-the-tasks) · [Train](#3-train-a-policy) · [Visualize](#4-watch-and-record-progress) · [Evaluate](#5-evaluate-and-watch-the-saved-policy) · [Resume](#6-resume-training) · [All options](#training-option-reference)

## 1. Install the PPO environment

For an NVIDIA CUDA machine:

```bash
uv python install 3.12
uv sync --project ppo --locked --extra cuda

uv run --project ppo --locked --extra cuda python -c \
  "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

This creates or updates `ppo/.venv`; activation is unnecessary. Keep `--extra cuda` on subsequent commands so synchronization retains the CUDA backend. Keep the calibration environment separate.

For CPU-only use, install with `uv sync --project ppo --locked --extra cpu`, replace `--extra cuda` with `--extra cpu` throughout, and use `--device cpu` for training. Use four environments for a CPU smoke test.

### Have the calibration bundle available

The commands use:

```text
data/calibrations/so101-12v-20260907-174850-482396032/
```

**This directory is ignored by Git.** A fresh clone needs the entire exported bundle copied into that location, including `bundle.json`, model/meshes, motor files, configuration and replays. See [calibration import instructions](../README.md#import-the-selected-finished-fit) if you need to export it.

An existing bundle is sufficient for PPO. No calibration-environment installation, raw telemetry, new recording or refit is required to train with it.

## 2. Understand the tasks

The default task is **position reaching**, with two modes listed below. `--task pick-place` selects a separate six-joint task with physical contacts; see [Pick and place](#pick-and-place). Random starts, wide workspaces, and robust training apply only to reaching.

| Mode | Objective | Target selection |
| --- | --- | --- |
| `fixed` | Learn to reach one particular position | Every episode uses approximately **(0.3146, −0.0333, 0.2508) metres** in the robot's base frame |
| `random` | Learn to reach different positions | Each environment selects from a deterministic bank of **4,096 reachable positions** in the wide workspace on reset; the target remains fixed during that episode |

New random-mode runs use **`--workspace wide`** by default. Targets cover the usable range of the three controlled joints, constrained by model limits, calibrated encoder limits, joint-limit margins, collisions and floor clearance. The selected real calibration produces a target bank spanning roughly **80 × 87 × 44 cm** in base-frame X/Y/Z; this is a bounding box around a shaped reachable region, not a box in which every point is reachable. Joint-space samples are spatially thinned to avoid crowding nearly identical tool positions.

Use **`--workspace near`** for the original 512-target task, roughly **3.5–13 cm from home**. Fixed-mode runs default to that original workspace. The target coordinates are part of the policy's observation. The target bank is generated separately from PPO/reset randomness; changing `--seed` does not change the training bank itself.

### Reaching behavior and success criteria

- PPO controls **shoulder pan, shoulder lift and elbow flex**. Wrist flex, wrist roll and gripper hold their home commands.
- Episodes start near home by default; `--start-mode random` opts into wider reachable starts. Policy commands run at **50 Hz**. Near-workspace episodes last **four seconds**; wide episodes allow travel through the larger region at the existing 8°/s command limit—**35 seconds for the selected calibration**.
- Success requires the tool to stay within **15 mm** of the target while every joint moves slower than **0.15 rad/s**, for **one continuous second**.
- Rewards encourage proximity and penalize excessive joint speeds and abrupt command changes.
- Reaching uses calibrated free-space dynamics. The separate pick-place task enables contacts and grasping. Throwing, general orientation-control tasks, moving-target tracking, vision and physical PPO execution are not implemented.

## 3. Train a policy

Each invocation needs a **new output directory**. Rename `--run` if the example directory already exists. The examples use 500 iterations, the CLI default; this is a starting budget, not a convergence guarantee. Wide tasks take substantially longer to learn; 500 iterations provide only 320 simulated seconds per environment, about nine full 35-second episodes.

### Fixed-target reach

```bash
uv run --project ppo --locked --extra cuda so101-train \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --mode fixed --device cuda:0 \
  --num-envs 1024 --iterations 500 \
  --run runs/fixed_reach \
  --visualize both --visualize-every 25
```

### Random-target reach

```bash
uv run --project ppo --locked --extra cuda so101-train \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --mode random --workspace wide --start-mode random --device cuda:0 \
  --num-envs 1024 --iterations 500 \
  --run runs/random_reach \
  --visualize both --visualize-every 25
```

**The random-mode command above now moves the target dot between preview episodes automatically.** The dot stays still during each episode; the hand starts at a different checked pose between rotating previews. No extra flag is needed: `--visualize-targets auto` is the default. Training already samples different targets across environments and resets.

These commands display and record previews, so they require a working graphical desktop. For headless recording, use the command in [the visualization section](#4-watch-and-record-progress). To train without previews, omit the visualization options.

**Use a fresh run for the wide workspace.** Its larger action range has a different policy contract; old near-workspace checkpoints remain usable for near training/evaluation but cannot be resumed into the wide workspace.

### Optional random starting poses

Add `--start-mode random` to either reaching mode to train from varied reachable poses:

```bash
uv run --project ppo --locked --extra cuda so101-train \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --mode random --workspace wide --start-mode random --device cuda:0 \
  --num-envs 1024 --iterations 500 \
  --run runs/random_reach_random_starts \
  --visualize both --visualize-every 25
```

This setting applies to **training, evaluation, and previews**. New runs default to `--start-mode home`, which retains the small ±0.02-radian jitter. In random-start mode, each resetting environment independently samples a joint pose from the selected workspace. Shoulder pan, shoulder lift and elbow vary; wrist and gripper stay at home, and initial velocities are zero.

The wide workspace generates 4,096 accepted starting poses across the usable joint range, with collision and floor-clearance checks along sampled paths from home. The near workspace retains its original 512-candidate sampler. Both enforce model, calibration and command limits. Evaluation and previews use a separate start bank from training. These are checked simulation starts, not a guarantee of a collision-free physical trajectory or success for every start–target pair within its episode budget. The start-mode flag selects starting conditions within the chosen workspace; workspace selection determines action range and episode duration.

Rotating random-mode previews vary both start and target. `--visualize-targets fixed` repeats one sampled start and target for direct comparison; fixed-target mode also keeps a repeatable preview start. The camera stays fixed. Clips identify the start bank index, and preview metadata records the actual initial joint angles.

### Optional robustness training

Add `--robust` to either mode to vary motor properties around the imported calibration:

| Property | Variation |
| --- | --- |
| Voltage | ±5% |
| Effective proportional gain | ±10% |
| Friction | ±10% |
| Fitted command delays | Unchanged |

For example, train random reaching with those variations:

```bash
uv run --project ppo --locked --extra cuda so101-train \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --mode random --robust --device cuda:0 \
  --num-envs 1024 --iterations 500 \
  --run runs/random_reach_robust \
  --visualize both --visualize-every 25
```

For robust fixed reaching, change `--mode random` to `--mode fixed` and use `--run runs/fixed_reach_robust`.

Robustness is a training variation, not another task. These ranges are heuristic, not measured calibration uncertainties. Previews and automatic final evaluation still use **nominal calibration**.

### What training produces

Every iteration collects **32 control steps from every environment**, then updates the policy. With 1,024 environments and 500 iterations, that is **16,384,000 transitions**. An iteration is not a complete episode.

The run directory contains checkpoints, `policy.onnx`, `policy_manifest.json`, training configuration, a frozen `calibration/`, ONNX export checks and a 20-episode CPU evaluation by default. Keep the directory together. Judge learning by success rate and final distance, not by iteration count alone.

## 4. Watch and record progress

| Option | What happens |
| --- | --- |
| `--visualize off` | No previews; default behavior |
| `--visualize show` | Display periodic episodes in a MuJoCo window |
| `--visualize save` | Save individual MP4 episodes and a compiled timelapse |
| `--visualize both` | Display and save |
| `--visualize-every N` | Capture every N completed PPO rollout/update iterations; default **25** |
| `--visualize-targets auto` | Default: rotate targets for random-mode previews; retain the original target for fixed mode |
| `--visualize-targets fixed` | Repeat the same preview target and starting pose for direct progress comparisons |
| `--visualize-targets rotate` | Explicitly rotate random-mode preview targets; fixed-mode training still retains its original target |
| `--timelapse-speed S` | Compilation playback multiplier; default **4×**; individual clips remain real time |

Previews briefly pause learning and run the current deterministic policy in a **separate CPU simulation**. The camera is reused. The starting pose is reused unless random starts and rotating previews are both enabled. With `--mode random`, the target now changes between captures by default, but remains stationary during each episode. Fixed-mode previews retain the original target. Add `--visualize-targets fixed` to reuse one target for direct progress comparisons. Training state and RNG streams are preserved.

Random-mode **training already varies its targets** independently across environments and resets. Preview target rotation uses a separate, reproducible generator and the existing held-out preview bank. Target rotation does not change training targets or require retraining. Enabling `--start-mode random` does change the training start distribution. In the wide workspace, consecutive preview targets and starting tool positions each move at least **15 cm** when possible; near targets retain the **3 cm** threshold. A farthest-position fallback handles a narrower bank.

Captures include the starting policy, each interval and the final policy. Displayed episodes run in real time; closing the window stops further display while training and requested recording continue.

### Compare progress on one target

To keep the dot in one place while training on random targets, use:

```bash
uv run --project ppo --locked --extra cuda so101-train \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --mode random --device cuda:0 \
  --num-envs 1024 --iterations 500 \
  --run runs/random_reach_comparison \
  --visualize both --visualize-every 25 --visualize-targets fixed
```

This option controls previews only. Use `auto` or `rotate` to see different goals; use `fixed` to compare policy snapshots under the same conditions.

### Headless recording

On an NVIDIA machine without a desktop:

```bash
env MUJOCO_GL=egl uv run --project ppo --locked --extra cuda so101-train \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --mode random --device cuda:0 \
  --num-envs 1024 --iterations 500 \
  --run runs/random_reach_recorded \
  --visualize save --visualize-every 25 --timelapse-speed 4
```

Outputs appear under `runs/<run>/visualization/`:

```text
visualization/
  episode_000000.mp4       # Starting policy
  episode_000025.mp4       # After 25 completed iterations
  ...
  manifest.json           # Preview conditions, metrics and any errors
  timelapse.mp4            # Compilation generated when training exits
```

Clips are **1280×720 at 25 FPS**, with labels for completed rollouts, episode time, distance, outcome, and target bank index/XYZ coordinates. At the default 4× playback, a full near episode takes one second and a full 35-second wide episode takes 8.75 seconds. Wide previews use a zoomed-out camera; recordings reserve a separate text header so targets stay visible. On interruption, completed clips are retained and compilation is attempted; the incomplete current clip is discarded.

### Live learning charts

In another terminal:

```bash
uv run --project ppo --locked --extra cuda tensorboard --logdir runs
```

Open **http://localhost:6006**. TensorBoard shows training metrics; the periodic previews show behavior at the displayed targets. Use `--visualize-targets fixed` to compare a single repeatable condition.

## 5. Evaluate and watch the saved policy

Evaluate random reaching over 100 episodes:

```bash
uv run --project ppo --locked --extra cuda so101-eval \
  --policy runs/random_reach/policy.onnx \
  --episodes 100 \
  --output reports/random-reach-evaluation.json
```

The policy manifest selects its training mode, workspace, action mapping, episode duration, start mode and frozen calibration automatically. Random-mode evaluation uses a **different target-bank seed** from training. For the fixed policy, substitute `runs/fixed_reach/policy.onnx` and a distinct report path.

To evaluate a policy under a different starting condition, explicitly override the saved setting:

```bash
uv run --project ppo --locked --extra cuda so101-eval \
  --policy runs/random_reach_random_starts/policy.onnx \
  --start-mode home --episodes 100 \
  --output reports/random-start-policy-home-evaluation.json
```

Use `--start-mode random` instead to test varied starts. Reports label the effective start mode; the accompanying CSV records each initial joint pose and random-start bank index. Compare scores under the same workspace and starting condition. Evaluation cannot switch a saved policy between near and wide action mappings. Older policies without a saved start mode use `home`.

Watch ten evaluation episodes on a graphical desktop:

```bash
uv run --project ppo --locked --extra cuda so101-eval \
  --policy runs/random_reach/policy.onnx \
  --viewer --episodes 10 \
  --output reports/random-reach-viewer-evaluation.json
```

Numerical evaluation runs in CPU MuJoCo even when the environment has CUDA dependencies installed. Compare success rate and final distance; the periodic preview alone does not measure performance across the random target bank.

## 6. Resume training

Continue the same random-reaching run for 200 additional iterations:

```bash
uv run --project ppo --locked --extra cuda so101-train \
  --calibration runs/random_reach/calibration \
  --resume runs/random_reach/checkpoint.pt \
  --mode random --device cuda:0 \
  --num-envs 1024 --iterations 200 \
  --run runs/random_reach_more \
  --visualize both --visualize-every 25
```

Use the **original frozen bundle**, preserve your intended task/robustness settings, and choose a new run directory. Add `--robust` again when continuing a robust run. Changed calibration requires a new training run.

Resume inherits the saved task mode, workspace and start mode when their options are omitted. Old checkpoints without workspace metadata inherit `near`. Switching `--workspace` changes the policy contract and requires a fresh run. To introduce random starts to an existing near-home policy, add the option explicitly and continue training in a new run:

```bash
uv run --project ppo --locked --extra cuda so101-train \
  --calibration runs/random_reach/calibration \
  --resume runs/random_reach/checkpoint.pt \
  --mode random --start-mode random --device cuda:0 \
  --num-envs 1024 --iterations 200 \
  --run runs/random_reach_random_starts_more \
  --visualize both --visualize-every 25
```

Use `--start-mode home` to explicitly return to near-home starts. The new run records the selected setting; existing checkpoints and policies are unchanged.

You can change `--visualize-targets` when resuming without retraining from scratch. An already running process will not pick up updated visualization code or options; resume a saved checkpoint with the desired settings.

Preview counts restart at zero for the new invocation, beginning with the loaded policy. The preview target and random-start sequences also restart. The visualization manifest records the source checkpoint, target-selection mode, initial conditions, per-episode targets and cumulative training control steps.

## Training option reference

| Option | Default | Meaning |
| --- | --- | --- |
| `--calibration PATH` | **Required** | Exported calibration bundle |
| `--task reach/pick-place` | `reach` for new runs; inherited on resume | Select reaching or fixed six-joint pick-and-place |
| `--mode fixed/random` | `fixed` for new runs; inherited on resume | Reaching mode |
| `--workspace near/wide` | `wide` for new random runs, `near` for fixed; inherited on resume | Reachable region, action mapping and episode budget; changing workspace requires a fresh policy |
| `--start-mode home/random` | `home` for new runs; inherited on resume | Reaching: near-home jitter or checked random poses. Pick-place: fixed home pose only. Also used by evaluation and previews |
| `--num-envs N` | `1024` | Parallel simulated arms |
| `--iterations N` | `500` | PPO iterations to run; additional iterations on resume |
| `--device DEVICE` | `cuda:0` | Training device; also supports `cpu` |
| `--seed N` | `42` | Training and reset randomness |
| `--run PATH` | Timestamped directory under `runs/` | New output directory |
| `--resume PATH` | None | Training checkpoint to load; requires its original bundle |
| `--robust` | Off | Enable the motor-property variations described above for reaching only |
| `--eval-episodes N` | `20` | CPU evaluation episodes after training; `0` skips evaluation |
| `--visualize off/show/save/both` | `off` | Periodic policy previews |
| `--visualize-every N` | `25` | Preview every N completed iterations; must be positive |
| `--visualize-targets auto/fixed/rotate` | `auto` | Rotate random-mode previews or explicitly repeat one target; fixed-mode training always retains its original target |
| `--timelapse-speed S` | `4` | Compilation speed; must be finite and positive |
| `--allow-synthetic` | Off | Allow synthetic calibration fixtures for software testing |
| `--help` | — | Show CLI usage |

Success thresholds, network architecture and PPO learning rate are configured in code. Workspace selection also selects the episode budget and coarse distance-reward scale (10 cm near, 35 cm wide); these are not separate CLI options. See [task configuration](../ppo/src/so101_ppo/task.py), [policy/control contract](../ppo/src/so101_ppo/contract.py) and [implementation details](IMPLEMENTATION.md).

```bash
uv run --project ppo --locked --extra cuda so101-train --help
uv run --project ppo --locked --extra cuda so101-eval --help
```

All commands operate in simulation. Policy manifests retain `hardware_ready: false`. Back up complete bundles and run directories separately: real calibration bundles, recordings, runs and generated reports are ignored by Git.

## Pick and place

Use `--task pick-place` for the fixed brick-and-square task. It controls all six joints and rewards first touch, a sustained pickup, and release/settling inside the square. It needs a fresh checkpoint. See the [pick-and-place guide](../ppo/README.md#fixed-pick-and-place) for training, evaluation, scripted physical validation, and previews. Existing commands continue to select reaching by default.
