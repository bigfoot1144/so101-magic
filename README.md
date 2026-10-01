# SO-101 12 V: calibration to PPO

This repository calibrates the SO-101 and trains simulated position reaching or fixed pick-and-place using its fitted motors. Reaching supports fixed and random targets; pick-and-place controls all six joints with physical finger contact. See the [pick-and-place guide](ppo/README.md#fixed-pick-and-place) for its commands and limits. The selected real calibration is ready for an initial simulation run. No additional recording or refit is required for that run. Physical PPO deployment is not implemented; policy manifests retain `hardware_ready: false`.

For reaching, the policy controls shoulder pan, shoulder lift and elbow flex. Wrist flex, wrist roll and gripper hold their home commands. The fixed target is `(0.3145965, -0.0332624, 0.2507611)` metres in the base frame. Success means reaching within 15 mm with all joint speeds below 0.15 rad/s for one continuous second in a four-second near-workspace episode. New random tasks use the wide workspace, with a larger action range and 35-second episodes for the selected calibration.

**Start with the [PPO quickstart](docs/QUICKSTART.md)** for task descriptions, fixed/random training commands, robustness, visualization, evaluation, resume and the complete training option reference.

All commands below run from the repository root.

## Install two separate Python 3.12 environments

Install `uv`, Git and the system graphics/build libraries described in [the calibration guide](docs/CALIBRATION.md). Preserve any existing environment used for hardware work.

```bash
uv python install 3.12
uv venv --python 3.12 .venv-calibration
CC=gcc CXX=g++ uv pip install --python .venv-calibration/bin/python \
  --no-sources --torch-backend cpu \
  -r calibration/requirements-jetson.in -r calibration/requirements-interface.in
uv pip check --python .venv-calibration/bin/python

uv sync --project ppo --locked --extra cuda
```

Calibration uses MuJoCo 3.12.0; PPO uses the locked MuJoCo 3.7.0 / mjlab 1.3.0 / Warp stack. Both pin BAM revision `620a64fe67c1afe94fca81da73b128c7aed17c5f`. Keep `.venv-calibration`, `ppo/.venv`, and the existing root `.venv` separate. PPO alone is sufficient on a training machine receiving an already exported bundle.

The CUDA extra explicitly selects Torch 2.10.0+cu128. The supplied lockfile originally selected CPU Torch on Linux even without the CPU extra; the lock now includes a CUDA alternative while retaining the original dependency versions. CPU and CUDA extras are mutually exclusive. Keep the chosen extra on every `uv run` so synchronization preserves the backend.

For a machine without CUDA, install with `uv sync --project ppo --locked --extra cpu`, replace `--extra cuda` with `--extra cpu` on **every** subsequent `uv run --project ppo --locked` command, and use `--device cpu` for replay, checks and training. Use four environments for a CPU smoke run.

## Import the selected finished fit

| Artifact | Exact local path |
| --- | --- |
| Recorded session | `sessions/session-20260907-174056-522799423/` |
| Finished parameters | `sessions/session-20260907-174056-522799423/fit-20260907-174850-482396032/params.json` |
| Portable bundle | `data/calibrations/so101-12v-20260907-174850-482396032/` |

```bash
.venv-calibration/bin/python calibration/export_for_ppo.py \
  --session sessions/session-20260907-174056-522799423 \
  --params sessions/session-20260907-174056-522799423/fit-20260907-174850-482396032/params.json \
  --out data/calibrations/so101-12v-20260907-174850-482396032 \
  --all-replays
```

This bundle has already been exported in the migrated workspace. The exporter refuses existing destinations; skip this command if the bundle is present. For a future **finished** fit, substitute its session and `params.json`, and choose a new unique destination such as `data/calibrations/so101-12v-FIT_TIMESTAMP`. Multiple matching sessions can follow `--session`; `--all-replays` includes all held-out waveforms.

Export reads the accepted mapping, frozen model/meshes, finished parameters and complete run manifests/recordings. **`telemetry.jsonl` is unnecessary.** The archive remains under `sessions/`; incomplete runs and unrelated sessions are not imported. The bundle contains six BAM motor files, controller settings, delays, checksums, three held-out logs and simulator references, and existing fit diagnostics. Duplicate PNG/CSV plots and optimizer checkpoints remain in the raw archive. Export never refits, changes offsets, writes recordings or contacts hardware.

## Check the real calibration

```bash
uv run --project ppo --locked --extra cuda so101-replay \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --mjlab --device cuda:0 --output reports/real-replay-parity.json

uv run --project ppo --locked --extra cuda so101-check \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --mjlab --device cuda:0
```

Replay compares CPU and mjlab against all three calibration references. Its existing limits are 0.05-degree per-joint RMSE and 0.25-degree maximum discrepancy. Errors against real encoder measurements are reported separately: simulator parity does not establish physical accuracy. The model check verifies the original geometry, task bounds, observations, command rounding/delays and independent resets. Its known-FK-command endpoint error is a feasibility diagnostic; PPO may learn to compensate servo tracking error. First CUDA use compiles kernels; these checks need no display.

The supplied fit reduced training RMSE from 0.422 to 0.266 degrees and held-out RMSE from 0.453 to 0.264 degrees. Mapping review recorded only **0.088 degrees of wrist-roll coverage**, so that direction was weakly checked. Shoulder lift, elbow flex, wrist flex and wrist roll exhausted their 48-evaluation budgets. These are limitations to retain when judging results, not instructions to repeat calibration automatically. New-pose evaluation is a later validation step that writes reports without changing parameters; see [the calibration guide](docs/CALIBRATION.md).

## Train, export and resume

A longer CUDA run:

```bash
uv run --project ppo --locked --extra cuda so101-train \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --device cuda:0 --mode fixed --num-envs 1024 --iterations 300 \
  --run runs/fixed_calibrated
```

The iteration count is a starting budget, not a convergence guarantee. Reduce the environment count if GPU memory is limited. Each run saves `checkpoint.pt`, `policy.onnx` (including observation normalization), `policy_manifest.json`, configuration, export comparison results, a frozen `calibration/`, and a 20-episode CPU evaluation by default.

The initial smoke test uses four environments, two iterations and two evaluation episodes:

```bash
uv run --project ppo --locked --extra cuda so101-train \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --device cuda:0 --mode fixed --num-envs 4 --iterations 2 --eval-episodes 2 \
  --run runs/real_smoke

uv run --project ppo --locked --extra cuda so101-train \
  --calibration runs/real_smoke/calibration \
  --device cuda:0 --mode fixed --num-envs 4 --iterations 1 --eval-episodes 2 \
  --resume runs/real_smoke/checkpoint.pt --run runs/real_smoke_resume
```

These smoke directories already exist after verification; choose new names to repeat. Every training invocation requires a new run directory. For a longer resume, use the same pattern with `runs/fixed_calibrated/checkpoint.pt`, its frozen `calibration/`, a new output directory, and the desired additional iterations. **Resume requires the original bundle. Changed calibration means a new training run.** Earlier uncalibrated checkpoints cannot be resumed under this contract.

## Periodic training visualization

Add `--visualize show`, `save`, or `both` to the training command. Visualization is off by default. Each preview briefly pauses training and runs the current policy for one episode in a separate CPU simulation. The camera repeats; the starting pose also repeats unless `--start-mode random` is enabled with rotating previews. Random-mode previews rotate targets between captures by default; fixed-mode previews retain the original target. Add `--visualize-targets fixed` to repeat one target for direct progress comparisons. It uses deterministic actions and the copied observation normalizer, with nominal frozen calibration even for robust training.

Show and save a preview every 25 completed PPO rollout/update iterations:

```bash
uv run --project ppo --locked --extra cuda so101-train \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --device cuda:0 --num-envs 1024 --iterations 300 \
  --run runs/fixed_visualized \
  --visualize both --visualize-every 25
```

Record without a desktop on an NVIDIA machine:

```bash
env MUJOCO_GL=egl uv run --project ppo --locked --extra cuda so101-train \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --device cuda:0 --num-envs 1024 --iterations 300 \
  --run runs/fixed_recorded \
  --visualize save --visualize-every 25 --timelapse-speed 4
```

| Option | Default | Behavior |
| --- | --- | --- |
| `--visualize off/show/save/both` | `off` | Disable, display, record, or display and record previews |
| `--visualize-every N` | `25` | Preview every N completed rollout/update iterations |
| `--visualize-targets auto/fixed/rotate` | `auto` | Auto rotates random-mode previews; fixed repeats one target; fixed-mode training always retains its original target |
| `--timelapse-speed S` | `4` | Speed multiplier for the compilation; individual clips remain real time |

Add `--start-mode random` to opt into reachable random starts for training, evaluation and previews; new runs otherwise retain near-home starts. Resume and evaluation inherit the saved start mode. Fixed previews repeat one sampled start; rotating previews vary it. See the [random-start quickstart](docs/QUICKSTART.md#optional-random-starting-poses).

New `--mode random` runs default to `--workspace wide`, sampling 4,096 targets across the checked reachable region. Add `--start-mode random` to vary the initial hand pose over the same region. Use `--workspace near` for the original task. Wide policies require a fresh run; older checkpoints retain their near action mapping.

Random training already chooses targets independently for each environment on reset. Preview rotation changes only visualization, using the existing held-out preview bank and a local seeded generator. Each target stays still throughout its preview episode. Consecutive wide targets and random-start tool positions are at least 15 cm apart when possible (near targets: 3 cm); otherwise the farthest available target is selected. Target bank index and XYZ coordinates are shown in the overlay and per-episode metadata.

One rollout here collects 32 control steps from every training environment, followed by a PPO update; it is not one completed episode. Captures include the starting policy, each interval, and the final policy, without a duplicate final clip. A resumed invocation starts its own preview count at zero, showing the loaded policy first. Its target sequence also restarts; metadata records the source checkpoint and cumulative training control steps.

Saved output is in `runs/<run>/visualization/`:

- `episode_000000.mp4`, `episode_000025.mp4`, etc.: 1280×720, 25 FPS, up to the workspace episode duration each (near: 4 s; selected wide calibration: 35 s). Text shows rollouts completed in this invocation, episode time, distance and outcome.
- `manifest.json`: calibration identity, initial preview conditions, target-selection mode, per-episode target coordinates/metrics, frame counts, source checkpoint and any visualization errors.
- `timelapse.mp4`: completed clips in chronological order, at the selected speed, compiled when training exits. At the default 4× speed, each full episode occupies one second.

Showing an episode takes its simulated duration plus overhead (up to 35 seconds for wide previews). Recording runs as fast as simulation, rendering and encoding allow. The desktop window stays at the last preview between captures; closing it disables further display while training and requested recording continue. Display mode needs a working graphical desktop. Headless recording needs a working OpenGL backend (EGL in the NVIDIA example), but no desktop.

Rendering and encoding are checked before learning. A later visualization error is reported in the terminal and manifest, disables further previews, and lets training continue. On Ctrl-C, completed clips are retained and compilation is attempted; an incomplete current clip is discarded. This does not add training-checkpoint recovery on interruption. The training environments, policy state and RNG streams are not modified by previews.

## Evaluate ONNX and view

```bash
uv run --project ppo --locked --extra cuda so101-eval \
  --policy runs/real_smoke/policy.onnx \
  --episodes 2 --output reports/real-smoke-onnx-evaluation.json

uv run --project ppo --locked --extra cuda so101-eval \
  --policy runs/fixed_calibrated/policy.onnx \
  --episodes 100 --output reports/fixed-evaluation-100.json

uv run --project ppo --locked --extra cuda so101-eval \
  --policy runs/fixed_calibrated/policy.onnx --viewer --episodes 10

uv run --project ppo --locked --extra cuda so101-eval \
  --calibration data/calibrations/so101-12v-20260907-174850-482396032 \
  --baseline zero --episodes 100 --output reports/home-baseline.json

uv run --project ppo --locked --extra cuda tensorboard --logdir runs
```

Playback automatically loads the calibration frozen beside the policy. The viewer needs a graphical desktop; numerical evaluation is headless. Two smoke iterations test execution and export, not learned policy quality.

## Verification and repository layout

```bash
PYTHONPATH="$PWD/calibration" \
SO101_TEST_XML="$PWD/sessions/session-20260907-174056-522799423/model/model.xml" \
  .venv-calibration/bin/python -m unittest discover -s calibration/tests -v
uv run --project ppo --locked --extra cuda pytest -q ppo/tests

QT_QPA_PLATFORM=offscreen .venv-calibration/bin/python calibration/tests/gui_smoke.py \
  --config sessions/session-20260907-174056-522799423/config.json \
  --params sessions/session-20260907-174056-522799423/initial_params.json
```

| Location | Purpose |
| --- | --- |
| `calibration/` | Hardware interface, fitter, offline exporter and tests |
| `ppo/` | Training package, required assets/licenses, tests and `uv.lock` |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | Calibration and later new-pose evaluation |
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | Dynamics, policy contract and provenance details |
| [docs/validation/imported/](docs/validation/imported/README.md) | Supplied historical synthetic validation, preserved |
| [examples/synthetic_calibration/](examples/README.md) | Tracked synthetic fixture; requires `--allow-synthetic` |
| [data/README.md](data/README.md) | Local portable bundle storage and backup rules |
| `sessions/` | Unchanged local raw archive, ignored by Git |
| `runs/`, `reports/` | New generated policies and checks, ignored by Git |

Fresh verification results are in `reports/`, including `SUMMARY.md`; historical synthetic results are not results on this arm. Source, docs and synthetic fixtures are tracked. Real bundles, recordings, environments and generated results are excluded. Removed legacy files remain recoverable from Git history; migration also saved a temporary backup at `/tmp/so101-legacy-before-migration.tar.gz`.

## Backups and optional Docker

Back up the **entire** uniquely named bundle directory outside this checkout; it is sufficient for PPO and is not backed up by Git. Keep exported policies with `policy_manifest.json` and their frozen `calibration/`; retaining the complete run directory also preserves resume checkpoints and configuration. Back up raw sessions separately if future fitting or inspection is needed.

The retained Docker files provide an optional CUDA development shell:

```bash
docker compose build
docker compose run --rm codex
```

Inside `/workspace`, use the same root commands above and keep the two Python environments separate. The host needs NVIDIA Container Toolkit for `gpus: all`; installation is not performed by Compose. Docker binds this checkout, so use a separate checkout if host environments are incompatible with the container. Graphical viewer forwarding is not configured. Docker is optional and was not part of the native verification run.
