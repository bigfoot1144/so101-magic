# SO-101 interactive calibration

Desktop pose matching, estimated gravity torque, coordinated motion recording, and automatic MuJoCo/BAM fitting for an assembled **12 V SO-101 follower**. Target: **Jetson Orin Nano Super, a supported JetPack 7 release, Ubuntu 24.04, Python 3.12, and one uv-managed `.venv`**.

You can command a small pose change, adjust the rendered joint angles to match the physical arm, and save the corresponding gravity calculation automatically. **Pose matching alone cannot identify motor torque, friction, inertia, or dynamic response.** The separate automatic motion run collects the data needed for a limited effective dynamics fit.

This is experimental hardware-control code. It has offline behavioral, numerical, and synthetic GUI checks; it has not been tested on your physical arm or Jetson. Establish accuracy from new real measurements before deploying a policy.

## Start here if LeRobot and alignment already work

Copy this package's four Python files, the requirements files, README, and tests into your existing repository. The package does not contain an arm calibration, `work/config.json`, or a `.venv`. Reuse your existing ones. Do **not** run `init` again in an existing `work` directory.

From the repository root, install the GUI into the same environment:

```bash
sudo apt update
sudo apt install -y libegl1 libopengl0 libxcb-cursor0 libxkbcommon-x11-0
uv pip install --python .venv/bin/python -r requirements-interface.in
uv pip check --python .venv/bin/python
```

First try the interface with **no serial connection or physical motion**:

```bash
uv run --no-project --python .venv/bin/python so101_calibrate.py gui \
  --config work/config.json --params work/seed_params.json --demo
```

Close the demo. Close LeRobot and other serial clients. In a terminal on the Jetson's graphical desktop, set `FOLLOWER_PORT` to the actual follower device and launch:

```bash
ls -l /dev/serial/by-id/
# Replace the example path with the follower's actual device:
export FOLLOWER_PORT=/dev/serial/by-id/REPLACE_WITH_FOLLOWER_DEVICE

uv run --no-project --python .venv/bin/python so101_calibrate.py gui \
  --port "$FOLLOWER_PORT" --config work/config.json --params work/seed_params.json
```

Two windows should appear: the control panel and the MuJoCo reference viewer. Continue with [Using the interface](#using-the-interface). There are no custom MuJoCo hotkeys in this interface; use the panel's buttons, sliders, or numeric fields.

## Installation from a fresh clone

The initial robot setup follows the official [LeRobot installation](https://huggingface.co/docs/lerobot/installation), [SO-101 setup](https://huggingface.co/docs/lerobot/so101), and [teleoperation](https://huggingface.co/docs/lerobot/il_robots#teleoperate) sequence. The commands below use the pinned LeRobot revision in this repository and uv throughout.

### 1. Check the Jetson and install one environment

Use [NVIDIA's board-specific downloads and installation instructions](https://developer.nvidia.com/embedded/jetpack/downloads) for a blank Jetson. Select a release that explicitly supports the Orin Nano; the major label “JetPack 7” alone does not establish board support. Do not reinstall a working Jetson just to run this package.

```bash
uname -m
cat /etc/os-release
cat /etc/nv_tegra_release
ldd --version
```

This dependency set expects `aarch64`, Ubuntu 24.04, and glibc 2.39 or newer. The pinned Qt ARM64 wheel needs glibc 2.39. JetPack 6 / Ubuntu 22.04 needs a different Qt build; these commands do not silently upgrade the OS.

```bash
sudo apt update
sudo apt install -y git curl ca-certificates unzip build-essential python3-dev linux-libc-dev \
  libgl1 libegl1 libopengl0 libglfw3 libxcb-cursor0 libxkbcommon-x11-0

# Only if uv is not installed:
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Substitute this repository's actual GitHub URL:
git clone https://github.com/YOUR_ORG/YOUR_REPO.git so101-calibration
cd so101-calibration

uv python install 3.12
uv venv --python 3.12 .venv
CC=gcc CXX=g++ uv pip install --python .venv/bin/python --no-sources --torch-backend cpu \
  -r requirements-jetson.in -r requirements-interface.in
uv pip check --python .venv/bin/python

uv run --no-project --python .venv/bin/python -c "import numpy, scipy, mujoco, bam, lerobot, scservo_sdk; from PySide6 import QtWidgets; print('Imports OK')"
uv run --no-project --python .venv/bin/python -m unittest discover -s tests -v
```

Keep the provided dependency pins. BAM requires Python 3.12; this LeRobot revision needs NumPy below 2.3. `--no-sources --torch-backend cpu` keeps this calibration installation on CPU PyTorch. Installing CUDA policy inference or training dependencies requires choosing a build compatible with your exact JetPack release; this package does not configure GPU policy deployment. The selected LeRobot extras cover setup, calibration, and this teleoperation test.

Every subsequent command runs from this repository root and explicitly uses the same `.venv`. Do not run Python with `sudo`. On a working installation, add `requirements-interface.in`; do not recreate the environment.

### 2. Discover ports and configure motors only if necessary

Secure the follower base. Use the correct 12 V follower supply and the leader's separately specified supply. Support the arm before torque can be released. Keep the workspace clear and motor power accessible.

```bash
uv run --no-project --python .venv/bin/python lerobot-find-port
ls -l /dev/serial/by-id/

# Replace both examples with the identified devices:
export FOLLOWER_PORT=/dev/serial/by-id/REPLACE_WITH_FOLLOWER_DEVICE
export LEADER_PORT=/dev/serial/by-id/REPLACE_WITH_LEADER_DEVICE
```

The port-discovery command prompts you to unplug/reconnect an adapter. Run it for each arm. Set these variables in each new terminal. If the devices are owned by `dialout` and your account lacks access:

```bash
sudo usermod -aG dialout "$USER"
```

Log out and back in, return to the repository, and set the port variables again.

**Skip motor setup if these assembled arms already work.** Brand-new motors may share an ID; follow LeRobot's prompts with exactly one motor connected to the adapter at a time:

```bash
uv run --no-project --python .venv/bin/python lerobot-setup-motors \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT"

uv run --no-project --python .venv/bin/python lerobot-setup-motors \
  --teleop.type=so101_leader --teleop.port="$LEADER_PORT"
```

Restore the daisy chain afterward. This may require isolating cables, but does not require removing horns or dismantling the links. The follower must have IDs 1–6 in this order, at 1,000,000 baud: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.

### 3. Calibrate follower and leader in LeRobot

Reuse an existing calibration only if it belongs to these exact assembled arms. Back it up before intentional recalibration. Homing/travel calibration is different from the MuJoCo zero alignment below.

```bash
mkdir -p calibration/follower calibration/leader

uv run --no-project --python .venv/bin/python lerobot-calibrate \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" \
  --robot.id=my_follower_12v --robot.calibration_dir="$PWD/calibration/follower"

uv run --no-project --python .venv/bin/python lerobot-calibrate \
  --teleop.type=so101_leader --teleop.port="$LEADER_PORT" \
  --teleop.id=my_leader --teleop.calibration_dir="$PWD/calibration/leader"
```

Follow the supported middle-pose prompt, then gently move the requested joints through their travel without forcing stops. Follow the printed wrist-roll exception; a 0–4095 calibration range does not mean cables allow unlimited rotation. Existing-calibration prompts distinguish reuse from recalibration.

Expected outputs are `calibration/follower/my_follower_12v.json` and `calibration/leader/my_leader.json`. Always use the **follower** file for MuJoCo. Do not change homing midway through a dataset.

### 4. Pass a normal teleoperation test

A leader is required for this test. A follower-only owner can use the later interface, but cannot perform leader teleoperation without a leader. Put the arms in similar, supported, clear poses before starting.

```bash
uv run --no-project --python .venv/bin/python lerobot-teleoperate \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" \
  --robot.id=my_follower_12v --robot.calibration_dir="$PWD/calibration/follower" \
  --teleop.type=so101_leader --teleop.port="$LEADER_PORT" \
  --teleop.id=my_leader --teleop.calibration_dir="$PWD/calibration/leader"
```

This uses the pinned LeRobot defaults, including 60 Hz and P=16, I=0, D=32. It omits the earlier `max_relative_target=5` clipping and 30 Hz override that made teleoperation sluggish. Begin with small movements and check the joint identities/directions, then move several joints together. Test the empty gripper away from its stops. Support the follower before Ctrl+C releases torque. Teleoperation is a functional check, not a measurement of sim-to-real accuracy.

The interface's current BAM model supports **P-only control**. Before its dynamics tests, repeat a small teleoperation test with the same command plus `--robot.position_d_coefficient=0`. If the arm cannot hold the intended poses acceptably with P=16, I=D=0, stop here: a D=32 actuator model needs to be implemented and identified instead. A P-only fit does not validate normal D=32 deployment.

### 5. Create the MuJoCo model and review directions

Close teleoperation. Run `init` once in a new output directory:

```bash
uv run --no-project --python .venv/bin/python so101_sysid.py init \
  --calibration "$PWD/calibration/follower/my_follower_12v.json" --out work

uv run --no-project --python .venv/bin/python so101_sysid.py inspect \
  --port "$FOLLOWER_PORT" --config work/config.json --out after_calibration.json

uv run --no-project --python .venv/bin/python so101_sysid.py align \
  --port "$FOLLOWER_PORT" --config work/config.json
```

`init` downloads the pinned SO-101 MJCF and meshes and creates `seed_params.json`. It refuses to overwrite `work/config.json`. If that config already exists, skip `init`. An `inspect` output name must also be new. `init` and `inspect` do not render; `align` opens the viewer after the torque-OFF prompt.

With the physical arm supported and torque OFF, move one joint at a time. Check both direction and link orientation at several angles. The alignment controls avoid MuJoCo's ordinary digit/letter shortcuts:

| Key | Alignment action |
|---|---|
| F8 / F9 | Previous / next joint |
| F10 | Reverse selected joint sign |
| Home / End | Offset − / + 0.5° |
| Insert / Delete | Offset − / + 5° |
| F12 | Save the reviewed mapping |

Focus the 3D view when using these alignment keys. The new desktop interface uses widgets instead. Homing is already included in the reported encoder positions; the code does not add the EEPROM homing offset a second time.

Run the model-dependent offline checks:

```bash
SO101_TEST_XML="$PWD/work/so101_bam.xml" \
  uv run --no-project --python .venv/bin/python -m unittest discover -s tests -v
```

Then launch the interface using the commands at the top of this README.

## Using the interface

### A. Command a pose, match the simulator, and calculate gravity torque

1. The connection begins **read-only with torque OFF**. If another program left motors enabled, the application refuses the connection; support the arm and use torque-OFF `align` first. No other process may own the same serial port.
2. Manually support a clear pose away from all travel limits. Partially open the empty gripper. Check **Arm supported; workspace clear**, then click **Enable hold**. It snapshots the original settings, prepares P=16, I=D=0, reads the current pose again, seeds every goal before enabling any motor, and holds that pose. Remove support and keep clear once it is holding.
3. To command a new pose, click **Preview current sent goal**, adjust the joint sliders/numbers, inspect the preview and physical clearance, then click **MOVE ROBOT to preview**. Changed motors move together. This is joint-position control, not Cartesian gripper dragging or collision-free path planning.
4. Once the real arm settles, click **Freeze measurement for matching**. This latches a median encoder pose from a stable window and copies it into the editor. Adjust the **edited reference** to the physical link orientations. These edits do not move the real arm. Use numbers for measured reference angles.
5. The **Gravity estimate** column updates automatically from the edited pose. Click **Save matched pose + gravity**. The file records the frozen encoder pose, your reference angles, measurement method, and gravity estimates for both poses. If the physical arm moved after freezing, the capture is rejected so the pair cannot silently mix different poses.
6. Repeat at several distinct poses. Three pairs permit an offset fit; approximately five to ten well-measured poses with varied joint orientations provide a more useful check. A direction needs meaningful travel to assess: the report flags coverage below 15°.
7. Support the arm and click **Supported: torque OFF**. Click **Fit offsets and save candidate**. Review `mapping_report.json` in the session folder. The fit uses median offsets, preserves your reviewed signs, and flags conflicting signs, limited coverage, or large residuals. It never writes homing/limits to the motors and never overwrites the active mapping.

The mapping fit creates `mapping_candidate.json`. Close the interface and launch a new session with that path after reviewing the report:

```bash
# Substitute the session folder printed in the panel:
uv run --no-project --python .venv/bin/python so101_calibrate.py gui \
  --port "$FOLLOWER_PORT" \
  --config sessions/SESSION_FOLDER/mapping_candidate.json \
  --params work/seed_params.json
```

The candidate references that session's frozen model. Keep the folder together. Changing coordinates invalidates prior dynamics data; record new runs with the accepted mapping. If the existing alignment already meets independently measured accuracy, keep it and skip the extra offset-fitting captures.

You can also capture supported torque-OFF poses. For gravity matching, a gauge or camera must measure the physical links independently of the servo encoder. **Enter MuJoCo joint coordinates**, not an unconverted inclinometer's absolute world angle; parent frames, joint axes and zero conventions matter. Selecting a measurement method labels the data; it does not perform camera calibration or convert external measurements automatically.

### B. Collect dynamics automatically

Use the accepted mapping and actual deployment payload. For an initial empty-arm test, remove objects and keep the gripper partially open. Do not rest a link on a support during recording; support forces would contaminate the identification.

1. Enable a clear supported pose and let the arm settle without hand support.
2. Open **Automatic dynamics** and click **RECORD coordinated run**.
3. Attend the approximately 47-second run. It moves all six motors simultaneously using different frequencies and phases. It records two training motions and a different held-out motion, returning to the center goal between them. You do not choose joints, frequencies, filenames or per-motor gain options.
4. Repeat at two or more different clear poses before fitting. Small moves can use the panel. For a substantially different pose, support, disable, reposition manually, then enable again. If the five-minute enabled timer is near expiry, disable/re-enable before another run. Watch the timer while matching poses too.
5. Support and disable torque. Click **FIT + validate all complete runs**. The hardware worker locks out motor enable while a separate process fits the six motor models. Incomplete/aborted runs are excluded.

The bounded initial search estimates each joint's **effective resistance/gain scale, constant friction, viscous friction, and command delay**, while keeping CAD mass/inertia, torque constant and armature fixed. It minimizes full-arm position error over all training logs, one parameter block at a time. Coordinated movements save operator effort; they do not guarantee every parameter is uniquely identifiable. Small motions also constrain viscous friction and latency weakly.

Fitting may take tens of minutes or longer on the Jetson, particularly with multiple poses. It runs without holding the real arm. The default is an initial search of 48 evaluations per joint; inspect optimizer budget/near-bound notes in the report. If fit quality is limited by optimizer progress, rerun offline with a larger budget instead of repeating hardware setup:

```bash
uv run --no-project --python .venv/bin/python so101_calibrate.py fit \
  --session sessions/SESSION_FOLDER --params work/seed_params.json \
  --max-evals 120 --passes 2
```

For recordings spread across application sessions, repeat `--session` for each folder. Their calibration, mapping, geometry and physics must agree. A new output directory is created automatically. Further optimization should use training data; repeated tuning to the same held-out trace makes it development data rather than an independent test.

### C. Assess the result, then test a new pose

The fit folder contains:

| Output | Meaning |
|---|---|
| `params.json` | Candidate motor parameters, frozen before held-out evaluation |
| `REPORT.md` / `report.json` | Initial versus fitted errors, coverage, scope, and limitations |
| `validation-*.png` | Sent targets, real encoder trajectories, initial model, fitted model |
| `validation-*.csv` | Time in seconds and explicitly labeled joint angles in degrees |
| `fit_progress.json` | Per-joint optimization progress and parameter-bound notes |

The dynamic prediction is a free rollout initialized once from each trial's measured starting state. It is not reset to the encoder trajectory at every frame. The live pose viewer is kinematic and must not be treated as this validation. The parameters stay in JSON; changing a static MJCF gain is not equivalent to using the BAM controller and command-delay model.

Check each joint's RMSE, p95 and maximum error against your task's tolerance. An overall improvement is insufficient if a critical joint becomes worse. The built-in split holds out a waveform at each training center, **not an entire pose, speed envelope or payload**.

For an independent pose test, close and reopen the interface with the same accepted mapping, record at a new clear pose, and disable torque. Do not press FIT on that new session. Evaluate it using the frozen fitted file:

```bash
uv run --no-project --python .venv/bin/python so101_calibrate.py evaluate \
  --session sessions/NEW_SESSION_FOLDER \
  --params sessions/TRAINING_SESSION_FOLDER/FIT_FOLDER/params.json \
  --baseline work/seed_params.json
```

This evaluates all motions in the new session without optimizing. The command rejects known training log paths and mismatched fitted coordinates/models. Keep a genuinely new recording; copying a training log to a different filename does not make it independent.

If the new-pose test fails, collect training examples that cover the missing operating conditions, revisit geometry/controller assumptions, or use a richer actuator model. Reserve another new test afterward. The current small-motion interface does not validate fast movements or contact tasks.

## What this can and cannot tell you about torque

The live calculation is the model's gravity load at zero velocity:

`estimated gravity torque = qfrc_bias(q, qvel=0)`

It is the signed joint torque needed to balance modeled gravity, excluding friction, contacts, cable forces, and unmodeled payload. Radians are used internally; hinge torques are in N·m. The robot's base orientation relative to gravity, link masses and centers of mass must be correct. The calculation does **not** read a calibrated torque sensor and is **never sent as a motor torque command**.

MuJoCo inverse dynamics can calculate model-dependent required forces for a specified position, velocity and acceleration. A manually matched still pose supplies only position. It cannot reverse-calculate unique motor constants or friction. The servo's load/current registers are retained as raw telemetry; this package does not assume they are calibrated output-shaft torque measurements. Firmware P=16 is not 16 N·m/rad.

## What else effective sim-to-real needs

| Requirement | What remains beyond this interface |
|---|---|
| Metric geometry and zero accuracy | Check tool/link poses using a digital gauge, measured fixture, or calibrated external tracking. Fit/check base and tool frames, link dimensions and joint axes when errors vary with pose. Visual matching alone is approximate. |
| Relevant dynamics | Test the intended poses, speeds, payload, supply conditions and controller. This baseline covers P=16, I=D=0 and small free-space movements. It does not model D=32, overload timing, backlash, compliance or directional/load-dependent friction. |
| Grasping/contact | Validate gripper geometry, compliance, contact friction and grip force with the intended objects. Current free-space replay disables geom contacts. |
| Deployment integration | Use the same radians/signs/offsets, motor controller, target timing, command delays, limits and BAM actuator logic in the task simulator. Convert LeRobot's gripper normalization separately if your policy uses percent opening. |
| Vision-based policies | Calibrate camera intrinsics/extrinsics, observations and timestamps. Validate image/task differences and use appropriate visual/dynamics randomization. |
| Final acceptance | Measure new real tool trajectories and task outcomes, including contact where relevant, against task-specific tolerances. Joint-encoder agreement alone is not a millimeter-level tool accuracy measurement. |

You do not need a perfect physical model for every sim-to-real task. A useful baseline is independent coordinate/geometry measurement, a validated effective position-response model, and randomization over the remaining uncertainty. More demanding torque-sensitive or contact tasks need additional measurements. Calibrated force/torque sensing or carefully measured known loads and lever arms are options for torque identification without taking the arm apart. A camera with rigid link markers can replace much of the manual reference matching, but requires camera/marker-frame calibration and a geometric estimator; that automation is not implemented here.

## Temperature warnings and fault recovery

The supplied temperature report was 29–33 °C, with agreement between group and individual byte-63 readings while torque was OFF. This shows normal readings at that moment; it does not explain or invalidate the earlier under-load cutoff. See `TEMPERATURE_REVIEW.md` for the report-specific details.

Temperature is now **warning-only at 50 °C or higher**, as requested. There is no software stop based solely on the temperature value, including above the former 55 °C cutoff. The control panel shows a large red banner naming the motors and their recent peak readings. A brief spike stays visible for five seconds after the last high sample; sustained high readings keep the banner visible. Motion and recording continue. The legacy CLI recorder prints temperature warnings to stderr, rate-limited to one every five seconds.

Every temperature sample is still logged, and the worker adds `temperature_warning` events to `telemetry.jsonl`. Status alarms, voltage and motion checks still stop operation. The motor's own protection settings are unchanged. If another fault occurs, the fault handler still attempts torque-off, restores settings and independently reads temperatures.

Other stops include status alarms, voltage outside 9–12.6 V, >8° tracking error, travel/model limits, a >250 ms loop/read stall, missing UI heartbeat for 1.5 seconds, and the five-minute enabled timeout. These are experimental software limits, not certified safety functions. The measured and commanded encoder limits do not prevent self-collision, table collision, cable snagging or a released arm falling.

**STOP / torque OFF**, closing the MuJoCo viewer, and loss of the UI trigger worker shutdown and torque release. Normal disable restores the pre-hold gains/profile and leaves torque off. A serial/power failure can prevent software from disabling a servo; support the arm and cut motor power if shutdown is not confirmed. Restart the application after a fault or STOP, once the cause is addressed.

The interface rejects poses within 3° of encoder/model limits. In the submitted report, the elbow and gripper were inside that margin and wrist flex was very close. With torque OFF, manually support/reposition to a clear middle pose and partially open the gripper. The application does not automatically drive away from an unverified end-stop pose.

## Files, reproducibility and software checks

Each application launch creates a unique `sessions/session-*` folder (or `DEMO-*`). It freezes the model and meshes, mapping and starting parameters. Settings snapshots precede writes; `telemetry.jsonl` records initial hold, measurements and sent goals. Completed runs contain their own command timestamps, telemetry, controller settings and split labels. JSON angles are radians unless a field explicitly says degrees. Hardware telemetry and goals include host timing, not synchronized onboard servo clocks.

The frozen model uses about 16 MB per session for the current stock meshes; telemetry and plots add space. Preserve complete experiment folders and record dependency versions. Do not commit `.venv`, local caches, or large telemetry/model copies accidentally; choose your repository's ignore/data-storage policy.

```bash
uv pip freeze --python .venv/bin/python > requirements-resolved.txt
SO101_TEST_XML="$PWD/work/so101_bam.xml" \
  uv run --no-project --python .venv/bin/python -m unittest discover -s tests -v

# Optional: exercise Qt + the worker without a display or physical serial bus:
QT_QPA_PLATFORM=offscreen uv run --no-project --python .venv/bin/python tests/gui_smoke.py \
  --config work/config.json --params work/seed_params.json
```

Demo mode is a first-order toy plant, not an SO-101 dynamics model. Its files are marked synthetic. Hardware fitting/evaluation rejects synthetic data by default, and synthetic candidate mappings are not accepted for a physical session. Do not use demo fits for a real robot.

Qt's offscreen platform is only for the software smoke test. For normal use, launch from a local graphical desktop. If Qt reports a missing xcb dependency, verify the OS libraries above. If a headless shell has no display, use the desktop instead. The ordinary MuJoCo UI still has its built-in visibility keys; this package does not reassign them.

## Primary references

- [LeRobot installation](https://huggingface.co/docs/lerobot/installation) and [SO-101 setup/calibration](https://huggingface.co/docs/lerobot/so101).
- [Pinned LeRobot source](https://github.com/huggingface/lerobot/tree/3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e), used to verify actual CLI defaults and register mapping.
- [Pinned SO-101 model](https://github.com/TheRobotStudio/SO-ARM100/tree/eecbe3e0a9ebb23e25ad7b2759b03884c6660903/Simulation/SO101).
- [BAM fitting and validation](https://bam.readthedocs.io/en/latest/identification/fitting.html) and [pinned controller implementation](https://github.com/Rhoban/bam/tree/620a64fe67c1afe94fca81da73b128c7aed17c5f). This package's limited whole-arm fitter is distinct from BAM's bench workflow.
- [MuJoCo dynamics](https://mujoco.readthedocs.io/en/stable/computation/index.html) and [passive viewer](https://mujoco.readthedocs.io/en/3.12.0/python.html#passive-viewer).
- [Qt wheel/platform details](https://pypi.org/project/PySide6-Essentials/6.11.2/) and [NVIDIA JetPack installation](https://developer.nvidia.com/embedded/jetpack/downloads).
