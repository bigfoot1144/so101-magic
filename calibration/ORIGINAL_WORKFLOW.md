Start from your existing calibration files. You do **not** need to repeat motor ID setup or LeRobot calibration.

These commands assume your repository is:

```bash
cd ~/projects/so-101/so101-calibration

```

Use a terminal on the Jetson’s graphical desktop for the viewer.

**1. Install the system dependencies and copy in the code**

Download [so101\_calibration\_interface.zip](sandbox:/workspace/scratch/33237082a5ef/output/so101_calibration_interface.zip) into `~/Downloads`.

```bash
sudo apt update
sudo apt install -y git curl unzip build-essential python3-dev linux-libc-dev \
  libgl1 libegl1 libopengl0 libglfw3 libxcb-cursor0 libxkbcommon-x11-0

unzip -o ~/Downloads/so101_calibration_interface.zip -d /tmp/so101-interface
cp -a /tmp/so101-interface/so101_calibration_interface/. .

```

The package contains no robot calibration files, so this preserves `./calibration`.

**2. Install everything into the main ****`.venv`**

If uv is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

```

Create the environment if you do not already have it:

```bash
uv python install 3.12
uv venv --python 3.12 .venv

```

If your existing `.venv` already uses Python 3.12, reuse it instead.

Install the dependencies:

```bash
CC=gcc CXX=g++ uv pip install --python .venv/bin/python \
  --no-sources --torch-backend cpu \
  -r requirements-jetson.in -r requirements-interface.in

uv pip check --python .venv/bin/python

```

This installs the Python dependencies together, including LeRobot, MuJoCo, BAM and the interface. Calibration and fitting use CPU computation.

**3. Identify your calibration files and ports**

Check the calibration directory:

```bash
ls -R ./calibration

```

Using the filenames from your earlier setup:

```bash
export FOLLOWER_CAL="$PWD/calibration/follower/my_follower_12v.json"
export LEADER_CAL="$PWD/calibration/leader/my_leader.json"

test -f "$FOLLOWER_CAL" && echo "Follower calibration found"
test -f "$LEADER_CAL" && echo "Leader calibration found"

```

Change those two paths if your files are elsewhere inside `./calibration`.

Find the adapters:

```bash
uv run --no-project --python .venv/bin/python lerobot-find-port
ls -l /dev/serial/by-id/

```

Follow the discovery prompt for each adapter. Set the actual paths:

```bash
export FOLLOWER_PORT=/dev/serial/by-id/REPLACE_WITH_FOLLOWER_DEVICE
export LEADER_PORT=/dev/serial/by-id/REPLACE_WITH_LEADER_DEVICE

```

These variables must be set again when you open a new terminal.

If you get a permissions error and the ports belong to `dialout`:

```bash
sudo usermod -aG dialout "$USER"

```

Log out and back in, then return to the repository and set the variables again.

**4. Run a short teleoperation test using the existing calibrations**

Start with similar, supported poses and a clear workspace. This command derives each calibration ID from its filename:

```bash
uv run --no-project --python .venv/bin/python lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$(basename "$FOLLOWER_CAL" .json)" \
  --robot.calibration_dir="$(dirname "$FOLLOWER_CAL")" \
  --robot.position_d_coefficient=0 \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$(basename "$LEADER_CAL" .json)" \
  --teleop.calibration_dir="$(dirname "$LEADER_CAL")"

```

This uses **P=16, I=0, D=0**, matching the interface’s supported controller. It keeps the normal 60 Hz default and omits the target clipping that made the earlier test sluggish.

Check:

- Each leader joint moves the correct follower joint in the correct direction.
- Several joints move smoothly together.
- The follower holds the tested poses without sustained oscillation or excessive sag.
- The empty gripper responds correctly.

If prompted about calibration, reuse the existing calibration. If the program unexpectedly asks you to perform a new calibration, stop and check the file path and ID first.

Support the follower before pressing **Ctrl+C**. Close teleoperation completely before continuing.

If P-only control cannot hold the intended poses acceptably, stop here. The current model does not support identifying D=32 operation.

**5. Create the MuJoCo model**

Because you are starting with only LeRobot calibrations, run:

```bash
uv run --no-project --python .venv/bin/python so101_sysid.py init \
  --calibration "$FOLLOWER_CAL" \
  --out work

export CONFIG="$PWD/work/config.json"

```

This downloads the model and meshes and creates:

- `work/config.json`
- `work/so101_bam.xml`
- `work/seed_params.json`

If `work/config.json` already exists, `init` will refuse to overwrite it. Reuse it only if it belongs to this follower calibration.

Take a read-only inspection:

```bash
uv run --no-project --python .venv/bin/python so101_sysid.py inspect \
  --port "$FOLLOWER_PORT" \
  --config "$CONFIG" \
  --out initial_inspection.json

```

Neither command opens a rendered window.

**6. Establish the joint directions and approximate offsets**

```bash
uv run --no-project --python .venv/bin/python so101_sysid.py align \
  --port "$FOLLOWER_PORT" \
  --config "$CONFIG"

```

Support the arm before confirming torque OFF. The MuJoCo window should then appear.

Move one physical joint at a time and check that its rendered motion matches.

| KeyAction       |                                    |
| --------------- | ---------------------------------- |
| F8 / F9         | Previous / next joint              |
| F10             | Reverse selected joint’s direction |
| Home / End      | Offset − / + 0.5°                  |
| Insert / Delete | Offset − / + 5°                    |
| F12             | Save the reviewed mapping          |

Check multiple poses, not just one. Press **F12** after reviewing all six joints, then close the viewer.

This establishes the coordinate mapping. It does not identify motor dynamics.

**7. Check the software and try the interface in demo mode**

```bash
SO101_TEST_XML="$PWD/work/so101_bam.xml" \
  uv run --no-project --python .venv/bin/python \
  -m unittest discover -s tests -v

uv run --no-project --python .venv/bin/python so101_calibrate.py gui \
  --config "$CONFIG" \
  --params work/seed_params.json \
  --demo

```

You should see the control panel and MuJoCo viewer. Try the preview controls and buttons. Demo mode never opens the physical serial port.

Close the demo before proceeding.

**8. Launch the physical interface and match poses**

```bash
uv run --no-project --python .venv/bin/python so101_calibrate.py gui \
  --port "$FOLLOWER_PORT" \
  --config "$CONFIG" \
  --params work/seed_params.json

```

For the first physical run:

1. With torque OFF, support the arm in a clear pose away from its travel limits. Partially open the empty gripper. Your earlier inspection showed the elbow and gripper too close to their stops.
2. Check **Arm supported; workspace clear**, then click **Enable hold**.
3. Once it holds, remove your support and keep clear.
4. Click **Preview current sent goal**, make a small adjustment, then click **MOVE ROBOT to preview**.
5. After the arm settles, click **Freeze measurement for matching**.
6. Adjust the rendered reference to match the physical links. These edits do not move the arm.
7. Click **Save matched pose + gravity**.

The gravity estimate updates automatically as you edit. It is calculated from the CAD model, not measured motor torque.

Repeat across several distinct poses. Three captures permit an offset fit; five to ten carefully measured poses provide a more useful check. Independent angle or pose measurements improve accuracy over visual matching.

Watch the five-minute hold timer. Support and disable the arm before it expires. Faults, STOP, or closing the viewer can also release torque.

**9. Fit and accept the coordinate offsets**

Support the arm and click **Supported: torque OFF**, then:

- Click **Fit offsets and save candidate**.
- Click **Open results folder**.
- Review `mapping_report.json`, especially residuals and direction warnings.

The candidate is saved as `mapping_candidate.json`. It does not overwrite your current mapping.

After accepting the result, close the application and set `CONFIG` to that candidate’s actual path:

```bash
export CONFIG="$PWD/sessions/REPLACE_WITH_POSE_SESSION/mapping_candidate.json"

```

Keep that session folder intact because the candidate references its frozen model.

If the initial alignment already meets your independently checked accuracy requirement, you can keep `work/config.json` and skip this refinement.

**10. Record the dynamics automatically**

Launch a new session with the accepted mapping:

```bash
uv run --no-project --python .venv/bin/python so101_calibrate.py gui \
  --port "$FOLLOWER_PORT" \
  --config "$CONFIG" \
  --params work/seed_params.json

```

Then:

1. Enable a clear pose and let it settle without hand support.
2. Open **Automatic dynamics**.
3. Click **RECORD coordinated run**.
4. Attend the approximately **47-second** recording. All six motors move together through two training motions and one held-out motion.
5. Repeat at two or more different clear poses in the same application session.

Use small commanded moves between nearby poses. For a substantially different starting pose, support, disable, reposition manually, and enable again.

Do not support a link or hold an object by hand during recording. Those external forces would contaminate the fit.

**11. Fit and inspect the results**

Support the arm, click **Supported: torque OFF**, then click:

**FIT + validate all complete runs**

The fitter runs offline and locks out motor enable. It can take tens of minutes or longer on the Jetson.

In the resulting `fit-*` directory, inspect:

| FileWhat to check   |                                                     |
| ------------------- | --------------------------------------------------- |
| `REPORT.md`         | Initial versus fitted error for every joint         |
| `validation-*.png`  | Measured motion versus independent simulated motion |
| `fit_progress.json` | Optimizer limits and parameters near their bounds   |
| `params.json`       | The resulting candidate actuator parameters         |

The live pose viewer is not dynamic validation. Use these recorded-motion plots and errors.

**12. Validate at a new pose without refitting**

Close and reopen the interface using the same `CONFIG`. Record another coordinated run at a new clear pose, then support and disable the arm.

Do **not** click FIT on this new session.

Evaluate using the previously fitted parameters:

```bash
uv run --no-project --python .venv/bin/python so101_calibrate.py evaluate \
  --session sessions/REPLACE_WITH_NEW_SESSION \
  --params sessions/REPLACE_WITH_TRAINING_SESSION/REPLACE_WITH_FIT_FOLDER/params.json \
  --baseline work/seed_params.json

```

This tests the frozen model against new recordings without optimizing it.

You now have a coordinate mapping, estimated actuator response, and validation results. To use them in your task simulator, retain the same mapping, **P=16/I=D=0 controller**, BAM actuator logic, command delays and timing. The parameter JSON does not automatically modify another MuJoCo application.

Grasping still needs contact/gripper validation, and accurate tool placement needs independent geometry measurements. The code has passed offline tests, but its physical behavior and accuracy must be established on your arm.