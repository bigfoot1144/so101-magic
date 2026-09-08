# Calibration workflow

All commands run from the repository root. The selected fit is finished and already contains the data required for initial simulation training. Its raw session is `sessions/session-20260907-174056-522799423/`; parameters are in `fit-20260907-174850-482396032/params.json` beneath that session. Its portable bundle is `data/calibrations/so101-12v-20260907-174850-482396032/`. Follow [the root README](../README.md) to export and train without repeating calibration.

## A. Later new-pose evaluation

The fit's three held-out waveforms cover the recorded centers, not an entirely independent pose. A later evaluation can record a new session at different clear poses using the accepted mapping and settings. Support and disable the arm afterward; do not fit this evaluation session.

```bash
.venv-calibration/bin/python calibration/so101_calibrate.py evaluate \
  --session sessions/REPLACE_WITH_NEW_SESSION \
  --params sessions/session-20260907-174056-522799423/fit-20260907-174850-482396032/params.json \
  --baseline sessions/session-20260907-174056-522799423/initial_params.json \
  --out reports/new-pose-evaluation
```

Evaluation writes reports and plots without changing fitted parameters. It is a later validation step, not a prerequisite for the initial simulation run. Preserve the raw mapping, training and evaluation sessions separately from portable PPO bundles.

New fits record SHA-256 hashes of training recordings. Evaluation rejects matching content even when copied and renamed. Legacy fits remain readable and unchanged; their recorded session/run/filename identities are compared independently of parent directories. Legacy fits without hashes cannot identify a recording whose session/run/filename identity has also been changed, so retain archive names and provenance.

The supplied mapping report at `sessions/session-20260907-171042-327746454/mapping_report.json` has only 0.088 degrees of wrist-roll coverage and flags weak direction coverage. The selected fit exhausted its 48-evaluation budget for shoulder lift, elbow flex, wrist flex and wrist roll; shoulder pan and gripper reported optimizer success. Keep these limitations with the results without automatically refitting or changing offsets.

## B. Start calibration in this package if needed

This section assumes you already have the follower's LeRobot calibration JSON and configured motor IDs. Reuse those files. Initial motor setup and LeRobot calibration are described in [the original reference guide](../calibration/REFERENCE_README.md) if you are setting up a different arm.

Commands below start in the repository root. The GUI dependency pins follow the attached interface's JetPack 7 / Ubuntu 24.04 / Python 3.12 setup. An existing working calibration environment is preferable to reinstalling it.

### Install the separate calibration environment

On the Jetson desktop:

```bash
sudo apt update
sudo apt install -y git curl build-essential python3-dev linux-libc-dev \
  libgl1 libegl1 libopengl0 libglfw3 libxcb-cursor0 libxkbcommon-x11-0 \
  libglib2.0-0t64 libfontconfig1 libdbus-1-3

uv python install 3.12
uv venv --python 3.12 .venv-calibration

CC=gcc CXX=g++ uv pip install --python .venv-calibration/bin/python \
  --no-sources --torch-backend cpu \
  -r calibration/requirements-jetson.in \
  -r calibration/requirements-interface.in

uv pip check --python .venv-calibration/bin/python
```

The separate calibration environment is at the repository root. PPO later has its own `ppo/.venv`. Calibration does not require the training GPU stack.

### Set the existing calibration path and follower port

```bash
export FOLLOWER_CAL=/absolute/path/to/my_follower_12v.json
export FOLLOWER_PORT=/dev/serial/by-id/REPLACE_WITH_FOLLOWER_DEVICE
```

If you need to identify the adapter:

```bash
.venv-calibration/bin/lerobot-find-port
ls -l /dev/serial/by-id/
```

For optional leader teleoperation, use the command in the supplied `ORIGINAL_WORKFLOW.md` and retain **`--robot.position_d_coefficient=0`**, so the functional check uses P16/I0/D0. The older reference README's teleoperation example uses LeRobot's D32 default and is not the controller used by this fit. Close teleoperation before using the calibration interface.

### Create and inspect the model

```bash
.venv-calibration/bin/python calibration/so101_sysid.py init \
  --calibration "$FOLLOWER_CAL" --out calibration/work

export CONFIG="$PWD/calibration/work/config.json"

.venv-calibration/bin/python calibration/so101_sysid.py inspect \
  --port "$FOLLOWER_PORT" --config "$CONFIG" \
  --out calibration/initial_inspection.json
```

`init` creates the model, meshes, config and 12 V seed parameters. It refuses to overwrite an existing setup. Reuse your existing config if you already completed this step for the same arm.

### Review joint directions and offsets

```bash
.venv-calibration/bin/python calibration/so101_sysid.py align \
  --port "$FOLLOWER_PORT" --config "$CONFIG"
```

Support the arm before the tool disables torque. Move one physical joint at a time and compare several poses with the rendered arm.

| Key | Action |
| --- | --- |
| F8 / F9 | Previous / next joint |
| F10 | Reverse selected direction |
| Home / End | Offset by -/+ 0.5 degrees |
| Insert / Delete | Offset by -/+ 5 degrees |
| F12 | Save the reviewed mapping |

### Run the interface

Optional software tests and demo:

```bash
PYTHONPATH="$PWD/calibration" SO101_TEST_XML="$PWD/calibration/work/so101_bam.xml" \
  .venv-calibration/bin/python -m unittest discover -s calibration/tests -v

.venv-calibration/bin/python calibration/so101_calibrate.py gui \
  --config "$CONFIG" --params calibration/work/seed_params.json \
  --out sessions --demo
```

Demo output is synthetic and is never a motor calibration. Close it before opening a hardware session.

```bash
.venv-calibration/bin/python calibration/so101_calibrate.py gui \
  --port "$FOLLOWER_PORT" --config "$CONFIG" \
  --params calibration/work/seed_params.json --out sessions
```

For pose refinement, enable a settled hold, remove external support during measurements, freeze the measurement, adjust the simulated reference to match the physical links, and save the matched pose. Collect at least three distinct poses. Support and disable the arm before fitting offsets. Review `mapping_report.json`, then use the accepted `mapping_candidate.json` as `CONFIG` in a new session. Keep its parent session directory because it contains the model assets.

For dynamics, open **Automatic dynamics**, record the approximately 47-second coordinated run, and repeat at two or more clear starting poses. Do not hold a link during a recording. The interface's existing support, hold-timer and stop behavior are unchanged.

After recording, support and disable the arm, then use **FIT + validate all complete runs**. Alternatively, run the fitter offline:

```bash
.venv-calibration/bin/python calibration/so101_calibrate.py fit \
  --session sessions/REPLACE_WITH_DYNAMICS_SESSION \
  --params calibration/work/seed_params.json
```

For fitting multiple sessions, repeat the `--session` flag. All sessions must use the same mapping and model.

A finished fit can be exported immediately for simulation. New-pose evaluation in section A remains a later step. To import a future fit, use a uniquely named directory:

```bash
.venv-calibration/bin/python calibration/export_for_ppo.py \
  --session sessions/REPLACE_WITH_FINISHED_SESSION \
  --params sessions/REPLACE_WITH_TRAINING_SESSION/fit-REPLACE_WITH_FIT/params.json \
  --out data/calibrations/so101-12v-REPLACE_WITH_UNIQUE_FIT_ID --all-replays
```

Continue with PPO installation and training in [the main README](../README.md).

## What does not need repeating

PPO training, evaluation and checkpoint resume do not require another hardware recording. Re-export and retrain when the accepted mapping, fitted motor model or effective controller settings change. Changing a payload or moving much faster than the recorded motions calls for validation of that new operating condition, not an automatic rerun of every setup step.
