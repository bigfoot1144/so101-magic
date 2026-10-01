# Calibrated SO-101 PPO

Start with the [PPO quickstart](../docs/QUICKSTART.md) for both reaching modes, training commands, robustness, visualization, evaluation, resume and all CLI options. See [the root instructions](../README.md) for calibration import and replay checks.

From the repository root, the normal training command is:

```bash
uv run --project ppo --locked --extra cuda so101-train --calibration data/calibrations/so101-12v-20260907-174850-482396032 --mode fixed --num-envs 1024 --iterations 300 --run runs/fixed_calibrated
```

The calibration bundle is required. There is no fallback to the earlier 7.4 V motor fit. Both fixed-target and random-target modes train position reaching with a 15 mm, one-second hold criterion. Controller settings now match the supplied calibration code: P16, 8 degrees/second command slew, 50 Hz control and 2 ms physics.

For periodic training previews, add `--visualize show`, `save` or `both` and `--visualize-every 25`. Saved episodes and a compiled timelapse appear in the run's `visualization/` directory. See [visualization commands and headless setup](../README.md#periodic-training-visualization).

Random-mode previews rotate targets by default; training already varies its targets across resets. Add `--visualize-targets fixed` to repeat one preview target for progress comparisons. This option changes visualization only.

Add `--start-mode random` for reachable random starting poses in training, evaluation and previews. It is opt-in for new runs and inherited on resume/evaluation. Fixed previews repeat one sampled start. See the [random-start guide](../docs/QUICKSTART.md#optional-random-starting-poses).

New random tasks default to `--workspace wide`: 4,096 checked targets across the usable joint range, with longer episodes and an expanded action mapping. Use `--start-mode random` to broaden hand starts too. `--workspace near` retains the original small region. Wide policies need a fresh run; legacy checkpoints keep their near contract.

## Fixed pick and place

Add `--task pick-place` to train a new six-joint manipulation policy. This task requires `--mode fixed`, `--start-mode home`, and nominal physics: random starts, the wide workspace, `--robust`, and rotating preview targets are not supported. Each invocation needs a new output directory; change the example run name if it already exists.

From the repository root:

```bash
uv run --project ppo --locked --extra cuda so101-train --task pick-place --calibration data/calibrations/so101-12v-20260907-174850-482396032 --num-envs 1024 --iterations 500 --run runs/pick_place_fixed
```

The scene contains a 32 × 16 × 10 mm, 2.5 g rectangular brick and an 80 mm square on the floor. The brick starts at `(0.22, 0, 0.005)` m with its long axis along Y; the square is centered at `(0.22, 0.09)` m. The arm starts at a fixed, open-gripper pose above the brick. All six joints use the imported BAM calibration, including the wrist and gripper, with the same 8°/s command limit. Episodes last up to 60 seconds.

First fingertip contact earns **+1**. Holding the brick with both fingers at least 20 mm clear of the floor for 0.2 seconds earns **+10**. After a pickup, releasing the entire brick inside the square and letting it settle on the floor without robot contact for 0.5 seconds earns **+100** and ends the episode. Settling requires linear speed below 0.02 m/s and angular speed below 0.2 rad/s. Each bonus pays once per episode. A positive proximity reward is paid every control step, increasing as the fingertip approaches the brick: `0.02 × 0.4 × (1 − tanh(distance / 0.08))`. There is additional dense guidance for bilateral grasp, lift clearance, and carrying toward the square. Grasp/lift/carry guidance requires current physical contact; pushing the brick directly into the square does not count as success. All dense guidance is bounded to at most 60 reward over a full episode, below the +100 placement bonus. Small motion penalties retain smoothness without overwhelming approach rewards.

The brick moves through physical contact, with no attachment constraints. Collision meshes divide each original jaw into eight sections to preserve its fingertip profile. Elliptic friction with increased friction impedance prevents excessive solver slip on the light block. These are simulation contact parameters, not newly measured hardware properties.

Evaluate a policy, or check the physical scene with the scripted grasp-and-place baseline:

```bash
uv run --project ppo --locked --extra cuda so101-eval --policy runs/pick_place_fixed/policy.onnx --episodes 20 --viewer
uv run --project ppo --locked --extra cuda so101-eval --task pick-place --calibration data/calibrations/so101-12v-20260907-174850-482396032 --baseline oracle --episodes 1 --viewer
```

Training prints **Rollout mean step reward** and its touch/pickup/placement, shaping, and motion components after every PPO update, plus current tool-to-brick distance. These fresh values are also recorded under TensorBoard's `Rollout/` tags. The upstream **Mean reward** / `Train/mean_reward` is the return of the last 100 completed episodes. With fixed 3,000-step episodes and 32-step rollouts, that value can remain unchanged for about 94 updates; the logger explicitly reports when no new episodes finished. Neither statistic implies that pickup or placement has been learned—check the milestone metrics and previews too.

The scripted baseline uses inverse kinematics and ordinary calibrated servo actions; it does not train or supervise PPO. Evaluation reports touch, pickup, placement success, and failure rates. A successful scripted baseline verifies physical feasibility, not PPO convergence.

Existing `--visualize show|save|both` options work with the brick and square. Use `MUJOCO_GL=egl` for headless recording. Resume with `--resume runs/pick_place_fixed/checkpoint.pt`; task selection is inherited. Pick-place has its own versioned 46-observation / 6-action contract and requires a fresh policy rather than a reaching checkpoint. Reward version 2 uses positive state-based guidance and a tenfold lower movement penalty. Start a fresh run for this revision: its manifest schema is 5, and checkpoints from the earlier discounted-difference reward are rejected on resume.

This first version uses fixed positions, orientation, starting pose, and nominal physics. Random mode, random starts, rotating previews, the wide reaching workspace, and `--robust` are rejected for this task. Layout parameters live in `PickPlaceConfig`, and brick/target state is already observed so later layout randomization can reuse the policy interface.
