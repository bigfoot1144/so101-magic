"""Periodic deterministic policy previews, isolated from PPO training state."""
from contextlib import contextmanager
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .contract import CONTROL_DT, EPISODE_STEPS, SUCCESS_DISTANCE
from .cpu import CpuArm
from .evaluate import run_episode
from .workspace import workspace_config, task_bank
from .starts import EVAL_START_SEED, home_start, resolve_start_mode, sample_start, start_bank, start_metadata

WIDTH, HEIGHT, FPS = 1280, 720, 25
WIDE_HEADER_HEIGHT = 180
MIN_TARGET_SEPARATION = 0.03


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def snapshot_policy(actor):
    # Never call runner.get_inference_policy(): it changes the training modules.
    # The pinned export wrapper copies the MLP and normalizer, excluding the
    # stochastic distribution's cached autograd graph after a PPO update.
    # Copy the wrapper as well so even output-module state cannot be shared.
    model = copy.deepcopy(actor.as_onnx(verbose=False)).cpu().eval()

    def policy(obs):
        with torch.inference_mode():
            return model(torch.from_numpy(np.asarray(obs, np.float32)[None])).numpy()[0]
    return policy


def configure_camera(camera, workspace="near", task="reach"):
    camera.lookat[:] = [0.20, 0, 0.16]
    camera.distance, camera.elevation, camera.azimuth = 0.75, -25, 135
    if workspace == "wide":
        camera.lookat[:] = [0., 0., .22]
        camera.distance = 1.25
    if task == "pick-place":
        camera.lookat[:] = [.22, .045, .08]
        camera.distance, camera.elevation = .65, -35


def add_target(scene, goal):
    import mujoco
    index = scene.ngeom
    if index >= scene.maxgeom:
        raise RuntimeError("No scene geometry slot for preview target")
    mujoco.mjv_initGeom(scene.geoms[index], mujoco.mjtGeom.mjGEOM_SPHERE,
                      np.full(3, SUCCESS_DISTANCE), np.asarray(goal, float),
                      np.eye(3).ravel(), np.array([0.2, 0.85, 0.3, 0.6]))
    scene.ngeom += 1


def overlay_text(rollouts, state, target_index=None, target=None, start_index=None):
    if "picked" in state:
        status = "FAILED" if state["failed"] else "SUCCESS" if state["success"] else "PICK AND PLACE"
        if state["step"] == state.get("episode_steps") and not state["success"] and not state["failed"]:
            status = "TIMEOUT - NO SUCCESS"
        return (f"Rollouts completed in this run: {rollouts}\n"
                f"Episode: {state['time_s']:.2f} s   Target distance: {state['distance_m']*1000:.1f} mm\n"
                f"Touch: {state['touched']}   Pickup: {state['picked']}   Placed: {state['placed']}\n{status}")
    status = "FAILED" if state["failed"] else "SUCCESS" if state["success"] else "REACHING"
    if state["step"] == state.get("episode_steps", EPISODE_STEPS) and status == "REACHING":
        status = "TIMEOUT - NO SUCCESS"
    text = (f"Rollouts completed in this run: {rollouts}\n"
            f"Episode: {state['time_s']:.2f} s   Distance: {state['distance_m'] * 1000:.1f} mm\n"
            f"{status}   |   Deterministic preview / nominal calibration")
    if target_index is not None:
        text += (f"\nTarget #{target_index} (bank index)   XYZ: "
                 f"({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) m")
    if start_index is not None:
        text += f"\nStart #{start_index} (reachable pose bank)"
    return text


def video_writer(path):
    import imageio.v2 as imageio
    return imageio.get_writer(str(path), format="FFMPEG", fps=FPS, codec="libx264",
                              pixelformat="yuv420p", macro_block_size=1)


def compile_timelapse(clips, output, speed):
    """Stream frames at a constant output FPS; speed changes sampling, not FPS."""
    import imageio.v2 as imageio
    output = Path(output)
    temporary = output.with_name(output.stem + ".partial.mp4")
    source_index, output_index = 0, 0
    try:
        with video_writer(temporary) as writer:
            for clip in clips:
                with imageio.get_reader(str(clip), format="FFMPEG") as reader:
                    for frame in reader:
                        while math.floor(output_index * speed) == source_index:
                            writer.append_data(frame)
                            output_index += 1
                        source_index += 1
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output_index


class PreviewLogger:
    """Delegate all upstream logging, then notify once per completed PPO update."""
    def __init__(self, logger, on_iteration):
        self._logger, self._on_iteration = logger, on_iteration

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def log(self, *args, **kwargs):
        result = self._logger.log(*args, **kwargs)
        self._on_iteration()
        return result


class TrainingVisualization:
    def __init__(self, calibration, run, mode, every=25, speed=4.0,
                 task_mode="fixed", source_checkpoint=None, target_selection="auto",
                 start_mode="home", workspace="near", task="reach"):
        if mode not in ("show", "save", "both"):
            raise ValueError("Expected show, save or both")
        if every < 1 or not math.isfinite(speed) or speed <= 0:
            raise ValueError("Preview interval and timelapse speed must be positive")
        if target_selection not in ("auto", "fixed", "rotate"):
            raise ValueError("Expected auto, fixed or rotate preview targets")
        self.task = task
        if task == "pick-place":
            from .pick_place import validate_options
            validate_options(task_mode, start_mode, workspace, targets=target_selection)
        self.workspace = workspace_config(calibration, workspace)
        self.calibration = calibration
        self.min_target_separation = .15 if workspace == "wide" else MIN_TARGET_SEPARATION
        self.start_mode = resolve_start_mode(start_mode)
        self.start_index = None
        self.target_selection = ("rotate" if task_mode == "random" and target_selection != "fixed"
                                 else "fixed")
        self.root = Path(run) / "visualization"
        self.root.mkdir(parents=True, exist_ok=False)
        self.every, self.speed = every, speed
        self.show, self.save = mode in ("show", "both"), mode in ("save", "both")
        self.completed = 0
        self.last_capture = None
        self.disabled = False
        self.view = self.renderer = None
        self.records, self.errors = [], []
        self.metadata = {"schema": 1, "task": task, "calibration_sha256": calibration.digest,
            "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
            "task_mode": task_mode, **start_metadata(self.start_mode), **self.workspace.metadata(), "seed": 2026, "bank_seed": 54321,
            "target_selection_requested": target_selection, "target_selection": self.target_selection,
            "min_target_separation_m": self.min_target_separation if self.target_selection == "rotate" else None,
            "min_start_separation_m": .15 if workspace == "wide" and self.start_mode == "random" and self.target_selection == "rotate" else None,
            "policy": "deterministic; copied actor and frozen observation normalizer",
            "dynamics": "CPU MuJoCo + BAM; nominal frozen calibration",
            "rollout_unit": "one PPO rollout/update iteration; counts local to this invocation",
            "every": every, "fps": FPS, "width": WIDTH, "height": HEIGHT,
            "timelapse_speed": speed, "hardware_ready": False}
        try:
            with preserve_rng():
                self._prepare(calibration, task_mode)
            self._write_manifest()
        except BaseException:
            self._close_resources()
            raise

    def _prepare_targets(self, task_mode):
        # Preserve the old first target and pose, including the RNG draw order.
        self._target_rng = np.random.default_rng(2026)
        self._goals, _ = task_bank(self.calibration, task_mode, seed=54321, workspace=self.workspace.name)
        self.target_index = int(self._target_rng.integers(len(self._goals)))
        self.goal = self._goals[self.target_index].copy()
        self.q = home_start(self._target_rng, self.robot.bounds)
        self.metadata.update(initial_q_rad=self.q.tolist(), initial_target_m=self.goal.tolist(),
                             initial_target_index=self.target_index, target_bank_size=len(self._goals))
        if self.target_selection == "fixed":
            # Retain the previous shared-target field only when it is shared.
            self.metadata["target_m"] = self.goal.tolist()

    def _prepare_starts(self, calibration):
        if self.start_mode == "random":
            self._starts = start_bank(calibration, EVAL_START_SEED, self.workspace.name)
            if self.workspace.name == "wide":
                self._start_goals = task_bank(calibration, "random", EVAL_START_SEED, "wide")[0]
            self._start_rng = np.random.default_rng(2026)
            self.start_index, self.q = sample_start(self._starts, self._start_rng)
            self.metadata.update(initial_q_rad=self.q.tolist(), initial_start_index=self.start_index,
                                 start_bank_size=len(self._starts))

    def _select_next_start(self):
        if self.start_mode == "random" and self.target_selection == "rotate":
            if self.workspace.name == "wide":
                distances = np.linalg.norm(self._start_goals - self._start_goals[self.start_index], axis=1)
                candidates = np.flatnonzero(distances >= .15)
                self.start_index = (int(self._start_rng.choice(candidates)) if len(candidates)
                                    else int(np.argmax(distances)))
                self.q = self._starts[self.start_index].copy()
            else:
                self.start_index, self.q = sample_start(self._starts, self._start_rng, self.start_index)

    def _select_next_target(self):
        if self.target_selection == "fixed":
            return
        distances = np.linalg.norm(self._goals.astype(float) - self.goal, axis=1)
        candidates = np.flatnonzero(distances >= self.min_target_separation)
        self.target_index = (int(self._target_rng.choice(candidates)) if len(candidates)
                             else int(np.argmax(distances)))
        self.goal = self._goals[self.target_index].copy()

    def _recording_frame(self, scene_frame):
        # Give wide scenes their own viewport, so a high target cannot disappear
        # behind the status overlay. Keep the public MP4 dimensions unchanged.
        if self.workspace.name == "wide":
            return np.pad(scene_frame, ((WIDE_HEADER_HEIGHT, 0), (0, 0), (0, 0)))
        return scene_frame

    def _prepare(self, calibration, task_mode):
        import mujoco
        if self.task == "pick-place":
            from .pick_place import CONFIG
            from .pick_place_cpu import CpuPickPlace
            self.robot = CpuPickPlace(calibration)
            self.goal = np.array(CONFIG.target_position)
            self.q = np.array(CONFIG.start_q)
            self.target_index = 0
            for key in ('workspace', 'workspace_sampler_version', 'reward_distance_scale_m'):
                self.metadata.pop(key, None)
            self.metadata.update(initial_q_rad=self.q.tolist(), target_m=self.goal.tolist(),
                brick_position_m=list(CONFIG.brick_position), square_size_m=CONFIG.square_size,
                episode_seconds=CONFIG.episode_seconds)
        else:
            self.robot = CpuArm(calibration, workspace=self.workspace.name)
            self._prepare_targets(task_mode)
            self._prepare_starts(calibration)
        self.robot.reset(self.goal, self.q)
        self.camera = mujoco.MjvCamera()
        configure_camera(self.camera, self.workspace.name, self.task)
        if self.save:
            from PIL import ImageFont
            self.font = ImageFont.load_default(size=23)
            self.robot.model.vis.global_.offwidth = WIDTH
            self.robot.model.vis.global_.offheight = HEIGHT
            try:
                render_height = HEIGHT - WIDE_HEADER_HEIGHT if self.workspace.name == "wide" else HEIGHT
                self.renderer = mujoco.Renderer(self.robot.model, height=render_height, width=WIDTH)
                self.renderer.update_scene(self.robot.data, camera=self.camera)
                frame = self._recording_frame(self.renderer.render())
                probe = self.root / "preflight.mp4"
                try:
                    with video_writer(probe) as writer:
                        writer.append_data(frame)
                    import imageio.v2 as imageio
                    with imageio.get_reader(str(probe), format="FFMPEG") as reader:
                        assert reader.get_data(0).shape == (HEIGHT, WIDTH, 3)
                finally:
                    probe.unlink(missing_ok=True)
            except Exception as exc:
                raise RuntimeError("Preview recording preflight failed. For headless NVIDIA recording, "
                                   "start with env MUJOCO_GL=egl; check OpenGL and FFmpeg support.") from exc
        if self.show:
            import glfw
            if not glfw.init():
                raise RuntimeError("Preview display requires a working graphical desktop. Use --visualize save "
                                   "with MUJOCO_GL=egl for headless recording.")
            import mujoco.viewer
            self.view = mujoco.viewer.launch_passive(self.robot.model, self.robot.data)
            configure_camera(self.view.cam, self.workspace.name, self.task)

    def _write_manifest(self):
        document = {**self.metadata, "episodes": self.records, "errors": self.errors}
        path = self.root / "manifest.json"
        tmp = path.with_suffix(".partial.json")
        tmp.write_text(json.dumps(document, indent=2) + "\n")
        tmp.replace(path)

    def _error(self, message):
        self.errors.append(message)
        print(f"Visualization: {message}", file=sys.stderr, flush=True)
        try:
            self._write_manifest()
        except Exception as exc:
            print(f"Visualization: could not write error metadata: {exc}", file=sys.stderr)

    def iteration(self, runner):
        self.completed += 1
        if self.completed % self.every == 0:
            self.capture(runner)

    def capture(self, runner):
        if self.disabled or self.last_capture == self.completed or not (self.show or self.save):
            return
        print(f"Preview: {self.completed} rollouts completed in this run", flush=True)
        try:
            with preserve_rng():
                if self.last_capture is not None:
                    self._select_next_target()
                    self._select_next_start()
                policy = snapshot_policy(runner.alg.get_policy())
                self._episode(policy, runner.env.unwrapped.common_step_counter)
            self.last_capture = self.completed
        except Exception as exc:
            self.disabled = True
            self._error(f"disabled after preview failure: {type(exc).__name__}: {exc}; training continues")

    def _episode(self, policy, training_steps):
        from PIL import Image, ImageDraw
        path = self.root / f"episode_{self.completed:06d}.mp4"
        temporary = path.with_name(path.stem + ".partial.mp4")
        writer = None
        frame_count = 0
        try:
            if self.save:
                writer = video_writer(temporary)

            def on_step(robot, state):
                nonlocal frame_count
                text = overlay_text(self.completed, state, self.target_index, self.goal, self.start_index)
                if self.view is not None:
                    if not self.view.is_running():
                        self.view.close()
                        self.view = None
                        self.show = False
                        print("Preview window closed; training and requested recording continue.", flush=True)
                    else:
                        with self.view.lock():
                            self.view.user_scn.ngeom = 0
                            if self.task == "reach":
                                add_target(self.view.user_scn, self.goal)
                        self.view.set_texts((None, None, text, ""))
                        self.view.sync()
                # 50 Hz control -> 25 FPS. Include an early failure's last frame.
                if writer is not None and (state["step"] % 2 == 0 or state["failed"] or state["success"]):
                    self.renderer.update_scene(robot.data, camera=self.camera)
                    if self.task == "reach":
                        add_target(self.renderer.scene, self.goal)
                    frame = Image.fromarray(self._recording_frame(self.renderer.render()))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((12, 12, 950, 172 if self.start_index is not None else 142), fill=(15, 20, 27))
                    draw.multiline_text((24, 20), text, font=self.font, fill="white", spacing=5)
                    writer.append_data(np.asarray(frame))
                    frame_count += 1
                if self.show:
                    time.sleep(max(0, CONTROL_DT - (time.monotonic() - state["wall_step_start"])))

            result = run_episode(self.robot, policy, self.goal, self.q, on_step)
            if writer is not None:
                writer.close()
                writer = None
                temporary.replace(path)
            record = {"completed_rollouts": self.completed, "training_control_steps": training_steps,
                      "clip": path.name if self.save else None, "frames": frame_count,
                      "target_index": self.target_index, "target_m": self.goal.tolist(),
                      "start_index": self.start_index, "initial_q_rad": self.q.tolist(), **result}
            self.records.append(record)
            self._write_manifest()
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    self._error(f"incomplete clip cleanup failed: {exc}")
            try:
                temporary.unlink(missing_ok=True)
            except Exception as exc:
                self._error(f"could not remove incomplete clip: {exc}")

    def _close_resources(self):
        for resource in (self.view, self.renderer):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    print(f"Visualization cleanup: {exc}", file=sys.stderr)
        self.view = self.renderer = None

    def close(self):
        # Called from training's finally block, including on Ctrl-C. Never mask
        # a training exception with a video encoding or cleanup exception.
        self._close_resources()
        clips = [self.root / row["clip"] for row in self.records if row["clip"]]
        if clips:
            try:
                frames = compile_timelapse(clips, self.root / "timelapse.mp4", self.speed)
                self.metadata.update(timelapse="timelapse.mp4", timelapse_frames=frames)
                self._write_manifest()
            except Exception as exc:
                self._error(f"timelapse compilation failed: {type(exc).__name__}: {exc}; episode clips retained")
