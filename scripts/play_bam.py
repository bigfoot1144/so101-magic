import math
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from bam.model import load_model
from bam.mujoco import MujocoController


ROOT = Path(__file__).resolve().parents[1]

XML_PATH = (
    ROOT
    / "generated"
    / "so101"
    / "scene_bam.xml"
)

MOTOR_NAME = "feetech_sts3215_12V"
FRICTION_MODEL = "m1"
SUPPLY_VOLTAGE = 12.0

BAM_PARAMS_PATH = (
    ROOT
    / "params"
    / "bam"
    / MOTOR_NAME
    / f"{FRICTION_MODEL}.json"
)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

STEP_DEGREES = 3.0
STEP_RADIANS = math.radians(STEP_DEGREES)


def get_joint_id(
    model: mujoco.MjModel,
    name: str,
) -> int:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if joint_id == -1:
        raise RuntimeError(
            f"Joint {name!r} was not found."
        )

    return joint_id


def get_joint_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
) -> float:
    joint_id = get_joint_id(model, name)
    qpos_index = model.jnt_qposadr[joint_id]

    return float(data.qpos[qpos_index])


def get_joint_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
) -> float:
    joint_id = get_joint_id(model, name)
    dof_index = model.jnt_dofadr[joint_id]

    return float(data.qvel[dof_index])


def get_joint_range(
    model: mujoco.MjModel,
    name: str,
) -> tuple[float, float]:
    joint_id = get_joint_id(model, name)

    if not model.jnt_limited[joint_id]:
        return -math.inf, math.inf

    return (
        float(model.jnt_range[joint_id][0]),
        float(model.jnt_range[joint_id][1]),
    )


def clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:
    return max(lower, min(upper, value))


def print_model_info(
    model: mujoco.MjModel,
) -> None:
    print()
    print("MuJoCo model")
    print("------------")
    print(f"timestep:  {model.opt.timestep:.6f} s")
    print(f"joints:    {model.njnt}")
    print(f"dofs:      {model.nv}")
    print(f"actuators: {model.nu}")
    print()

    print("Controlled joints")
    print("-----------------")

    for index, name in enumerate(
        JOINT_NAMES,
        start=1,
    ):
        lower, upper = get_joint_range(
            model,
            name,
        )

        if math.isfinite(lower) and math.isfinite(upper):
            range_text = (
                f"[{math.degrees(lower):+.1f}°, "
                f"{math.degrees(upper):+.1f}°]"
            )
        else:
            range_text = "unlimited"

        print(
            f"{index}: "
            f"{name:16s} "
            f"{range_text}"
        )

    print()


def create_bam_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> MujocoController:
    print(
        f"Loading BAM motor={MOTOR_NAME!r} "
        f"friction_model={FRICTION_MODEL!r}"
    )

    if not BAM_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Missing BAM parameters: {BAM_PARAMS_PATH}\n"
            "Add the vendored STS3215 12 V parameter file first."
        )

    bam_model = load_model(
        str(BAM_PARAMS_PATH)
    )

    # BAM's STS3215 actuator class defaults to 7.4 V.
    # The identified parameters below are for the 12 V servo,
    # so explicitly set the physical supply voltage.
    bam_model.actuator.vin = SUPPLY_VOLTAGE

    controller = MujocoController(
        model=bam_model,
        actuator=JOINT_NAMES,
        mujoco_model=model,
        mujoco_data=data,
    )

    # BAM 1.0.2 compatibility workaround.
    #
    # STS3215Actuator.compute_control() expects
    # q_target_smooth, but BAM 1.0.2 does not initialize
    # it when loading bundled actuator parameters.
    #
    # Initialize it from the physical joint state.
    actuator = bam_model.actuator

    if not hasattr(
        actuator,
        "q_target_smooth",
    ):
        actuator.q_target_smooth = np.array(
            data.qpos[
                controller.qpos_indexes
            ],
            dtype=float,
            copy=True,
        )

        print(
            "Applied BAM 1.0.2 STS3215 "
            "q_target_smooth compatibility fix."
        )

    # Start desired positions at the actual robot pose.
    controller.q_target = np.array(
        data.qpos[
            controller.qpos_indexes
        ],
        dtype=float,
        copy=True,
    )

    controller.last_ts = float(data.time)

    print("BAM model loaded successfully.")

    return controller


def make_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> dict[str, float]:
    return {
        name: get_joint_position(
            model,
            data,
            name,
        )
        for name in JOINT_NAMES
    }


def main() -> None:
    if not XML_PATH.exists():
        raise FileNotFoundError(
            f"{XML_PATH} does not exist.\n"
            "Run:\n"
            "  uv run python "
            "scripts/prepare_bam_model.py"
        )

    print(f"Loading {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(model)

    print_model_info(model)

    mujoco.mj_resetData(
        model,
        data,
    )

    mujoco.mj_forward(
        model,
        data,
    )

    controller = create_bam_controller(
        model,
        data,
    )

    targets = make_targets(
        model,
        data,
    )

    ranges = {
        name: get_joint_range(
            model,
            name,
        )
        for name in JOINT_NAMES
    }

    state = {
        "selected": 0,
        "paused": False,
        "reset_requested": False,
    }

    def selected_joint() -> str:
        return JOINT_NAMES[
            state["selected"]
        ]

    def print_joint_status(
        name: str,
    ) -> None:
        position = get_joint_position(
            model,
            data,
            name,
        )

        velocity = get_joint_velocity(
            model,
            data,
            name,
        )

        target = targets[name]

        marker = (
            " <"
            if name == selected_joint()
            else ""
        )

        print(
            f"{name:16s} "
            f"q={position:+.4f} rad "
            f"({math.degrees(position):+7.2f}°)  "
            f"target={target:+.4f} rad "
            f"({math.degrees(target):+7.2f}°)  "
            f"dq={velocity:+.4f} rad/s"
            f"{marker}"
        )

    def print_selected_joint() -> None:
        name = selected_joint()

        print()
        print(
            f"Selected "
            f"{state['selected'] + 1}/"
            f"{len(JOINT_NAMES)}: "
            f"{name}"
        )

        print_joint_status(name)

    def print_all_joint_status() -> None:
        print()
        print("Joint state")
        print("-----------")

        for name in JOINT_NAMES:
            print_joint_status(name)

        print()

    def select_previous_joint() -> None:
        state["selected"] = (
            state["selected"] - 1
        ) % len(JOINT_NAMES)

        print_selected_joint()

    def select_next_joint() -> None:
        state["selected"] = (
            state["selected"] + 1
        ) % len(JOINT_NAMES)

        print_selected_joint()

    def change_target(
        delta: float,
    ) -> None:
        name = selected_joint()

        lower, upper = ranges[name]

        targets[name] = clamp(
            targets[name] + delta,
            lower,
            upper,
        )

        print_joint_status(name)

    def key_callback(
        keycode: int,
    ) -> None:
        # MuJoCo's viewer reserves 0-5 for geom groups
        # and most letter keys for visualization flags.
        #
        # 6-9 and F8-F10 are unused by the documented
        # viewer shortcut map.

        if keycode == glfw.KEY_6:
            select_previous_joint()

        elif keycode == glfw.KEY_7:
            select_next_joint()

        elif keycode == glfw.KEY_8:
            change_target(
                -STEP_RADIANS
            )

        elif keycode == glfw.KEY_9:
            change_target(
                +STEP_RADIANS
            )

        elif keycode == glfw.KEY_F8:
            print_all_joint_status()

        elif keycode == glfw.KEY_F9:
            state["paused"] = not state[
                "paused"
            ]

            print(
                "Simulation paused."
                if state["paused"]
                else "Simulation running."
            )

        elif keycode == glfw.KEY_F10:
            state[
                "reset_requested"
            ] = True

    print()
    print("BAM playground")
    print("--------------")
    print(f"motor:          {MOTOR_NAME}")
    print(f"friction model: {FRICTION_MODEL}")
    print(f"BAM params:     {BAM_PARAMS_PATH}")
    print(f"supply voltage: {SUPPLY_VOLTAGE:.1f} V")
    print()
    print("Controls")
    print("--------")
    print("6    previous joint")
    print("7    next joint")
    print(
        f"8    target -"
        f"{STEP_DEGREES:.0f} degrees"
    )
    print(
        f"9    target +"
        f"{STEP_DEGREES:.0f} degrees"
    )
    print("F8   print all joint states")
    print("F9   pause/unpause")
    print("F10  reset")
    print()
    print(
        "0-5 remain MuJoCo's normal "
        "geom visibility controls."
    )
    print()
    print(
        "Click the MuJoCo viewer first, "
        "then use the keys."
    )
    print()

    print_selected_joint()

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            if state["reset_requested"]:
                mujoco.mj_resetData(
                    model,
                    data,
                )

                mujoco.mj_forward(
                    model,
                    data,
                )

                controller = create_bam_controller(
                    model,
                    data,
                )

                targets = make_targets(
                    model,
                    data,
                )

                state[
                    "reset_requested"
                ] = False

                print()
                print("Simulation reset.")

                print_all_joint_status()

            if not state["paused"]:
                for name in JOINT_NAMES:
                    controller.set_q_target(
                        name,
                        targets[name],
                    )

                # BAM:
                #
                # target position
                # current position
                # current velocity
                #       ↓
                # STS3215 controller model
                #       ↓
                # motor torque + friction
                controller.update()

                # MuJoCo:
                #
                # torque
                # + gravity
                # + inertia
                # + contacts
                # + constraints
                #       ↓
                # next q / qvel
                mujoco.mj_step(
                    model,
                    data,
                )

            viewer.sync()

            elapsed = (
                time.perf_counter()
                - step_start
            )

            remaining = (
                model.opt.timestep
                - elapsed
            )

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()