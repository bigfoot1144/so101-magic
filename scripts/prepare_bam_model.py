from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

UPSTREAM_DIR = (
    ROOT
    / "external"
    / "SO-ARM100"
    / "Simulation"
    / "SO101"
)

OUTPUT_DIR = ROOT / "generated" / "so101"

SOURCE_ROBOT = UPSTREAM_DIR / "so101_new_calib.xml"
SOURCE_SCENE = UPSTREAM_DIR / "scene.xml"

OUTPUT_ROBOT = OUTPUT_DIR / "so101_bam.xml"
OUTPUT_SCENE = OUTPUT_DIR / "scene_bam.xml"


ACTUATORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def replace_actuator_block(xml: str) -> str:
    start = xml.index("<actuator>")
    end = xml.index("</actuator>", start) + len("</actuator>")

    lines = ["<actuator>"]

    for name in ACTUATORS:
        lines.append(
            f'  <motor name="{name}" joint="{name}" gear="1"/>'
        )

    lines.append("</actuator>")

    replacement = "\n".join(lines)

    return xml[:start] + replacement + xml[end:]


def rewrite_mesh_directory(xml: str) -> str:
    assets = UPSTREAM_DIR / "assets"

    return xml.replace(
        'meshdir="assets"',
        f'meshdir="{assets}"',
    )


def main() -> None:
    if not SOURCE_ROBOT.exists():
        raise FileNotFoundError(
            f"Missing SO-101 MJCF: {SOURCE_ROBOT}\n"
            "Did you initialize the SO-ARM100 submodule?"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    robot_xml = SOURCE_ROBOT.read_text()

    robot_xml = replace_actuator_block(
        robot_xml
    )

    robot_xml = rewrite_mesh_directory(
        robot_xml
    )

    OUTPUT_ROBOT.write_text(
        robot_xml
    )

    scene_xml = SOURCE_SCENE.read_text()

    scene_xml = scene_xml.replace(
        'include file="so101_new_calib.xml"',
        'include file="so101_bam.xml"',
    )

    OUTPUT_SCENE.write_text(
        scene_xml
    )

    print(f"Created {OUTPUT_ROBOT}")
    print(f"Created {OUTPUT_SCENE}")


if __name__ == "__main__":
    main()