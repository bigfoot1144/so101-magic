# Third-party material

The MJCF and STL files under `src/so101_ppo/assets/so101` are from
TheRobotStudio/SO-ARM100, revision
`eecbe3e0a9ebb23e25ad7b2759b03884c6660903`, directory `Simulation/SO101`.
The original `so101_new_calib.xml` is renamed `so101.xml`, with unchanged content.
The upstream Apache 2.0 license is included in that directory.

BAM is installed from Rhoban/bam revision
`620a64fe67c1afe94fca81da73b128c7aed17c5f`; the upstream Apache 2.0 license is
included as `src/so101_ppo/assets/BAM_LICENSE`. The calibration interface's
experimental 12 V seed parameters cite revision
`ce4176b525326e7ac08c6dfd29299dfd01653191` in its source. The example bundle uses
those unfitted seed parameters, with documented synthetic test variations.
The packaged historical 7.4 V seed is retained as a reference asset; training requires an explicit calibration bundle. No trained policy is included.

The source in `../calibration/` was supplied by the user. Hardware and optimizer behavior are preserved. Offline evaluation provenance and its tests were updated to detect copied training recordings; export preserves original parameter bytes and fit diagnostics. Export/demo scripts and repository documentation support the PPO bridge.

BAM and mjlab are dependencies. RSL-RL provides PPO. MicroDuck RL is an
architectural reference; no MicroDuck assets or policy weights are included.
