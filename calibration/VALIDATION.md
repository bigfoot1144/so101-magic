# Software validation

Validated in this workspace on Linux x86_64, Python 3.12.13, NumPy 2.2.6, MuJoCo 3.12.0, the pinned BAM implementation, and PySide6-Essentials 6.11.2. No physical servo connection was made. No Jetson GPU/display or real motor calibration accuracy is claimed.

- 25 offline tests passed with the actual downloaded SO-101 model available. These include coordinate conversion, analytic gravity versus potential-energy differentiation, frozen-model equivalence, partial enable failure, goal-before-enable ordering, travel/speed limits, stale UI/loop handling, warning-only temperature spikes during recording, the 50 °C warning boundary, five-second banner persistence, and preserved status/voltage faults, settings restoration, stable pose capture, and a complete coordinated training/validation recording through a toy plant.
- The fitting integration test runs the actual SciPy optimizer and MuJoCo/BAM rollouts on deliberately synthetic data. It checks that held-out rollouts occur only after saving the candidate and that report/CSV/PNG export succeeds. It does not establish physical parameter identifiability or real-world accuracy.
- The actual Qt panel and multiprocessing worker passed an offscreen GUI smoke test: read-only connection, enable hold, freeze/edit/capture, disable, motor-enable lock during offline fitting, and shutdown. The control-panel screenshot was inspected for legibility and clipping.
- The passive-viewer call signature was checked against the installed MuJoCo version. The new two-window interface still needs an attended desktop/physical smoke test on the user's Jetson; its 3D OpenGL path was not exercised on that hardware.

Reproduce the checks using the commands in README.md. Without `SO101_TEST_XML`, the three model-dependent tests skip; passing the remaining tests does not replace those checks. Demo data is explicitly marked synthetic and must never be used as calibration evidence for physical hardware.
