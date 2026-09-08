# Calibration interface

For this combined package, follow [the calibration workflow](../docs/CALIBRATION.md) and then [the PPO instructions](../README.md).

Hardware behavior and optimizer behavior are preserved. The offline fitter now records training-content hashes and checks portable legacy identities to reject reused recordings during independent evaluation. `export_for_ppo.py` is the new offline calibration-to-PPO bridge. `make_demo_bundle.py` generates synthetic software-test data only.

The original standalone README is preserved as `REFERENCE_README.md`; the separately supplied instructions are preserved as `ORIGINAL_WORKFLOW.md`. Their older folder paths refer to the original standalone setup. Use the combined-package guide for current commands.
