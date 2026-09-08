# Local calibration bundles

`calibrations/so101-12v-20260907-174850-482396032/` is the selected real bundle, exported from `sessions/session-20260907-174056-522799423/fit-20260907-174850-482396032/params.json` with all three held-out replays. See [root commands](../README.md).

Real bundles are local and ignored by Git. Export each finished fit to a uniquely named directory; existing destinations are never overwritten. Copy and back up the whole directory, including `bundle.json`, configuration, frozen model/meshes, motor files, fitted parameters, source diagnostics and replays. Internal paths are portable. Raw telemetry is unnecessary for export or PPO and stays in the separate session archive.

Treat a bundle as immutable. A changed calibration requires new training; checkpoint resume requires the original bundle. Keep `policy.onnx` with `policy_manifest.json` and the run's frozen `calibration/` directory. Back up full run directories to retain checkpoints and training configuration. Preserve raw sessions separately for future fitting and audits.
