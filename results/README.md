# Recorded experiment results

Each dated folder contains the original protocols, metrics, calibration summaries,
trajectories, training histories and saved neural checkpoints. All final seeds are retained.
The original benchmark has two separately recorded reproduction runs; SIRD has 12 final
neural fits and the two-region extension has nine.

These files were moved without changing their bytes. Historical source hashes are verified
against `archive/2026-10-11/source/`; old-to-new locations are recorded in
`docs/provenance/path_map.json`. Observations and figures are now in separate folders.

Use `python scripts/verify_recorded_results.py` for read-only verification. Reproduction
runners write new outputs to `runs/`, which is excluded from version control.
