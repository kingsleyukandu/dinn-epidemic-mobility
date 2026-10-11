# Synthetic observations

The dated CSV files are the exact observations used by the recorded experiments.
They were relocated without changing their contents. The historical protocol hashes
remain valid and are checked by `scripts/verify_recorded_results.py`.

SIRD observations use the single-region population as denominator. Two-region observations
use the combined initial population as denominator. Noise is retained without clipping or
renormalization. Read each experiment's protocol for the known initial state, noise scale,
removed observation times and fixed random seeds.
