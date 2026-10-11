# Step 2: constrained SIRD validation

The original `notebooks/sird-dinn.ipynb` is preserved. The new implementation is
`src/sird_validation.py`; the fixed experiment runner is `scripts/validate_step2.py`.
Step 2 adds hard state constraints, not exact satisfaction of the ODE.

## Environment

Python 3.12.14; CPU; one torch thread per process. Exact package versions and
source/data hashes are recorded in `step2/protocol.json`. Install
`requirements-validation.txt` in an isolated environment.

## Reproduce without overwriting the recorded evidence

Use a fresh working copy containing `src/` and `scripts/`, with no
`validation/step2/` folder. Run commands from that copy's project root:

```bash
python scripts/check_step2_constraints.py
python scripts/validate_step2.py prepare
python scripts/validate_step2.py baselines
```

Run `python scripts/validate_step2.py run --case CASE --seed SEED` for every
combination of CASE (`clean`, `noisy`, `missing`, `forecast`) and SEED (`0`, `1`, `2`).
Then run:

```bash
python scripts/validate_step2.py aggregate
python scripts/summarize_step2.py
```

The preparation and fit commands reject overwriting completed records. Training
uses a fixed budget: 8,000 Adam updates, then at most 600 L-BFGS iterations. It does
not use true parameter error to stop or select a checkpoint. Three seeds measure
neural initialization sensitivity; they are not three independent noise datasets.
The pilot was used to check implementation and was excluded from the 12-run
validation summary. No seed was selected as the representative result.

## What is measured

Clean and noisy datasets have 54 timepoints (days 0, 3, ..., 159), observing all four
states. Noise is independent additive Gaussian with sigma 0.002 of total population;
negative measurements are retained. The missing dataset removes 16 interior
timepoints, preserving both endpoints. The initial state is known in every case.

ODE calibration uses least squares on log rates, the same observations and starting
rates as DINN, and bounds 0.0001 to 3 per day for each rate. Both reference generation
and calibration integrate with DOP853 (rtol 1e-10, atol 1e-12). This model-correct,
noiseless setting favors conventional ODE calibration.

The forecast fits only days 0 to 60 and collocation through day 60. It integrates the
ODE using inferred rates to forecast days 61 to 88. It does not evaluate neural
extrapolation. Persistence holds the day-60 states fixed. No future observations or
future collocation enter training.

Population fractions, not counts, are used throughout. The model enforces
nonnegative states, sum one, and the exact initial state for nonnegative time. ODE
residuals remain nonzero; monotonic recovered/deceased states are not hard
constraints. Only S and I observations cannot identify gamma and omega separately.

`neural_summary.csv` and each case's `ode_baseline.json` record all final estimates.
`verification.json` records checkpoint checks and monotonicity diagnostics.
Training histories are objective diagnostics; Adam history losses are evaluated
before the update while logged rates are recorded after the update. Final recorded
losses and estimates use the same final model. CPU runtimes were measured during
concurrent fits and are not a controlled speed benchmark.

These are controlled synthetic experiments. No real-data validation, uncertainty
coverage, regional mobility implementation, or forecasting superiority is claimed.
Repository publication and petition reconciliation remain subsequent stages.
