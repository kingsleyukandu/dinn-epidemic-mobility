# Step 3: two-region SIRD mobility

The implementation is `src/two_region_sird.py`. The original single-region notebook
and Step 2 source have not changed. `notebooks/two-region-sird-dinn.ipynb` displays
the recorded validation and can independently retrain the illustrative clean seed.
Its saved outputs were generated sequentially in a fresh Python process, without
a Jupyter kernel; the default notebook mode loads a saved checkpoint and recomputes
diagnostics, rather than repeating training.

## Model definition

State ordering: S1, I1, R1, D1, S2, I2, R2, D2. States are fractions of the combined
initial population, including cumulative deaths. Initial regional reference
populations are p1 = 0.6 and p2 = 0.4. Incidence in region j is
beta_j * S_j * I_j / p_j. These denominators are fixed initial reference populations,
not current regional populations. With movement set to zero, dividing each regional
state by its own p_j reduces to the original single-region SIRD equations.

tau12 moves S, I, and R from region 1 to 2; tau21 moves them from region 2 to 1.
Deceased states do not move. One movement coefficient per direction is shared by
the three living compartments. The generator uses columns for origins:

| Destination | Origin 1 | Origin 2 |
| --- | --- | --- |
| Region 1 | -tau12 | tau21 |
| Region 2 | tau12 | -tau21 |

Global population is conserved. Local population changes by net movement of
living states. The neural output uses one simplex across all eight compartments,
blended with the exact initial state. This enforces nonnegative states, the global
total, and exact initial conditions. ODE and local-balance identities remain soft
constraints. Small decreases in deceased states are recorded approximation errors;
regional recovered counts may legitimately decline through outward movement.

## Reproduction

Install `requirements-step3.txt` in an isolated Python 3.12 environment. Exact
runtime details are in `results/protocol.json`. Run from the project root:

```bash
python scripts/check_step3_model.py
python scripts/validate_step3.py prepare --output validation/step3/reproduction
python scripts/validate_step3.py baselines --output validation/step3/reproduction
```

For each CASE (`clean`, `noisy`, `missing`) and SEED (`0`, `1`, `2`), run:

```bash
python scripts/validate_step3.py run --case CASE --seed SEED --output validation/step3/reproduction
```

Then run:

```bash
python scripts/validate_step3.py aggregate --output validation/step3/reproduction
python scripts/summarize_step3.py --output validation/step3/reproduction
```

Use a fresh output path to preserve recorded evidence. Preparation and fitting
commands reject overwriting completed records. Model-check output is written to
`validation/step3/model_checks/checks.json`. All final neural seeds are retained;
the separately identified development pilot is excluded from validation totals.
The fresh clean seed-0 run exactly reproduced the pilot's compared metrics in this
recorded environment.

## Protocol and verification

All eight states and the initial state are observed/known. Clean data have 54 times
from 0 to 159 in steps of 3 days. Independent Gaussian noise has sigma 0.001 of the
combined initial population, RNG seed 20261013; negative measurements are retained,
without clipping or renormalization. Missingness removes 16 interior timepoints
with RNG seed 20261014, retaining both endpoints. There is one fixed dataset per
condition and three network seeds, not three independent noise realizations.

All runs use three tanh layers of width 48, float64, 8,000 Adam updates at 0.001,
then up to 1,000 L-BFGS iterations. Loss is data MSE plus 0.0001 times equation MSE
in normalized time t/160. There are 161 collocation points and 641 diagnostic times.
Fixed budgets and final checkpoints are used, without stopping or selecting based
on known true-rate error. The L-BFGS iteration count is recorded; a completed budget
does not certify convergence. Logged losses and rates refer to the same evaluated
model. CPU runtimes were collected during concurrent fits and are not a speed
benchmark.

ODE calibration uses least squares on log rates with the same observations,
known initial state, and starting rate guesses. Bounds are 0.0001 to 3 per day for
disease rates and 0.00001 to 0.5 for movement rates. Data generation and calibration
both use DOP853 (rtol 1e-10, atol 1e-12), a model-correct setting favorable to ODE
calibration. Model checking also compares Radau and uses matrix-exponential and
independent single-region limits.

`parameter_recovery.csv` contains all 96 disease/movement parameter estimates from
nine neural runs and three ODE fits. `diagnostics.csv`, `neural_summary.json`,
each run's metrics, training history, and dense trajectories preserve the results.
`verification.json` independently recomputes checkpoint metrics and observed-state
objectives, and verifies source/data hashes. Numerical trajectory checks use
1e-8 absolute tolerance on the combined-population-fraction scale. Algebraic tests
use roundoff tolerances.

Equal regions with identical epidemics and symmetric flows do not identify the
common movement scale: the limiting-case check changes both rates sixfold without
changing trajectories. Nonzero numerical sensitivity singular values in the
asymmetric experiment are a local check, not proof of global identifiability.

No real surveillance or mobility data, uncertainty intervals, practical forecast
validation, or method superiority is claimed. The multipatch SIS stage remains
planned. GitHub publication, Exhibit 12.3, and petition reconciliation belong to
the subsequent authorized stages.
