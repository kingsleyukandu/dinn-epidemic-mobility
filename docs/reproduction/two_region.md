# Two-region SIRD mobility experiments

Recorded observations are in `data/synthetic/two_region/2026-10-11/`; all nine final
fits and three conventional ODE calibrations are in `results/two_region/2026-10-11/`.
The original figures are in `figures/two_region/2026-10-11/`.

The model tracks S, I, R and D in each region, as fractions of the combined initial
population. Incidence uses fixed initial reference fractions p1=0.6 and p2=0.4.
Directional movement applies equally to S, I and R. Deceased states do not move.

Inspect the saved results without training:

```bash
python scripts/verify_recorded_results.py
```

The walkthrough at `notebooks/two-region-sird-dinn.ipynb` loads the included clean
seed-0 checkpoint by default and recomputes its diagnostics. Its saved cells are
executed sequentially in an ordinary Python process, not a Jupyter kernel.

For an independent experiment:

```bash
python tests/check_two_region_model.py
python scripts/validate_two_region.py prepare --output runs/two_region
python scripts/validate_two_region.py baselines --output runs/two_region
```

Run the following for every CASE (`clean`, `noisy`, `missing`) and SEED (`0`, `1`, `2`):

```bash
python scripts/validate_two_region.py run --case CASE --seed SEED --output runs/two_region
```

After completing all nine fits:

```bash
python scripts/validate_two_region.py aggregate --output runs/two_region
python scripts/summarize_two_region.py --output runs/two_region
```

The historical protocols and their hashes refer to the archived source. A new run
records the current runner/check filenames and current creation time. Model equations,
training budgets, synthetic observations, noise/missingness seeds and fitting settings
have not changed through repository organization.

All eight states and the initial state are known. Each noisy or missing condition has
one fixed dataset and three neural initializations. Small decreases in deceased neural
trajectories are recorded approximation errors. Equal symmetric regions cannot identify
the common movement scale. These experiments do not establish real-data forecasting
performance or superiority over conventional calibration.
