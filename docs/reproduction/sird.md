# Constrained SIRD experiments

Recorded observations are in `data/synthetic/sird/2026-10-11/` and results in
`results/sird/2026-10-11/`. All 12 final neural checkpoints are included: clean,
noisy, missing and chronological-holdout conditions, each with seeds 0, 1 and 2.
Existing figures are in `figures/sird/2026-10-11/`.

Inspect the saved results without retraining:

```bash
python scripts/verify_recorded_results.py
```

To create a separate experiment:

```bash
python tests/check_sird_constraints.py
python scripts/validate_sird.py prepare --output runs/sird
python scripts/validate_sird.py baselines --output runs/sird
```

Run the following command for each CASE (`clean`, `noisy`, `missing`, `forecast`)
and SEED (`0`, `1`, `2`):

```bash
python scripts/validate_sird.py run --case CASE --seed SEED --output runs/sird
```

After completing every fit:

```bash
python scripts/validate_sird.py aggregate --output runs/sird
python scripts/summarize_sird.py --output runs/sird
```

New experiments receive new creation timestamps and current source hashes. The saved
October protocols remain unchanged. Data preparation uses the same fixed synthetic design;
the forecast fit uses observations and collocation only through day 60.
Its future curve is obtained by integrating the ODE with the inferred rates.

Nonnegative states, initial conditions and population conservation are imposed by the
state transform. Equation residuals are penalized and cumulative-state monotonicity
is not imposed exactly. Read the recorded diagnostics and comparator results.
