# Presentation figures and tables

Experiment records: October 11, 2026. Assets are derived without retraining. The original notebook was public at commit `3c6275393c7f2aea437683c0e077996e6bff5d45`; the later validations and these presentation assets are local until publication.

## F01_original_sird: Original SIRD reproduction

The original public notebook was reproduced on October 11, 2026. All three rate errors were below 0.1%, but the linear output permits negative states and imperfect population accounting. The fitted trajectories and synthetic truth nearly overlap at this scale; T01 and T05 retain numerical errors. These are reproduction results, not June execution dates.

Sources:

- `results/original/2026-10-11/run_1/metrics.json`
- `results/original/2026-10-11/run_1/trajectory_audit.csv`

## F02_sird_rate_recovery: Constrained SIRD rate recovery

Each colored point is a network initialization seed; black crosses are direct ODE calibration. All seeds within a case use the same observations. Noisy means one fixed Gaussian noise realization (sigma 0.002); missing means 16 of 54 observation times removed, retaining both endpoints. Absolute relative error is 100 x |estimate - truth| / truth. These points are not confidence intervals.

Sources:

- `results/sird/2026-10-11/clean/ode_baseline.json`
- `results/sird/2026-10-11/clean/seed_0/metrics.json`
- `results/sird/2026-10-11/clean/seed_1/metrics.json`
- `results/sird/2026-10-11/clean/seed_2/metrics.json`
- `results/sird/2026-10-11/noisy/ode_baseline.json`
- `results/sird/2026-10-11/noisy/seed_0/metrics.json`
- `results/sird/2026-10-11/noisy/seed_1/metrics.json`
- `results/sird/2026-10-11/noisy/seed_2/metrics.json`
- `results/sird/2026-10-11/missing/ode_baseline.json`
- `results/sird/2026-10-11/missing/seed_0/metrics.json`
- `results/sird/2026-10-11/missing/seed_1/metrics.json`
- `results/sird/2026-10-11/missing/seed_2/metrics.json`

## F03_two_region_trajectories: Two-region synthetic trajectories

Clean-data fits show all three seeds. Solid curves are fitted neural states; dashed curves are synthetic ODE truth. Lower panels show the largest absolute state error in each region at each diagnostic time. All state fractions use the initial global population, not changing regional totals.

Sources:

- `results/two_region/2026-10-11/clean/seed_0/trajectory.csv`
- `results/two_region/2026-10-11/clean/seed_1/trajectory.csv`
- `results/two_region/2026-10-11/clean/seed_2/trajectory.csv`

## F04_two_region_rate_recovery: Disease and directional mobility rate recovery

Six disease rates and two directional movement rates were estimated from all eight observed states. tau12 means region 1 to region 2; tau21 means region 2 to region 1. The same rates move S, I, and R; D does not move. Noisy uses one fixed realization (sigma 0.001 in global population fractions). Missing retains 38 of 54 times. Points reflect three initializations, not repeated datasets. ODE calibration has lower overall trajectory RMSE in all three cases.

Sources:

- `results/two_region/2026-10-11/clean/ode_baseline.json`
- `results/two_region/2026-10-11/clean/seed_0/metrics.json`
- `results/two_region/2026-10-11/clean/seed_1/metrics.json`
- `results/two_region/2026-10-11/clean/seed_2/metrics.json`
- `results/two_region/2026-10-11/noisy/ode_baseline.json`
- `results/two_region/2026-10-11/noisy/seed_0/metrics.json`
- `results/two_region/2026-10-11/noisy/seed_1/metrics.json`
- `results/two_region/2026-10-11/noisy/seed_2/metrics.json`
- `results/two_region/2026-10-11/missing/ode_baseline.json`
- `results/two_region/2026-10-11/missing/seed_0/metrics.json`
- `results/two_region/2026-10-11/missing/seed_1/metrics.json`
- `results/two_region/2026-10-11/missing/seed_2/metrics.json`

## F05_population_accounting: Population accounting and remaining residuals

The left panel uses clean seed 0 and includes cumulative deceased states in each regional total. Movement changes regional totals while the global total remains one. Dashed regional curves are ODE truth. The right panel includes all clean seeds and shows nonzero differential-equation residuals. State nonnegativity, initial conditions, and global accounting are structural constraints; the ODE is enforced through a loss. Death monotonicity is not a structural guarantee.

Sources:

- `results/two_region/2026-10-11/clean/seed_0/trajectory.csv`
- `results/two_region/2026-10-11/clean/seed_1/trajectory.csv`
- `results/two_region/2026-10-11/clean/seed_2/trajectory.csv`

## F06_chronological_forecast: Chronological synthetic forecast test

The models used only 21 clean observations through day 60 and collocation points within that window. Estimated rates were then integrated in the SIRD ODE for days 61-88; this is not direct neural extrapolation. The right panel uses a log scale and includes all three initializations, direct ODE calibration, and day-60 persistence. One clean, correctly specified synthetic cutoff does not establish real-world forecasting accuracy. ODE calibration has the lowest RMSE.

Sources:

- `results/sird/2026-10-11/forecast_summary.json`
- `data/synthetic/sird/2026-10-11/forecast_observations.csv`
- `results/sird/2026-10-11/forecast/seed_0/forecast.csv`
- `results/sird/2026-10-11/forecast/seed_1/forecast.csv`
- `results/sird/2026-10-11/forecast/seed_2/forecast.csv`

## Table definitions

T01: original rates and absolute relative errors.

T02: every rate estimate from the 21 later DINN fits and seven ODE calibrations.

T03: worst absolute relative rate errors and trajectory RMSE range across three seeds; the ODE comparator has one fit per case.

T04: all 21 neural fits, with full precision state and residual diagnostics.

T05: original reproduction diagnostics.

T06: all-state RMSE and infected-state MAE on days 61-88.

Full-window trajectory diagnostics use 641 equally spaced points on days 0-160. RMSE is the square root of the mean squared error over all stated times and states. Original CSV states are printed at float32 precision; original aggregate diagnostics are taken from the stored metrics. Scientific conclusions should use the metrics and exact CSV tables rather than rounded labels.

SIRD and two-region observations are synthetic and fully observed. Noise and missing patterns each have one fixed realization. Seed variation is optimizer initialization variation, not statistical uncertainty. Disease rates and movement rates have units per day. Fixed initial regional fractions 0.6 and 0.4 are the transmission denominators. Real surveillance data, uncertain initial states, global identifiability, and deployment are outside these experiments.
