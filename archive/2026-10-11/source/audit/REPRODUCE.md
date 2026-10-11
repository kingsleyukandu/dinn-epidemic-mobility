# SIRD reproduction audit

This audit checks the notebook at commit `3c6275393c7f2aea437683c0e077996e6bff5d45`.
The original `notebooks/sird-dinn.ipynb` is retained without changes.

From the repository root, install the recorded dependencies and execute:

```bash
python -m pip install -r audit/requirements-audit.txt
python scripts/audit_step1.py
```

The runner executes the ordinary Python code cells in a new Python process in
notebook order. It clears old outputs, fixes CPU threading to one thread, records
detached losses and parameters every 100 iterations, and appends diagnostics.
It writes an executed notebook to `audit/sird-dinn-clean-run.ipynb` and numerical
results to `audit/results/`. No equations, seeds, initialization guesses, loss
weights, architecture, optimizer, or iteration counts are changed.

The experiment trains on 54 synthetic observations of all four compartments,
spaced every three days from day 0 to day 159. The residual collocation grid spans
day 0 to day 160. Diagnostics use 641 evenly spaced points over the same interval.
There is no held-out forecasting test in this audit.

Loss histories use normalized time `tau=t/160`. Equation residual diagnostics
are separately expressed per physical day by dividing the time derivative by
160. Relative errors are saved as both fractions and percentages. The logged
losses are measured immediately before each logged optimizer update; final
metrics are recomputed after the last update.

The audit environment uses Python 3.12.14. The original notebook metadata reports
Python 3.11.15 but contains no complete package-version manifest. This audit is a
reproduction in a recorded environment, not a reconstruction of that historical
environment. Results can change with software, floating-point arithmetic and
hardware; a repeated fresh run checks determinism in the recorded environment.

Positivity and conservation diagnostics are numerical grid checks. Small
residuals do not prove exact differential-equation satisfaction. Positive rate
parameters are enforced by exponentiation; compartment values are unconstrained
linear network outputs, and no explicit population-conservation term is used.

This is Step 1 only. Multiple random seeds, noisy or missing observations,
classical calibration comparisons and mobility extensions are later stages.
