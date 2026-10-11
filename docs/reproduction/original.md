# Original SIRD benchmark

The original public notebook remains at `notebooks/sird-dinn.ipynb`.
The historical commit is `3c6275393c7f2aea437683c0e077996e6bff5d45`.
Its exact Git blob is `2585e90aba6c964101b55e642a62279ea3318ece`.

The October 11 reproduction runs are in `results/original/2026-10-11/run_1/`
and `run_2/`. Their figures are in the corresponding `figures/original/` folders.
The executed reproduction notebooks are retained in `archive/2026-10-11/notebooks/`.

To reproduce independently:

```bash
python scripts/reproduce_original.py --output runs/original/reproduction.ipynb --results-dir runs/original/results
```

The source notebook is read without modification. Instrumentation records detached
training history and appends diagnostics. Ordinary Python cells are executed sequentially
in a fresh process, and the outputs are saved as a notebook. A Jupyter kernel is not used
by this runner. Choose unused output paths; existing completed outputs are rejected.

The preserved original implementation can produce negative neural state values and small
population-balance errors. These diagnostics remain in the results; the later constrained
model is a separate implementation.
