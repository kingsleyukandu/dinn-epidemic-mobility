# Verification guide

Disease-Informed Neural Networks for Epidemic Dynamics and Population Mobility

The original notebook is pinned to public commit `3c6275393c7f2aea437683c0e077996e6bff5d45`, dated June 19, 2026. Later experiments are dated October 11, 2026. The revised repository has not yet been published; commit-specific links for later files must be assigned after publication. A path in this guide refers to the current local revision until then.

## Read and check

1. Read `docs/report/DINN_Project_Report.pdf` (six pages) or its Markdown source. The report explains the model, benchmark design, numerical results, comparators, and limitations.
2. Read `README.md` for installation and navigation. `readme_extracts.json` preserves the selected ranges for a compact printed snapshot.
3. Compare `code_extracts.json` with the exact source ranges. Cell indices are zero-based; line numbers within a notebook cell are one-based. Module line numbers are one-based. Excerpts omit surrounding code; complete implementations remain available.
4. Compare `recorded_output_extracts.json` with its named metrics files. Values and JSON keys are unchanged; ordering and whitespace are presentation choices. All omitted keys are listed. The two-region seed-0 excerpt is illustrative; conclusions use all nine final seeds in the result tables, not that seed alone.
5. Use `source_index.csv` or `source_index.json` to check complete SHA-256 values and byte counts for report inputs, selected sources, result tables, figures, observations, and checkpoints. SHA-256 establishes file identity, not authorship or an independently certified execution date.

From the repository root, install the pinned dependencies in a Python 3.12 environment and run:

```bash
python scripts/verify_recorded_results.py
python tests/check_sird_constraints.py
python tests/check_two_region_model.py --output runs/verification_model_checks
```

Choose a fresh `runs/` output directory. The recorded-result verifier checks saved models and objectives without retraining. Its successful status is `PASS`. Earlier verification receipts document checks already performed; a new receipt should retain its actual execution time.

## Result and figure records

`results/presentation/2026-10-11/T02_parameter_recovery_all_fits.csv` retains every final rate estimate. T03 summarizes all cases, T04 retains state and residual diagnostics, and T06 records the chronological forecast comparisons. `results/presentation/2026-10-11/asset_manifest.json` connects figures and tables to their inputs. F03 and F06 already appear in the report; F02 and F04 provide supplementary rate-recovery views.

Original negative states and population drift, later nonzero equation residuals, small negative derivatives, identifiability limitations, and the more accurate ODE comparator are part of the record. These are controlled synthetic benchmarks with known initial conditions and fully observed states. SIS adaptation and empirical surveillance/mobility evaluation remain future work.

## Rebuild these extracts

```bash
python scripts/build_verification_sources.py --output runs/verification_sources
```

This command derives a fresh inventory and excerpts from existing files, refuses an existing output directory, and checks frozen report/presentation input hashes. It does not train models, alter observations, or publish the repository. Dates in the generated index describe index creation.
