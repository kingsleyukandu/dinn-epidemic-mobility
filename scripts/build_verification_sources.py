"""Extract exact source and result selections without fitting or changing records."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = '3c6275393c7f2aea437683c0e077996e6bff5d45'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Fresh output directory; existing directories are refused.')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing existing output directory: {out}')
    # Resolve all input selections before creating the output directory.
    codes = []
    notebook = json.loads((ROOT/'notebooks/sird-dinn.ipynb').read_text())
    selections = [
        ('original_loss', 'notebooks/sird-dinn.ipynb', 15, 1, 8),
        ('single_constraints', 'src/sird_validation.py', None, 53, 59),
        ('single_forward', 'src/sird_validation.py', None, 71, 86),
        ('two_region_equations', 'src/two_region_sird.py', None, 20, 43),
        ('two_region_forward', 'src/two_region_sird.py', None, 115, 129),
    ]
    for item_id, path, cell, start, end in selections:
        lines = (''.join(notebook['cells'][cell]['source']).splitlines() if cell is not None
                 else (ROOT/path).read_text().splitlines())
        codes.append(dict(id=item_id, source=path, source_sha256=digest(ROOT/path),
                          notebook_cell_index_zero_based=cell, line_start=start, line_end=end,
                          lines=lines[start-1:end]))
    results = []
    result_selections = [
        ('original_run_1', 'results/original/2026-10-11/run_1/metrics.json',
         ['utc_time','true_parameters','recovered_parameters','relative_errors_percent',
          'negative_entries','population_conservation_max_abs_error',
          'equation_residual_max_abs_per_day','initial_condition_max_abs_error']),
        ('two_region_clean_seed_0', 'results/two_region/2026-10-11/clean/seed_0/metrics.json',
         ['parameters','relative_errors_percent','trajectory_rmse','residual_max_abs_per_day',
          'conservation_max_abs_error','initial_max_abs_error','negative_entries',
          'minimum_D_derivative_per_day','regional_balance_residual_max_abs_per_day']),
        ('two_region_model_checks', 'results/two_region/2026-10-11/model_checks.json', ['status']),
    ]
    for item_id, path, keys in result_selections:
        value = json.loads((ROOT/path).read_text())
        results.append(dict(id=item_id, source=path, source_sha256=digest(ROOT/path),
                            selected_keys=keys, fields={k:value[k] for k in keys},
                            omitted_keys=[k for k in value if k not in keys]))
    checks = json.loads((ROOT/'results/two_region/2026-10-11/model_checks.json').read_text())
    results[-1]['selected_result_groups'] = ['population_balance','independent_solver','symmetric_limit']
    results[-1]['results'] = {k:checks['results'][k] for k in results[-1]['selected_result_groups']}
    readme_lines = (ROOT/'README.md').read_text().splitlines()
    readme = dict(source='README.md', source_sha256=digest(ROOT/'README.md'),
                  blocks=[dict(line_start=a,line_end=b,lines=readme_lines[a-1:b])
                          for a,b in [(1,48),(82,127)]],
                  omitted_lines='49-81: result tables and trajectory figure already represented in the report.')
    paths = {x['source'] for x in codes+results} | {'README.md','requirements.txt',
              'docs/report/project_report.md','docs/report/DINN_Project_Report.pdf',
              'docs/report/report_source_manifest.json','docs/provenance/readme_verification.json',
              'scripts/verify_recorded_results.py','tests/check_sird_constraints.py',
              'tests/check_two_region_model.py','scripts/build_presentation_assets.py',
              'scripts/build_verification_sources.py'}
    expected = {}
    manifest = json.loads((ROOT/'results/presentation/2026-10-11/asset_manifest.json').read_text())
    for item in manifest['input_files']+manifest['generated_files']:
        paths.add(item['path']); expected[item['path']] = item['sha256']
    report_manifest = json.loads((ROOT/'docs/report/report_source_manifest.json').read_text())
    for item in report_manifest['inputs']:
        paths.add(item['path']); expected[item['path']] = item['sha256']
    expected['docs/report/project_report.md'] = report_manifest['markdown_source_sha256']
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'results').glob('**/weights.pt'))
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'data/synthetic').glob('**/*.csv'))
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'results').glob('**/protocol.json'))
    source_index = []
    for path in sorted(paths):
        sha = digest(ROOT/path)
        if path in expected and sha != expected[path]:
            raise ValueError(f'Historical input hash mismatch: {path}')
        historical = path == 'notebooks/sird-dinn.ipynb'
        source_index.append(dict(path=path, sha256=sha, bytes=(ROOT/path).stat().st_size,
            public_commit=BASE if historical else None,
            publication_status='published original' if historical else 'local revision; release commit pending',
            permalink=f'https://github.com/kingsleyukandu/dinn-epidemic-mobility/blob/{BASE}/{path}' if historical else None))
    out.mkdir(parents=True)
    common = dict(created_utc=datetime.now(timezone.utc).isoformat(), public_original_commit=BASE,
                  later_release_commit=None, later_publication_status='pending',
                  operation='Exact excerpts and file inventory from existing records; no training.')
    payloads = {'code_extracts.json':codes,'recorded_output_extracts.json':results,
                'readme_extracts.json':readme,'source_index.json':source_index}
    for name, value in payloads.items():
        (out/name).write_text(json.dumps(dict(**common,items=value),indent=2)+'\n')
    with (out/'source_index.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(source_index[0]));writer.writeheader();writer.writerows(source_index)
    chunks = ['# Selected source code','',
              'Exact source lines; omitted portions are identified by each range. Notebook cell indices are zero-based.','']
    for c in codes:
        location = f"cell {c['notebook_cell_index_zero_based']}, " if c['notebook_cell_index_zero_based'] is not None else ''
        chunks += [f"## {c['source']} ({location}lines {c['line_start']}-{c['line_end']})",'',
                   f"SHA-256: `{c['source_sha256']}`",'', '```python',*c['lines'],'```','']
    (out/'code_extracts.md').write_text('\n'.join(chunks))
    (out/'verification_index.md').write_text('''# Verification guide

Disease-Informed Neural Networks for Epidemic Dynamics and Population Mobility

The original notebook is pinned to public commit `'''+BASE+'''`, dated June 19, 2026. Later experiments are dated October 11, 2026. The revised repository has not yet been published; commit-specific links for later files must be assigned after publication. A path in this guide refers to the current local revision until then.

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
''')
    print(json.dumps({'status':'PASS','source_files':len(source_index),'code_extracts':len(codes),
                      'output_extracts':len(results),'frozen_input_hashes_checked':len(expected),'output':str(out)}))

if __name__ == '__main__':
    main()
