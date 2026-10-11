"""Build presentation figures/tables from frozen records; no fitting or training.

Default targets are new presentation directories, separate from dated experiments.
Use --output runs/presentation-check for an independent regeneration. Existing
targets are rejected. All input files and generated assets have SHA-256 records.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATE = '2026-10-11'
COLORS = ['#0072B2', '#D55E00', '#009E73']
CASES = ['clean', 'noisy', 'missing']
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.hashsalt': 'dinn-presentation-2026-10-11'})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(target):
    figures = target / 'figures/presentation' / DATE
    tables = target / 'results/presentation' / DATE
    for path in [figures, tables]:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}; use a fresh --output directory.')
    inputs = {}
    assets = []
    entries = []

    def read(path):
        full = ROOT / path
        inputs[path] = sha(full)
        return full

    def js(path):
        return json.loads(read(path).read_text())

    def array(path):
        return np.genfromtxt(read(path), delimiter=',', names=True)

    def table(name, rows):
        full = tables / (name + '.csv')
        with full.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        assets.append(full)

    def figure(name, fig, title, caption, sources):
        for suffix in ['png', 'svg']:
            full = figures / (name + '.' + suffix)
            kwargs = {'metadata': {'Date': None}} if suffix == 'svg' else {}
            fig.savefig(full, dpi=240, facecolor='white', bbox_inches='tight', pad_inches=.12, **kwargs)
            assets.append(full)
        plt.close(fig)
        entries.append({'id': name, 'title': title, 'caption': caption, 'sources': sources})

    def style(ax, xlabel='Day', ylabel='Population fraction'):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.18)

    figures.mkdir(parents=True)
    tables.mkdir(parents=True)
    original_path = f'results/original/{DATE}/run_1/metrics.json'
    original = js(original_path)
    original_trajectory = f'results/original/{DATE}/run_1/trajectory_audit.csv'
    a = array(original_trajectory)
    table('T01_original_parameters', [dict(parameter=k, true_rate_per_day=v,
          estimated_rate_per_day=original['recovered_parameters'][k],
          absolute_relative_error_percent=original['relative_errors_percent'][k])
          for k, v in original['true_parameters'].items()])
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.6), layout='constrained')
    for ax, state, title in zip(axes.flat, ['s', 'i', 'r', 'd'],
                                ['Susceptible (S)', 'Infected (I)', 'Recovered (R)', 'Deceased (D)']):
        ax.plot(a['day'], a[state + '_dinn'], color=COLORS[0], label='Original DINN: repeat run 1')
        ax.plot(a['day'], a[state + '_true'], color='black', linestyle='--', label='Synthetic ODE truth')
        ax.set_title(title)
        style(ax)
    axes[0, 0].legend(fontsize=8, frameon=False)
    figure('F01_original_sird', fig, 'Original SIRD reproduction',
           'The original public notebook was reproduced on October 11, 2026. All three rate errors '
           'were below 0.1%, but the linear output permits negative states and imperfect population '
           'accounting. The fitted trajectories and synthetic truth nearly overlap at this scale; '
           'T01 and T05 retain numerical errors. These are reproduction results, not June execution dates.',
           [original_path, original_trajectory])

    metrics = {}
    baselines = {}
    recovery = []
    benchmarks = []
    diagnostics = []
    for model in ['sird', 'two_region']:
        protocol = js(f'results/{model}/{DATE}/protocol.json')
        truth = protocol['truth' if model == 'sird' else 'synthetic_truth']
        cases = CASES + (['forecast'] if model == 'sird' else [])
        for case in cases:
            baseline_path = f'results/{model}/{DATE}/{case}/ode_baseline.json'
            baseline = js(baseline_path)
            baselines[model, case] = baseline
            for k, value in baseline['parameters'].items():
                recovery.append(dict(model=model, case=case, method='ODE', seed='not_applicable',
                                     parameter=k, true_rate_per_day=truth[k], estimated_rate_per_day=value,
                                     absolute_relative_error_percent=baseline['relative_errors_percent'][k]))
            for seed in range(3):
                path = f'results/{model}/{DATE}/{case}/seed_{seed}/metrics.json'
                m = js(path)
                metrics[model, case, seed] = m
                for k, value in m['parameters'].items():
                    recovery.append(dict(model=model, case=case, method='DINN', seed=seed,
                                         parameter=k, true_rate_per_day=truth[k], estimated_rate_per_day=value,
                                         absolute_relative_error_percent=m['relative_errors_percent'][k]))
                diagnostics.append(dict(model=model, case=case, seed=seed,
                                        **{k: m[k] for k in ['trajectory_rmse', 'residual_max_abs_per_day',
                                           'conservation_max_abs_error', 'initial_max_abs_error',
                                           'minimum_compartment', 'negative_entries']}))
            if case != 'forecast':
                group = [metrics[model, case, s] for s in range(3)]
                disease = [k for k in group[0]['parameters'] if not k.startswith('tau')]
                movement = [k for k in group[0]['parameters'] if k.startswith('tau')]
                errors = lambda keys: max(m['relative_errors_percent'][k] for m in group for k in keys)
                benchmarks.append(dict(model=model, case=case, dinn_fits=3,
                    dinn_max_disease_error_percent=errors(disease),
                    dinn_max_mobility_error_percent=errors(movement) if movement else '',
                    dinn_trajectory_rmse_min=min(m['trajectory_rmse'] for m in group),
                    dinn_trajectory_rmse_max=max(m['trajectory_rmse'] for m in group),
                    ode_max_disease_error_percent=max(baseline['relative_errors_percent'][k] for k in disease),
                    ode_max_mobility_error_percent=max(baseline['relative_errors_percent'][k] for k in movement) if movement else '',
                    ode_trajectory_rmse=baseline['trajectory_rmse']))
    table('T02_parameter_recovery_all_fits', recovery)
    table('T03_benchmark_summary', benchmarks)
    table('T04_constraint_diagnostics_all_fits', diagnostics)
    table('T05_original_diagnostics', [dict(metric=k, value=original[k]) for k in
          ['trajectory_rmse_overall', 'equation_residual_max_abs_per_day',
           'population_conservation_max_abs_error', 'initial_condition_max_abs_error', 'negative_entries']])

    def rate_plot(model, keys, name, title, caption):
        fig, axes = plt.subplots(1 if model == 'sird' else 2, 3 if model == 'sird' else 4,
                                 figsize=(8.2, 3.2 if model == 'sird' else 4.8), layout='constrained')
        for ax, key in zip(axes.flat, keys):
            for i, case in enumerate(CASES):
                for seed in range(3):
                    ax.scatter(i + .12 * (seed-1), metrics[model, case, seed]['relative_errors_percent'][key],
                               color=COLORS[seed], marker=['o', 's', '^'][seed], s=30,
                               label=f'DINN seed {seed}' if i == 0 else None)
                ax.scatter(i, baselines[model, case]['relative_errors_percent'][key], color='black', marker='x',
                           s=45, label='ODE calibration' if i == 0 else None)
            ax.set_xticks(range(3), ['Clean', 'Noisy', 'Missing'], fontsize=9)
            ax.set_title(key.replace('tau', 'tau '))
            ax.set_ylim(bottom=-.04)
            ax.set_ylabel('Rate error (%)', fontsize=9)
            ax.grid(axis='y', alpha=.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False, fontsize=9)
        figure(name, fig, title, caption,
               [f'results/{model}/{DATE}/{case}/{part}' for case in CASES
                for part in ['ode_baseline.json'] + [f'seed_{s}/metrics.json' for s in range(3)]])

    rate_plot('sird', ['beta', 'gamma', 'omega'], 'F02_sird_rate_recovery', 'Constrained SIRD rate recovery',
              'Each colored point is a network initialization seed; black crosses are direct ODE calibration. '
              'All seeds within a case use the same observations. Noisy means one fixed Gaussian noise realization '
              '(sigma 0.002); missing means 16 of 54 observation times removed, retaining both endpoints. '
              'Absolute relative error is 100 x |estimate - truth| / truth. These points are not confidence intervals.')

    two_sources = [f'results/two_region/{DATE}/clean/seed_{s}/trajectory.csv' for s in range(3)]
    two_arrays = [array(p) for p in two_sources]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.8), layout='constrained')
    for ax, region in zip(axes[0], [1, 2]):
        for state, color in zip('SIRD', ['#0072B2', '#D55E00', '#009E73', '#CC79A7']):
            for seed, a in enumerate(two_arrays):
                ax.plot(a['day'], a[f'{state}{region}'], color=color, linewidth=1, alpha=.7)
            ax.plot(a['day'], a[f'{state}{region}_true'], color=color, linestyle='--', linewidth=2, label=state)
        ax.set_title(f'Region {region}: states')
        style(ax, ylabel='Global population fraction')
        ax.legend(ncol=4, frameon=False, fontsize=8)
    for ax, region in zip(axes[1], [1, 2]):
        for seed, a in enumerate(two_arrays):
            error = np.max(np.abs(np.column_stack([a[f'{s}{region}']-a[f'{s}{region}_true'] for s in 'SIRD'])), axis=1)
            ax.plot(a['day'], error, color=COLORS[seed], label=f'Seed {seed}')
        ax.set_title(f'Region {region}: largest state error')
        style(ax, ylabel='Largest absolute error')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[1, 0].legend(frameon=False, fontsize=8)
    figure('F03_two_region_trajectories', fig, 'Two-region synthetic trajectories',
           'Clean-data fits show all three seeds. Solid curves are fitted neural states; dashed curves are '
           'synthetic ODE truth. Lower panels show the largest absolute state error in each region at each '
           'diagnostic time. All state fractions use the initial global population, not changing regional totals.', two_sources)
    rate_plot('two_region', ['beta1', 'gamma1', 'omega1', 'tau12', 'beta2', 'gamma2', 'omega2', 'tau21'],
              'F04_two_region_rate_recovery', 'Disease and directional mobility rate recovery',
              'Six disease rates and two directional movement rates were estimated from all eight observed '
              'states. tau12 means region 1 to region 2; tau21 means region 2 to region 1. The same rates move '
              'S, I, and R; D does not move. Noisy uses one fixed realization (sigma 0.001 in global population '
              'fractions). Missing retains 38 of 54 times. Points reflect three initializations, not repeated datasets. '
              'ODE calibration has lower overall trajectory RMSE in all three cases.')

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), layout='constrained')
    a = two_arrays[0]
    for region, color in zip([1, 2], COLORS):
        total = sum(a[f'{s}{region}'] for s in 'SIRD')
        truth = sum(a[f'{s}{region}_true'] for s in 'SIRD')
        axes[0].plot(a['day'], total, color=color, label=f'Region {region}, seed 0')
        axes[0].plot(a['day'], truth, color=color, linestyle='--')
    axes[0].plot(a['day'], sum(a[f'{s}{r}'] for r in [1, 2] for s in 'SIRD'), color='black', label='Global total')
    axes[0].set_title('Regional redistribution; global accounting')
    axes[0].legend(fontsize=8, frameon=False)
    style(axes[0], ylabel='S + I + R + D (global fraction)')
    for seed, a in enumerate(two_arrays):
        residual = np.max(np.abs(np.column_stack([a[f'{s}{r}_residual_per_day'] for r in [1, 2] for s in 'SIRD'])), axis=1)
        axes[1].plot(a['day'], residual, color=COLORS[seed], label=f'Seed {seed}')
    axes[1].set_title('Equation residuals remain nonzero')
    axes[1].legend(fontsize=8, frameon=False)
    style(axes[1], ylabel='Max absolute ODE residual per day')
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    figure('F05_population_accounting', fig, 'Population accounting and remaining residuals',
           'The left panel uses clean seed 0 and includes cumulative deceased states in each regional total. '
           'Movement changes regional totals while the global total remains one. Dashed regional curves '
           'are ODE truth. The right panel includes all clean seeds and shows nonzero differential-equation '
           'residuals. State nonnegativity, initial conditions, and global accounting are structural constraints; '
           'the ODE is enforced through a loss. Death monotonicity is not a structural guarantee.', two_sources)

    forecast_summary_path = f'results/sird/{DATE}/forecast_summary.json'
    forecast_summary = js(forecast_summary_path)
    obs_path = f'data/synthetic/sird/{DATE}/forecast_observations.csv'
    obs = array(obs_path)
    assert len(obs) == 21 and obs['day'][-1] == 60
    forecast_paths = [f'results/sird/{DATE}/forecast/seed_{s}/forecast.csv' for s in range(3)]
    forecasts = [array(p) for p in forecast_paths]
    forecast_rows = []
    for seed, a in enumerate(forecasts):
        actual = float(np.sqrt(np.mean(np.column_stack([a[s]-a[s+'_true'] for s in 'SIRD'])**2)))
        assert abs(actual - forecast_summary['neural'][seed]['rmse_all_states']) < 1e-14
        assert np.array_equal(a['day'], np.arange(61, 89))
        forecast_rows.append(dict(method='ODE integration with DINN-inferred rates', seed=seed,
                                  forecast_rmse_all_states=actual,
                                  infected_mae=forecast_summary['neural'][seed]['infected_mae']))
    forecast_rows.append(dict(method='Direct ODE calibration', seed='not_applicable',
                             forecast_rmse_all_states=forecast_summary['ode']['forecast_rmse_all_states'],
                             infected_mae=forecast_summary['ode']['forecast_infected_mae']))
    forecast_rows.append(dict(method='Persistence at day 60', seed='not_applicable',
                             forecast_rmse_all_states=forecast_summary['neural'][0]['persistence_rmse_all_states'],
                             infected_mae=forecast_summary['neural'][0]['persistence_infected_mae']))
    table('T06_forecast_comparison', forecast_rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), layout='constrained', gridspec_kw={'width_ratios': [1.5, 1]})
    for seed, a in enumerate(forecasts):
        axes[0].plot(a['day'], a['I'], color=COLORS[seed], label=f'DINN rates: seed {seed}')
    axes[0].plot(a['day'], a['I_true'], color='black', linestyle='--', label='Synthetic truth')
    axes[0].axhline(obs['I'][-1], color='#777777', linestyle=':', label='Day-60 persistence')
    axes[0].set_title('Holdout: days 61-88')
    style(axes[0], xlabel='Forecast day', ylabel='Infected population fraction')
    axes[0].legend(fontsize=8, frameon=False)
    values = [r['forecast_rmse_all_states'] for r in forecast_rows]
    axes[1].scatter(range(5), values, c=COLORS + ['black', '#777777'], s=48)
    axes[1].set_yscale('log')
    axes[1].set_xticks(range(5), ['Seed 0', 'Seed 1', 'Seed 2', 'ODE', 'Persist.'], rotation=35, ha='right')
    axes[1].set_ylabel('All-state forecast RMSE (log scale)')
    axes[1].set_title('Both comparators retained')
    axes[1].grid(axis='y', alpha=.2)
    figure('F06_chronological_forecast', fig, 'Chronological synthetic forecast test',
           'The models used only 21 clean observations through day 60 and collocation points within that '
           'window. Estimated rates were then integrated in the SIRD ODE for days 61-88; this is not direct '
           'neural extrapolation. The right panel uses a log scale and includes all three initializations, '
           'direct ODE calibration, and day-60 persistence. One clean, correctly specified synthetic cutoff '
           'does not establish real-world forecasting accuracy. ODE calibration has the lowest RMSE.',
           [forecast_summary_path, obs_path] + forecast_paths)

    captions = '# Presentation figures and tables\n\n'
    captions += ('Experiment records: October 11, 2026. Assets are derived without retraining. '
                 'The original notebook was public at commit `3c6275393c7f2aea437683c0e077996e6bff5d45`; '
                 'the later validations and these presentation assets are local until publication.\n\n')
    for entry in entries:
        captions += f"## {entry['id']}: {entry['title']}\n\n{entry['caption']}\n\nSources:\n\n"
        captions += ''.join(f'- `{p}`\n' for p in entry['sources']) + '\n'
    captions += ('## Table definitions\n\nT01: original rates and absolute relative errors.\n\n'
                 'T02: every rate estimate from the 21 later DINN fits and seven ODE calibrations.\n\n'
                 'T03: worst absolute relative rate errors and trajectory RMSE range across three seeds; '
                 'the ODE comparator has one fit per case.\n\n'
                 'T04: all 21 neural fits, with full precision state and residual diagnostics.\n\n'
                 'T05: original reproduction diagnostics.\n\n'
                 'T06: all-state RMSE and infected-state MAE on days 61-88.\n\n'
                 'Full-window trajectory diagnostics use 641 equally spaced points on days 0-160. '
                 'RMSE is the square root of the mean squared error over all stated times and states. '
                 'Original CSV states are printed at float32 precision; original aggregate diagnostics '
                 'are taken from the stored metrics. Scientific conclusions should use the metrics '
                 'and exact CSV tables rather than rounded labels.\n\n'
                 'SIRD and two-region observations are synthetic and fully observed. Noise and missing '
                 'patterns each have one fixed realization. Seed variation is optimizer initialization '
                 'variation, not statistical uncertainty. Disease rates and movement rates have units '
                 'per day. Fixed initial regional fractions 0.6 and 0.4 are the transmission denominators. '
                 'Real surveillance data, uncertain initial states, global identifiability, and deployment '
                 'are outside these experiments.\n')
    (tables / 'captions.md').write_text(captions)
    assets.append(tables / 'captions.md')
    (tables / 'figure_index.json').write_text(json.dumps(entries, indent=2) + '\n')
    assets.append(tables / 'figure_index.json')
    assert len(diagnostics) == 21 and len(recovery) == 144
    for row in recovery:
        expected_error = 100 * abs(row['estimated_rate_per_day'] - row['true_rate_per_day']) / row['true_rate_per_day']
        assert abs(expected_error-row['absolute_relative_error_percent']) < 1e-12
    assert all(row['negative_entries'] == 0 and row['initial_max_abs_error'] == 0 for row in diagnostics)
    assert max(row['conservation_max_abs_error'] for row in diagnostics) < 5e-15
    for path, expected in inputs.items():
        assert sha(ROOT / path) == expected, f'Input changed: {path}'
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    operation='Presentation derivatives only; no model fitting or source changes',
                    experiment_date=DATE, public_base_commit=original['source_commit'],
                    input_files=[dict(path=p, sha256=h) for p, h in sorted(inputs.items())],
                    generated_files=[dict(path=str(p.relative_to(target)), sha256=sha(p), size_bytes=p.stat().st_size)
                                     for p in sorted(assets)],
                    builder_sha256=sha(Path(__file__)),
                    checks={'all_21_fits_in_tables': True, 'all_7_ode_comparators_in_tables': True,
                            'forecast_csv_rmse_matches_record': True, 'input_hashes_unchanged': True})
    (tables / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'figures': len(entries), 'tables': 6,
                      'input_files': len(inputs), 'output': str(target)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT,
                        help='Fresh output root; source records always come from this repository.')
    args = parser.parse_args()
    build(args.output.resolve())
