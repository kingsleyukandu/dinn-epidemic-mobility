"""Read-only integrity and checkpoint verification for the relocated records."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import sird_validation as single
import two_region_sird as two
from project_paths import guard_output


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(actual, expected, label):
    if isinstance(actual, dict):
        for key, value in actual.items():
            same(value, expected[key], f'{label}.{key}')
    else:
        assert abs(actual - expected) < 1e-12, (label, actual, expected)


def verify():
    torch.set_num_threads(1)
    mapping = json.loads((ROOT / 'docs/provenance/path_map.json').read_text())
    for entry in mapping['entries']:
        assert digest(ROOT / entry['preserved_path']) == entry['baseline_sha256'], entry['old_path']
        assert digest(ROOT / entry['new_path']) == entry['current_sha256'], entry['new_path']

    original = (ROOT / 'notebooks/sird-dinn.ipynb').read_bytes()
    assert hashlib.sha1(b'blob ' + str(len(original)).encode() + b'\0' + original).hexdigest() == '2585e90aba6c964101b55e642a62279ea3318ece'
    archive = ROOT / 'archive/2026-10-11/source'
    checkpoint_checks = []
    baseline_checks = []
    for name, module, cases in [('sird', single, ['clean', 'noisy', 'missing', 'forecast']),
                                ('two_region', two, ['clean', 'noisy', 'missing'])]:
        result = ROOT / 'results' / name / '2026-10-11'
        data = ROOT / 'data/synthetic' / name / '2026-10-11'
        protocol = json.loads((result / 'protocol.json').read_text())
        if name == 'sird':
            assert digest(ROOT / 'src/sird_validation.py') == protocol['module_sha256']
            assert digest(archive / 'scripts/validate_step2.py') == protocol['runner_sha256']
        else:
            for path, expected in protocol['source_sha256'].items():
                assert digest(archive / path) == expected, path
                if path.startswith('src/'):
                    assert digest(ROOT / path) == expected, path
        rows = json.loads((result / 'neural_summary.json').read_text())
        assert {(r['case'], r['seed']) for r in rows} == {(c, s) for c in cases for s in range(3)}
        assert len(rows) == len(cases) * 3
        for case in cases:
            path = data / f'{case}_observations.csv'
            assert digest(path) == protocol['dataset_sha256'][case]
            observed = np.genfromtxt(path, delimiter=',', names=True)
            states = ['S', 'I', 'R', 'D'] if name == 'sird' else module.STATE_NAMES
            y = np.column_stack([observed[n] for n in states])
            assert len(observed) == (21 if case == 'forecast' else 38 if case == 'missing' else 54)
            assert observed['day'][0] == 0
            assert observed['day'][-1] == (60 if case == 'forecast' else 159)
            for seed in range(3):
                folder = result / case / f'seed_{seed}'
                training = json.loads((folder / 'training.json').read_text())
                config = module.Configuration(**training['config' if name == 'sird' else 'configuration'])
                model_type = module.ConstrainedDINN if name == 'sird' else module.TwoRegionDINN
                model = model_type(config, training['initial_rates'])
                model.load_state_dict(torch.load(folder / 'weights.pt', map_location='cpu', weights_only=True))
                metrics, prediction, reference, residual = module.diagnose(model, np.linspace(0, config.horizon, 641))
                recorded = json.loads((folder / 'metrics.json').read_text())
                same(metrics, recorded, f'{name}.{case}.{seed}')
                assert metrics['negative_entries'] == 0 and metrics['initial_max_abs_error'] == 0
                assert metrics['conservation_max_abs_error'] < 5e-15 and np.isfinite(residual).all()
                tau = torch.tensor(observed['day']/config.horizon, dtype=torch.float64).reshape(-1, 1)
                fitted = model(tau).detach().numpy()
                assert abs(np.mean((fitted-y)**2) - training['final_data_mse']) < 1e-15
                if case == 'forecast':
                    forecast = json.loads((folder / 'forecast.json').read_text())
                    days = np.arange(61., 89.)
                    inferred = module.simulate(days, list(metrics['parameters'].values()))
                    truth = module.simulate(days, module.TRUE)
                    assert abs(np.sqrt(np.mean((inferred-truth)**2))-forecast['rmse_all_states']) < 1e-12
                checkpoint_checks.append({'model': name, 'case': case, 'seed': seed, 'status': 'PASS'})
            baseline = json.loads((result / case / 'ode_baseline.json').read_text())
            assert baseline['optimizer_success']
            fitted = module.simulate(observed['day'], list(baseline['parameters'].values()))
            assert abs(np.mean((fitted-y)**2)-baseline['fit_mse']) < 1e-12
            baseline_checks.append({'model': name, 'case': case, 'status': 'PASS'})
    return {'status': 'PASS', 'verified_utc': datetime.now(timezone.utc).isoformat(),
            'mapped_baseline_files': len(mapping['entries']), 'original_notebook_preserved': True,
            'historical_protocols_and_sources_preserved': True, 'source_and_data_hashes_verified': True,
            'checkpoint_checks': checkpoint_checks, 'baseline_checks': baseline_checks,
            'training_rerun': False, 'recorded_outputs_modified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='Optional new verification JSON outside recorded folders.')
    args = parser.parse_args()
    output = guard_output(args.output if args.output.is_absolute() else ROOT/args.output, ROOT) if args.output else None
    result = verify()
    if output:
        if output.exists():
            raise RuntimeError('Verification output already exists; choose a new path.')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'preserved_files': result['mapped_baseline_files'],
                      'checkpoints': len(result['checkpoint_checks']), 'ODE_baselines': len(result['baseline_checks'])}))
