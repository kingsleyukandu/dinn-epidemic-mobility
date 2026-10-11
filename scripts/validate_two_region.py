"""Fixed two-region synthetic protocol with preserved runs and ODE baselines."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import two_region_sird as m
from sird_validation import save_csv
from project_paths import guard_output
OUT = ROOT/'runs/two_region'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Protocol exists; use a different --output path for a new experiment.')
    OUT.mkdir(parents=True, exist_ok=True)
    checks = json.loads((ROOT/'runs/checks/two_region/checks.json').read_text())
    assert checks['status'] == 'PASS'
    times = np.arange(0.,160.,3.)
    clean = m.simulate(times,m.SYNTHETIC_RATES)
    noisy = clean + np.random.default_rng(20261013).normal(0.,.001,clean.shape)
    removed = np.sort(np.random.default_rng(20261014).choice(np.arange(1,len(times)-1),16,replace=False))
    keep = np.array([i for i in range(len(times)) if i not in removed])
    cases = {'clean':(times,clean),'noisy':(times,noisy),'missing':(times[keep],clean[keep])}
    hashes = {}
    for case,(t,y) in cases.items():
        save_csv(OUT/f'{case}_observations.csv',[{'day':float(day),**dict(zip(m.STATE_NAMES,state.tolist()))}
            for day,state in zip(t,y)])
        hashes[case] = digest(OUT/f'{case}_observations.csv')
    source = ROOT/'notebooks/sird-dinn.ipynb'
    blob = source.read_bytes()
    original_blob = hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
    assert original_blob == '2585e90aba6c964101b55e642a62279ea3318ece'
    files = ['src/two_region_sird.py','src/sird_validation.py','scripts/validate_two_region.py','tests/check_two_region_model.py']
    dump(OUT/'protocol.json',{
        'frozen_before_validation_runs':True,'created_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Controlled synthetic two-region SIRD mobility implementation and validation.',
        'configuration':asdict(m.Configuration()),'seeds':[0,1,2],'initial_rates':m.INITIAL_RATES.tolist(),
        'synthetic_truth':dict(zip(m.RATE_NAMES,m.SYNTHETIC_RATES.tolist())),
        'initial_state':dict(zip(m.STATE_NAMES,m.INITIAL_STATE.tolist())),
        'reference_region_population_fractions':m.REFERENCE.tolist(),
        'model':{'state_units':'Fractions of combined initial population, including cumulative deaths.',
            'incidence':'beta_j*S_j*I_j/p_j with fixed initial regional reference p_j.',
            'movement':'tau12 is region 1 to 2; tau21 is region 2 to 1; common to S, I, R; D does not move.',
            'hard_constraints':['Nonnegative neural states','Combined accounted population equals one','Exact initial neural state'],
            'soft_constraints':['ODE residuals','Regional balance identity','Nondecreasing deceased trajectories']},
        'observations':{'states':m.STATE_NAMES,'known_initial_state':True,
            'clean':{'timepoints':54,'days':'0,3,...,159'},
            'noisy':{'rng_seed':20261013,'sigma_combined_population_fraction':.001,
                'measurement_noise':'Independent Gaussian; no clipping or renormalization.',
                'negative_measurements':int((noisy<0).sum())},
            'missing':{'rng_seed':20261014,'removed_days':times[removed].tolist(),'retained_timepoints':len(keep)}},
        'ode_comparator':{'same_observations_initial_state_and_starting_rates':True,
            'method':'Least squares on log rates; DOP853 integration.',
            'disease_rate_bounds':[.0001,3.],'movement_rate_bounds':[.00001,.5]},
        'selection':'All final seeds reported. Fixed optimizer budget; no true-error stopping or best-seed selection.',
        'development_pilot':'One seed-0 clean pilot preceded the fixed protocol; excluded from validation totals.',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'torch':torch.__version__,'dtype':'float64','torch_threads_per_process':1,'device':'CPU','platform':platform.platform()},
        'source_sha256':{file:digest(ROOT/file) for file in files},'dataset_sha256':hashes,
        'original_source_commit':'3c6275393c7f2aea437683c0e077996e6bff5d45','original_notebook_git_blob':original_blob,
        'limitations':['All eight states observed; fixed known initial state.',
            'Only three neural initializations and one fixed dataset per imperfection.',
            'No real surveillance or mobility data; no forecast or uncertainty benchmark in Step 3.',
            'Common S/I/R movement and fixed reference transmission denominators are assumptions.',
            'Symmetric balanced regions do not identify the common movement scale.']})
    dump(OUT/'model_checks.json',checks)
    pilot = ROOT/'archive/development/two_region'
    if (pilot/'metrics.json').exists():
        dump(OUT/'development_pilot.json',{'excluded_from_validation_totals':True,
            'metrics':json.loads((pilot/'metrics.json').read_text()),
            'training':json.loads((pilot/'training.json').read_text())})
    print(f'Frozen protocol saved: {OUT}')


def data(case):
    protocol = json.loads((OUT/'protocol.json').read_text())
    for file,expected in protocol['source_sha256'].items():
        assert digest(ROOT/file) == expected,(file,'source changed')
    path = OUT/f'{case}_observations.csv'
    assert digest(path) == protocol['dataset_sha256'][case]
    a = np.genfromtxt(path,delimiter=',',names=True)
    return a['day'],np.column_stack([a[n] for n in m.STATE_NAMES])


def run(case,seed):
    t,y = data(case)
    folder = OUT/case/f'seed_{seed}'
    if (folder/'metrics.json').exists():
        raise RuntimeError('Completed run exists; preserve it.')
    model,training,history = m.train(t,y,seed,output=folder)
    grid = np.linspace(0.,160.,641)
    metrics,states,reference,res = m.diagnose(model,grid)
    dump(folder/'metrics.json',metrics)
    save_csv(folder/'trajectory.csv',[{'day':float(day),
        **dict(zip(m.STATE_NAMES,pred.tolist())),
        **dict(zip([n+'_true' for n in m.STATE_NAMES],truth.tolist())),
        **dict(zip([n+'_residual_per_day' for n in m.STATE_NAMES],r.tolist()))}
        for day,pred,truth,r in zip(grid,states,reference,res)])
    print(json.dumps({'case':case,'seed':seed,'metrics':metrics},indent=2))


def baselines():
    for case in ['clean','noisy','missing']:
        t,y = data(case)
        folder = OUT/case;folder.mkdir(exist_ok=True)
        if (folder/'ode_baseline.json').exists():
            raise RuntimeError('Completed baseline exists; preserve it.')
        result = m.fit_ode(t,y)
        grid = np.linspace(0.,160.,641)
        result['trajectory_rmse'] = float(np.sqrt(np.mean((m.simulate(grid,list(result['parameters'].values()))-
            m.simulate(grid,m.SYNTHETIC_RATES))**2)))
        dump(folder/'ode_baseline.json',result)
        print(case,json.dumps(result),flush=True)


def aggregate():
    rows = [];rate_rows = []
    for case in ['clean','noisy','missing']:
        for seed in [0,1,2]:
            folder = OUT/case/f'seed_{seed}'
            metric = json.loads((folder/'metrics.json').read_text())
            training = json.loads((folder/'training.json').read_text())
            rows.append({'case':case,'seed':seed,**metric,'training':training})
            for name in m.RATE_NAMES:
                rate_rows.append({'case':case,'method':'DINN','seed':seed,'parameter':name,
                    'true_rate_per_day':dict(zip(m.RATE_NAMES,m.SYNTHETIC_RATES))[name],
                    'estimated_rate_per_day':metric['parameters'][name],
                    'relative_error_percent':metric['relative_errors_percent'][name]})
        baseline = json.loads((OUT/case/'ode_baseline.json').read_text())
        for name in m.RATE_NAMES:
            rate_rows.append({'case':case,'method':'ODE','seed':'not_applicable','parameter':name,
                'true_rate_per_day':dict(zip(m.RATE_NAMES,m.SYNTHETIC_RATES))[name],
                'estimated_rate_per_day':baseline['parameters'][name],
                'relative_error_percent':baseline['relative_errors_percent'][name]})
    dump(OUT/'neural_summary.json',rows)
    save_csv(OUT/'parameter_recovery.csv',rate_rows)
    save_csv(OUT/'diagnostics.csv',[{'case':r['case'],'seed':r['seed'],
        **{k:v for k,v in r.items() if k not in ['case','seed','parameters','relative_errors_percent','rmse_by_state','training']}}
        for r in rows])
    print('Nine final neural runs and three ODE baselines aggregated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','baselines','aggregate'])
    parser.add_argument('--case',choices=['clean','noisy','missing'])
    parser.add_argument('--seed',type=int,choices=[0,1,2])
    parser.add_argument('--output',type=Path,default=Path('runs/two_region'))
    args = parser.parse_args()
    OUT = guard_output(args.output if args.output.is_absolute() else ROOT/args.output,ROOT)
    if args.action == 'prepare':prepare()
    elif args.action == 'run':
        if args.case is None or args.seed is None:parser.error('--case and --seed are required for run')
        run(args.case,args.seed)
    elif args.action == 'baselines':baselines()
    else:aggregate()
