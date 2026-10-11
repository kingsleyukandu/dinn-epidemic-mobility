"""Fixed-budget synthetic validation. Run prepare before any run; aggregate last."""
import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sird_validation import Configuration,TRUE,Y0,NAMES,simulate,train,diagnose,fit_ode,save_csv

OUT=ROOT/'validation'/'step2'
INITIAL=[.8,.1,.15]

def dump(path,value):
    path.write_text(json.dumps(value,indent=2))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Protocol exists; preserve it.')
    times=np.arange(0.,160.,3.)
    clean=simulate(times,TRUE)
    noisy=clean+np.random.default_rng(20261011).normal(0,.002,clean.shape)
    removed=np.sort(np.random.default_rng(20261012).choice(np.arange(1,len(times)-1),16,replace=False))
    kept=np.array([i for i in range(len(times)) if i not in removed])
    cases={'clean':(times,clean),'noisy':(times,noisy),'missing':(times[kept],clean[kept]),
           'forecast':(times[times<=60],clean[times<=60])}
    hashes={}
    for name,(t,y) in cases.items():
        rows=[{'day':float(a),**dict(zip(['S','I','R','D'],b.tolist()))} for a,b in zip(t,y)]
        save_csv(OUT/f'{name}_observations.csv',rows)
        hashes[name]=hashlib.sha256((OUT/f'{name}_observations.csv').read_bytes()).hexdigest()
    protocol={'frozen_before_validation_runs':True,'created_utc':'2026-10-11',
        'purpose':'Controlled synthetic validation; not clinical or operational forecasting.',
        'configuration':asdict(Configuration()),'seeds':[0,1,2],
        'initial_rates':INITIAL,'truth':dict(zip(NAMES,TRUE.tolist())),
        'initial_state':Y0.tolist(),'observation_states':['S','I','R','D'],
        'clean':{'observations':54,'days':'0 to 159 inclusive, every 3 days'},
        'noisy':{'sigma_population_fraction':.002,'rng_seed':20261011,
            'noise':'Independent additive Gaussian on all states; no clipping or renormalization.',
            'negative_measurements':int((noisy<0).sum())},
        'missing':{'removed_timepoints':times[removed].tolist(),'retained':len(kept),'rng_seed':20261012},
        'forecast':{'training_last_day':60,'forecast_days':[61,88],
            'description':'Infer rates using only days 0 to 60 and collocation through day 60; integrate SIRD with inferred rates from the known initial state for days 61 to 88.',
            'comparators':['ODE least squares','Persistence at day 60']},
        'comparator':'One deterministic ODE least-squares fit per dataset with the same starting rates, observations and known initial state.',
        'selection':'All three neural seeds reported, no best-seed selection or truth-based stopping.',
        'dataset_sha256':hashes,
        'module_sha256':hashlib.sha256((ROOT/'src/sird_validation.py').read_bytes()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,
            'scipy':scipy.__version__,'torch':torch.__version__,'platform':platform.platform(),
            'torch_threads_per_process':1,'device':'CPU'}}
    dump(OUT/'protocol.json',protocol)
    print(json.dumps(protocol,indent=2))

def data(case):
    protocol=json.loads((OUT/'protocol.json').read_text())
    path=OUT/f'{case}_observations.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['dataset_sha256'][case]
    assert hashlib.sha256((ROOT/'src/sird_validation.py').read_bytes()).hexdigest()==protocol['module_sha256']
    a=np.genfromtxt(path,delimiter=',',names=True)
    return a['day'],np.column_stack([a[n] for n in ['S','I','R','D']])

def run(case,seed):
    times,y=data(case)
    folder=OUT/case/f'seed_{seed}'
    if (folder/'metrics.json').exists():raise RuntimeError('Completed run exists; preserve it.')
    config=Configuration() if case!='forecast' else replace(Configuration(),horizon=60.)
    model,training,history=train(times,y,INITIAL,seed,config,output=folder)
    grid=np.linspace(0,config.horizon,641)
    metrics,states,reference,res=diagnose(model,grid)
    dump(folder/'metrics.json',metrics)
    save_csv(folder/'trajectory.csv',[{'day':float(t),
        **{n:float(v) for n,v in zip(['S','I','R','D'],pred)},
        **{n+'_true':float(v) for n,v in zip(['S','I','R','D'],truth)},
        **{n+'_residual_per_day':float(v) for n,v in zip(['S','I','R','D'],r)}}
        for t,pred,truth,r in zip(grid,states,reference,res)])
    if case=='forecast':
        future=np.arange(61.,89.)
        forecast=simulate(future,list(metrics['parameters'].values()))
        truth=simulate(future,TRUE)
        persistence=np.tile(y[-1],(len(future),1))
        dump(folder/'forecast.json',{'method':'SIRD integration using DINN-inferred rates',
            'rmse_all_states':float(np.sqrt(np.mean((forecast-truth)**2))),
            'infected_mae':float(np.mean(np.abs(forecast[:,1]-truth[:,1]))),
            'persistence_rmse_all_states':float(np.sqrt(np.mean((persistence-truth)**2))),
            'persistence_infected_mae':float(np.mean(np.abs(persistence[:,1]-truth[:,1])))})
        save_csv(folder/'forecast.csv',[{'day':float(t),
            **{n:float(v) for n,v in zip(['S','I','R','D'],pred)},
            **{n+'_true':float(v) for n,v in zip(['S','I','R','D'],ref)}}
            for t,pred,ref in zip(future,forecast,truth)])
    print(json.dumps({'case':case,'seed':seed,**metrics},indent=2))

def baselines():
    for case in ['clean','noisy','missing','forecast']:
        t,y=data(case);result=fit_ode(t,y,INITIAL)
        folder=OUT/case;folder.mkdir(exist_ok=True)
        if case=='forecast':
            future=np.arange(61.,89.);pred=simulate(future,list(result['parameters'].values()))
            truth=simulate(future,TRUE)
            result['forecast_rmse_all_states']=float(np.sqrt(np.mean((pred-truth)**2)))
            result['forecast_infected_mae']=float(np.mean(np.abs(pred[:,1]-truth[:,1])))
        else:
            dense=np.linspace(0,160,641)
            result['trajectory_rmse']=float(np.sqrt(np.mean((simulate(dense,list(result['parameters'].values()))-simulate(dense,TRUE))**2)))
        dump(folder/'ode_baseline.json',result)
        print(case,json.dumps(result))
    # S and I identify total removal, but cannot separate gamma and omega.
    alt=TRUE.copy();alt[1]=.1;alt[2]=TRUE[1]+TRUE[2]-alt[1]
    a=simulate(np.linspace(0,160,641),TRUE);b=simulate(np.linspace(0,160,641),alt)
    dump(OUT/'partial_observability.json',{'alternative_rates':alt.tolist(),
        'max_abs_SI_difference':float(np.max(np.abs(a[:,:2]-b[:,:2]))),
        'max_abs_RD_difference':float(np.max(np.abs(a[:,2:]-b[:,2:]))),
        'interpretation':'With only S and I observed, gamma and omega enter as their sum and cannot be separately identified.'})

def aggregate():
    rows=[];forecasts=[]
    for case in ['clean','noisy','missing','forecast']:
        for seed in [0,1,2]:
            folder=OUT/case/f'seed_{seed}'
            m=json.loads((folder/'metrics.json').read_text());t=json.loads((folder/'training.json').read_text())
            rows.append({'case':case,'method':'DINN','seed':seed,
                **m['parameters'],**{n+'_error_percent':v for n,v in m['relative_errors_percent'].items()},
                **{k:v for k,v in m.items() if k not in ['parameters','relative_errors_percent']},
                'runtime_seconds':t['runtime_seconds'],'final_data_mse':t['final_data_mse']})
            if case=='forecast':forecasts.append({'method':'DINN','seed':seed,**json.loads((folder/'forecast.json').read_text())})
    save_csv(OUT/'neural_summary.csv',rows)
    dump(OUT/'neural_summary.json',rows)
    dump(OUT/'forecast_summary.json',{'neural':forecasts,'ode':json.loads((OUT/'forecast/ode_baseline.json').read_text())})
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','baselines','aggregate'])
    p.add_argument('--case',choices=['clean','noisy','missing','forecast']);p.add_argument('--seed',type=int)
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run(args.case,args.seed)
    elif args.action=='baselines':baselines()
    else:aggregate()
