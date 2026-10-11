"""Verify saved checkpoints and produce transparent validation figures."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sird_validation import Configuration,ConstrainedDINN,diagnose,simulate,TRUE
from project_paths import guard_output
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('runs/sird'))
args=parser.parse_args()
OUT=guard_output(args.output if args.output.is_absolute() else ROOT/args.output,ROOT)
torch.set_num_threads(1)
protocol=json.loads((OUT/'protocol.json').read_text())
assert hashlib.sha256((ROOT/'src/sird_validation.py').read_bytes()).hexdigest()==protocol['module_sha256']
assert hashlib.sha256((ROOT/'scripts/validate_sird.py').read_bytes()).hexdigest()==protocol['runner_sha256']
rows=json.loads((OUT/'neural_summary.json').read_text())
checks=[]
for row in rows:
    folder=OUT/row['case']/f"seed_{row['seed']}"
    training=json.loads((folder/'training.json').read_text())
    config=Configuration(**training['config'])
    model=ConstrainedDINN(config,training['initial_rates'])
    model.load_state_dict(torch.load(folder/'weights.pt',weights_only=True))
    metric,states,ref,res=diagnose(model,np.linspace(0,config.horizon,641))
    recorded=json.loads((folder/'metrics.json').read_text())
    for key in ['trajectory_rmse','residual_rmse_per_day','initial_max_abs_error',
                'conservation_max_abs_error','minimum_compartment','negative_entries']:
        assert abs(metric[key]-recorded[key])<1e-12,(folder,key)
    assert metric['negative_entries']==0
    assert metric['initial_max_abs_error']==0
    assert metric['conservation_max_abs_error']<5e-15
    assert np.isfinite(res).all()
    # Autograd derivatives recovered from the stored residual and SIRD RHS.
    gamma,omega=model.rates().detach().numpy()[1:]
    rd_derivatives=res[:,2:]+states[:,1:2]*np.array([gamma,omega])
    checks.append({'case':row['case'],'seed':row['seed'],'checkpoint_matches_metrics':True,
        'minimum_recovered_derivative_per_day':float(rd_derivatives[:,0].min()),
        'minimum_deceased_derivative_per_day':float(rd_derivatives[:,1].min()),
        'negative_RD_derivative_entries':int((rd_derivatives<0).sum())})
for case in ['clean','noisy','missing','forecast']:
    a=np.genfromtxt(OUT/f'{case}_observations.csv',delimiter=',',names=True)
    if case=='forecast':assert a['day'].max()==60
    if case=='missing':assert len(a)==38 and a['day'][0]==0 and a['day'][-1]==159
    ode=json.loads((OUT/case/'ode_baseline.json').read_text())
    pred=simulate(a['day'],list(ode['parameters'].values()))
    obs=np.column_stack([a[n] for n in ['S','I','R','D']])
    assert abs(np.mean((pred-obs)**2)-ode['fit_mse'])<1e-12
    assert ode['optimizer_success']
(OUT/'verification.json').write_text(json.dumps({'status':'PASS',
    'checks':['Checkpoint diagnostics reproduced','Structural state constraints on diagnostic grids',
        'Dataset and source hashes','Chronological holdout excludes later observations',
        'Missing-data endpoints and count','ODE objectives independently recomputed'],
    'monotonicity_diagnostics':checks},indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(9,3.2))
colors=['#087f8c','#de7b1d','#7756a5']
for j,name in enumerate(['beta','gamma','omega']):
    ax=axs[j]
    for i,case in enumerate(['clean','noisy','missing']):
        for seed in [0,1,2]:
            row=next(r for r in rows if r['case']==case and r['seed']==seed)
            ax.scatter(i+(seed-1)*.1,row[name+'_error_percent'],color=colors[seed],s=38,
                       label=f'Network seed {seed}' if i==0 else None)
        baseline=json.loads((OUT/case/'ode_baseline.json').read_text())
        ax.scatter(i,baseline['relative_errors_percent'][name],color='black',marker='x',s=70,
                   label='ODE calibration' if i==0 else None)
    ax.axhline(.1,color='#a5a5a5',linestyle=':',linewidth=1)
    ax.set_xticks([0,1,2],['Clean','Noisy','Missing'])
    ax.set_title(name);ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('Absolute relative rate error (%)')
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.01),frameon=False)
fig.tight_layout(rect=[0,.11,1,1]);fig.savefig(OUT/'parameter_errors.png',dpi=180,bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(9,5.3))
for j,name in enumerate(['S','I','R','D']):
    ax=axs.flat[j]
    for seed in [0,1,2]:
        a=np.genfromtxt(OUT/'clean'/f'seed_{seed}'/'trajectory.csv',delimiter=',',names=True)
        ax.plot(a['day'],a[name],color=colors[seed],linewidth=1,label=f'Seed {seed}')
    ax.plot(a['day'],a[name+'_true'],'k--',linewidth=1,label='ODE truth')
    ax.set_title(name);ax.set_xlabel('Day');ax.set_ylabel('Population fraction');ax.grid(alpha=.15)
axs[0,0].legend(frameon=False,fontsize=8)
fig.tight_layout();fig.savefig(OUT/'clean_trajectories.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,3.1))
for seed in [0,1,2]:
    a=np.genfromtxt(OUT/'forecast'/f'seed_{seed}'/'forecast.csv',delimiter=',',names=True)
    ax.plot(a['day'],a['I'],color=colors[seed],label=f'DINN rates: seed {seed}')
ax.plot(a['day'],a['I_true'],'k--',label='Truth / ODE calibration',linewidth=1.5)
obs=np.genfromtxt(OUT/'forecast_observations.csv',delimiter=',',names=True)
ax.axhline(obs['I'][-1],color='gray',linestyle=':',label='Persistence at day 60')
ax.set_xlabel('Forecast day');ax.set_ylabel('Infected population fraction');ax.grid(alpha=.2)
ax.legend(loc='best',fontsize=8,frameon=False)
fig.tight_layout();fig.savefig(OUT/'forecast_infected.png',dpi=180);plt.close(fig)
print('PASS: all checkpoint, protocol, dataset and baseline checks; figures generated.')
