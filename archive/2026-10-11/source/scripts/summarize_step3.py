"""Recheck numerical evidence from checkpoints and draw all-seed figures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import two_region_sird as m


def summarize(out):
    torch.set_num_threads(1)
    protocol=json.loads((out/'protocol.json').read_text())
    for file,expected in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==expected
    rows=json.loads((out/'neural_summary.json').read_text())
    assert len(rows)==9
    checkpoint_checks=[]
    for row in rows:
        folder=out/row['case']/f"seed_{row['seed']}"
        training=json.loads((folder/'training.json').read_text())
        model=m.TwoRegionDINN(m.Configuration(**training['configuration']),training['initial_rates'])
        model.load_state_dict(torch.load(folder/'weights.pt',weights_only=True))
        metrics,states,reference,res=m.diagnose(model,np.linspace(0,160,641))
        for key in metrics:
            if isinstance(metrics[key],dict):
                for name,value in metrics[key].items():
                    assert abs(value-row[key][name])<1e-12,(folder,key,name)
            else:
                assert abs(metrics[key]-row[key])<1e-12,(folder,key)
        assert metrics['negative_entries']==0
        assert metrics['initial_max_abs_error']==0
        assert metrics['conservation_max_abs_error']<5e-15
        assert np.isfinite(res).all()
        a=np.genfromtxt(out/f"{row['case']}_observations.csv",delimiter=',',names=True)
        observed=np.column_stack([a[n] for n in m.STATE_NAMES])
        pred=model(torch.tensor(a['day']/160,dtype=torch.float64).reshape(-1,1)).detach().numpy()
        assert abs(np.mean((pred-observed)**2)-training['final_data_mse'])<1e-15
        checkpoint_checks.append({'case':row['case'],'seed':row['seed'],
            'all_recorded_metrics_reproduced':True,'observation_MSE_reproduced':True,
            'state_constraints_passed':True})
    baseline_checks=[]
    for case in ['clean','noisy','missing']:
        path=out/f'{case}_observations.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['dataset_sha256'][case]
        a=np.genfromtxt(path,delimiter=',',names=True)
        expected_count=38 if case=='missing' else 54
        assert len(a)==expected_count and a['day'][0]==0 and a['day'][-1]==159
        fit=json.loads((out/case/'ode_baseline.json').read_text())
        assert fit['optimizer_success']
        observed=np.column_stack([a[n] for n in m.STATE_NAMES])
        fitted=m.simulate(a['day'],list(fit['parameters'].values()))
        assert abs(np.mean((fitted-observed)**2)-fit['fit_mse'])<1e-15
        singular=np.asarray(fit['jacobian_singular_values'])
        rank=int((singular>singular[0]*max(observed.size,8)*np.finfo(float).eps).sum())
        baseline_checks.append({'case':case,'objective_reproduced':True,'log_rate_jacobian_numerical_rank':rank})
    pilot=json.loads((out/'development_pilot.json').read_text())
    clean0=next(r for r in rows if r['case']=='clean' and r['seed']==0)
    pilot_identical=all(pilot['metrics'][k]==clean0[k] for k in pilot['metrics'])
    verification={'status':'PASS','checkpoint_checks':checkpoint_checks,'baseline_checks':baseline_checks,
        'source_and_data_hashes_verified':True,'original_notebook_preserved':True,
        'pilot_vs_fresh_clean_seed0_metrics_identical':pilot_identical,
        'summary_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    original=(ROOT/'notebooks/sird-dinn.ipynb').read_bytes()
    assert hashlib.sha1(b'blob '+str(len(original)).encode()+b'\0'+original).hexdigest()==protocol['original_notebook_git_blob']
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#087f8c','#de7b1d','#7756a5']
    fig,axes=plt.subplots(1,2,figsize=(9,3.1))
    for j,name in enumerate(['tau12','tau21']):
        ax=axes[j]
        for i,case in enumerate(['clean','noisy','missing']):
            for seed in [0,1,2]:
                row=next(r for r in rows if r['case']==case and r['seed']==seed)
                ax.scatter(i+(seed-1)*.1,row['relative_errors_percent'][name],color=colors[seed],s=40,
                    label=f'Neural seed {seed}' if i==0 else None)
            ode=json.loads((out/case/'ode_baseline.json').read_text())
            ax.scatter(i,ode['relative_errors_percent'][name],color='black',marker='x',s=70,
                label='ODE calibration' if i==0 else None)
        ax.set_xticks([0,1,2],['Clean','Noisy','Missing'])
        ax.set_title('Region 1 to 2' if j==0 else 'Region 2 to 1')
        ax.set_ylabel('Absolute relative error (%)');ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.015),ncol=4,frameon=False)
    fig.tight_layout(rect=[0,.11,1,1]);fig.savefig(out/'mobility_recovery.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(10,5.4))
    for j,name in enumerate(m.STATE_NAMES):
        ax=axes.flat[j]
        for seed in [0,1,2]:
            a=np.genfromtxt(out/'clean'/f'seed_{seed}'/'trajectory.csv',delimiter=',',names=True)
            ax.plot(a['day'],a[name],color=colors[seed],linewidth=1,label=f'Seed {seed}')
        ax.plot(a['day'],a[name+'_true'],'k--',linewidth=1,label='ODE truth')
        ax.set_title(name);ax.set_xlabel('Day');ax.set_ylabel('Combined-population fraction');ax.grid(alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(out/'clean_trajectories.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,2.6))
    for region,ax in enumerate(axes):
        columns=m.STATE_NAMES[4*region:4*region+4]
        for seed in [0,1,2]:
            a=np.genfromtxt(out/'clean'/f'seed_{seed}'/'trajectory.csv',delimiter=',',names=True)
            ax.plot(a['day'],sum(a[n] for n in columns),color=colors[seed],label=f'Seed {seed}')
        ax.plot(a['day'],sum(a[n+'_true'] for n in columns),'k--',linewidth=1,label='ODE truth')
        ax.set_title(f'Region {region+1} accounted population');ax.set_xlabel('Day')
        ax.set_ylabel('Combined-population fraction');ax.grid(alpha=.15)
    axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(out/'regional_population.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,2.7));ax.axis('off')
    equations=[r'$\dot S_j=-\beta_j S_j I_j/p_j+(L S)_j$',
        r'$\dot I_j=\beta_j S_j I_j/p_j-(\gamma_j+\omega_j)I_j+(L I)_j$',
        r'$\dot R_j=\gamma_j I_j+(L R)_j$',r'$\dot D_j=\omega_j I_j$']
    for y,eq in zip([.88,.63,.38,.13],equations):ax.text(.06,y,eq,fontsize=18,va='center')
    fig.tight_layout();fig.savefig(out/'model_equations.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(json.dumps({'status':'PASS','pilot_repetition_identical':pilot_identical,
        'neural_runs':len(rows),'ODE_baselines':len(baseline_checks),'figures':4},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('validation/step3/results'))
    args=parser.parse_args();out=args.output if args.output.is_absolute() else ROOT/args.output
    summarize(out)
