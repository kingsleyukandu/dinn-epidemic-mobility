"""Execute the original SIRD notebook in a fresh Python process and append diagnostics.

This is a reproduction audit, not a new training method. The original equations,
network, optimizer, weights, seeds and 20,001 Adam steps are preserved. The only
training-cell additions store detached diagnostic values every 100 iterations.
"""
import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import nbformat


DIAGNOSTICS = r'''
import csv, json, platform, sys
from pathlib import Path
from datetime import datetime, timezone
import scipy

audit_dir = Path('audit/results')
audit_dir.mkdir(parents=True, exist_ok=True)
audit_times = np.linspace(0.0, T_max, 641)
audit_tau = torch.tensor(audit_times/T_max, dtype=torch.float32).reshape(-1,1)
audit_tau.requires_grad_(True)
audit_u = model(audit_tau)
audit_s, audit_i, audit_r, audit_d = [audit_u[:, k:k+1] for k in range(4)]
audit_derivatives = torch.cat([d_dtau(v, audit_tau) for v in [audit_s,audit_i,audit_r,audit_d]],dim=1)/T_max
audit_b, audit_g, audit_o = rates()
audit_rhs = torch.cat([-audit_b*audit_s*audit_i,
    audit_b*audit_s*audit_i-(audit_g+audit_o)*audit_i,
    audit_g*audit_i, audit_o*audit_i],dim=1)
audit_res = (audit_derivatives-audit_rhs).detach().numpy()
audit_states = audit_u.detach().numpy()
audit_true = simulate(audit_times, TRUE)
audit_final = {k:float(v.detach()) for k,v in zip(['beta','gamma','omega'],rates())}
audit_rel = {k:abs(audit_final[k]-TRUE[k])/abs(TRUE[k]) for k in TRUE}
audit_grid_min = {k:float(audit_states[:,j].min()) for j,k in enumerate(['s','i','r','d'])}
audit_grid_max = {k:float(audit_states[:,j].max()) for j,k in enumerate(['s','i','r','d'])}
audit_final_losses = {
    'data_mse':float(mse(model(tau_train),y_obs).detach()),
    'initial_condition_mse':float(mse(model(tau0),y0).detach()),
    'equation_residual_mse_tau_training_grid':float((physics_residual()**2).mean().detach())}
audit_final_losses['weighted_total'] = W_data*audit_final_losses['data_mse']+W_ic*audit_final_losses['initial_condition_mse']+W_phy*audit_final_losses['equation_residual_mse_tau_training_grid']
audit_metrics = {
    'utc_time':datetime.now(timezone.utc).isoformat(),
    'configuration':{'seed':0,'device':'cpu','dtype':'float32','cpu_threads':torch.get_num_threads(),
        'hidden_width':32,'hidden_layers':3,'activation':'tanh','optimizer':'Adam',
        'learning_rate':1e-3,'optimizer_steps':iters+1,'T_max_days':T_max,
        'population':N_POP,'initial_state':Y0,'observations':len(t_train),
        'first_observation_day':float(t_train[0]),'last_observation_day':float(t_train[-1]),
        'observation_spacing_days':3,'collocation_points':len(t_dense),
        'audit_grid_points':len(audit_times),'loss_weights':{'data':W_data,'equations':W_phy,'initial':W_ic}},
    'environment':{'python':sys.version.split()[0],'torch':torch.__version__,
        'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':plt.matplotlib.__version__,
        'platform':platform.platform()},
    'true_parameters':TRUE,'initial_parameter_guesses':Guess,'recovered_parameters':audit_final,
    'relative_errors_fraction':audit_rel,'relative_errors_percent':{k:100*v for k,v in audit_rel.items()},
    'all_parameter_errors_below_0_1_percent':all(v<0.001 for v in audit_rel.values()),
    'final_losses':audit_final_losses,
    'trajectory_rmse_by_compartment':{k:float(np.sqrt(np.mean((audit_states[:,j]-audit_true[:,j])**2))) for j,k in enumerate(['s','i','r','d'])},
    'trajectory_rmse_overall':float(np.sqrt(np.mean((audit_states-audit_true)**2))),
    'equation_residual_rmse_per_day_by_compartment':{k:float(np.sqrt(np.mean(audit_res[:,j]**2))) for j,k in enumerate(['s','i','r','d'])},
    'equation_residual_max_abs_per_day':float(np.abs(audit_res).max()),
    'population_conservation_max_abs_error':float(np.max(np.abs(audit_states.sum(axis=1)-1))),
    'minimum_by_compartment':audit_grid_min,'maximum_by_compartment':audit_grid_max,
    'negative_entries':int(np.sum(audit_states<0)),
    'initial_condition_max_abs_error':float(np.max(np.abs(audit_states[0]-np.array(Y0)))),
    'independent_ode_grid_conservation_max_abs_error':float(np.max(np.abs(audit_true.sum(axis=1)-1))),
    'recovery_death_monotonicity_min_daily_derivative':{'r':float(audit_derivatives.detach().numpy()[:,2].min()),'d':float(audit_derivatives.detach().numpy()[:,3].min())}}

(audit_dir/'metrics.json').write_text(json.dumps(audit_metrics,indent=2))
with (audit_dir/'loss_history.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['iteration','data_mse','equation_mse_tau','initial_mse','weighted_total','beta','gamma','omega']);w.writerows(audit_history)
with (audit_dir/'parameter_recovery.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['parameter','true','initial_guess','recovered','relative_error_fraction','relative_error_percent'])
    for k in TRUE:w.writerow([k,TRUE[k],Guess[k],audit_final[k],audit_rel[k],100*audit_rel[k]])
with (audit_dir/'trajectory_audit.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['day','s_true','i_true','r_true','d_true','s_dinn','i_dinn','r_dinn','d_dinn','s_residual_per_day','i_residual_per_day','r_residual_per_day','d_residual_per_day'])
    for j,t in enumerate(audit_times):w.writerow([t,*audit_true[j],*audit_states[j],*audit_res[j]])

fig,axs=plt.subplots(2,2,figsize=(10,6),sharex=True)
for j,(ax,k) in enumerate(zip(axs.ravel(),['S','I','R','D'])):
    ax.plot(audit_times,audit_true[:,j],label='ODE reference',lw=1.8)
    ax.plot(audit_times,audit_states[:,j],'--',label='DINN fresh run',lw=1.3)
    ax.scatter(t_train,y_train[:,j],s=6,alpha=.45,label='Synthetic observations')
    ax.set_title(k);ax.set_xlabel('Time (days)');ax.set_ylabel('Population fraction');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(audit_dir/'trajectory_comparison.png',dpi=180);plt.show()
hist=np.asarray(audit_history)
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
for j,k in [(1,'Data'),(2,'Equations in normalized time'),(3,'Initial condition')]:axs[0].semilogy(hist[:,0],np.maximum(hist[:,j],1e-20),label=k)
axs[0].set_xlabel('Adam iteration');axs[0].set_ylabel('Unweighted MSE');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
for j,k in [(5,'beta'),(6,'gamma'),(7,'omega')]:axs[1].plot(hist[:,0],hist[:,j],label=k);axs[1].axhline(TRUE[k],lw=.6,alpha=.4,color='gray')
axs[1].set_xlabel('Adam iteration');axs[1].set_ylabel('Rate per day');axs[1].legend();axs[1].grid(alpha=.2)
fig.tight_layout();fig.savefig(audit_dir/'training_diagnostics.png',dpi=180);plt.show()
print(json.dumps(audit_metrics,indent=2))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='notebooks/sird-dinn.ipynb')
    parser.add_argument('--output', default='runs/original/sird-dinn-reproduction.ipynb')
    parser.add_argument('--results-dir',default='runs/original/results')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0,str(root/'src'))
    from project_paths import guard_output
    guard_output(root/args.output,root)
    guard_output(root/args.results_dir,root)
    if (root/args.output).exists() or (root/args.results_dir/'metrics.json').exists():
        raise RuntimeError('Existing reproduction outputs must be preserved; choose new paths.')
    source = Path(args.source)
    raw = source.read_bytes()
    source_blob=hashlib.sha1(('blob %d\0'%len(raw)).encode()+raw).hexdigest()
    if source_blob!='2585e90aba6c964101b55e642a62279ea3318ece':
        raise ValueError('Source notebook differs from the audited GitHub blob. Use the pinned source commit.')
    notebook = nbformat.read(source, as_version=4)
    original_code = [c.source for c in notebook.cells if c.cell_type == 'code']
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
        cell.metadata.pop('execution', None)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        "import torch\n# Fix CPU threading for a recorded, reproducible audit environment.\ntorch.set_num_threads(1)"))
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'for iter in range(iters+1):' in cell.source:
            cell.source = 'audit_history = []\n' + cell.source.replace(
                '    loss.backward()',
                '    if iter % 100 == 0:\n'
                '        audit_history.append([iter, float(loss_data.detach()), float(loss_phys.detach()),\n'
                '            float(loss_ic.detach()), float(loss.detach()),\n'
                '            *[float(x.detach()) for x in rates()]])\n'
                '    loss.backward()')
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        '## Step 1 reproduction audit\n\nDiagnostics below use the same final trained model. '
        'They do not change the equations or training objective. Grid diagnostics '
        'are numerical checks, not proofs over continuous time.'))
    notebook.cells.append(nbformat.v4.new_code_cell(DIAGNOSTICS.replace("Path('audit/results')",'Path('+repr(args.results_dir)+')')))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    started=time.time()
    # All source cells contain ordinary Python. Execute in order in one new process.
    # This works even when notebook kernel sockets are unavailable in a container.
    os.environ['MPLBACKEND']='Agg'
    namespace={'__name__':'__main__'}
    count=0
    try:
        for cell_index,cell in enumerate(notebook.cells):
            if cell.cell_type!='code':continue
            count+=1;cell.execution_count=count
            stdout,stderr=io.StringIO(),io.StringIO()
            print(f'Executing code cell {count} (notebook index {cell_index})',flush=True)
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                exec(compile(cell.source,f'<notebook-cell-{cell_index}>','exec'),namespace)
            if stdout.getvalue():cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=stdout.getvalue()))
            if stderr.getvalue():cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=stderr.getvalue()))
            if 'plt' in namespace:
                plt=namespace['plt']
                for number in plt.get_fignums():
                    buffer=io.BytesIO();plt.figure(number).savefig(buffer,format='png',dpi=140)
                    cell.outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode()},metadata={}))
                plt.close('all')
    finally:
        notebook.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
        nbformat.write(notebook,out)
    metrics_path=root/args.results_dir/'metrics.json'
    metrics=json.loads(metrics_path.read_text())
    metrics['source_commit']='3c6275393c7f2aea437683c0e077996e6bff5d45'
    metrics['source_notebook_sha256']=hashlib.sha256(raw).hexdigest()
    metrics['source_blob_sha1']=source_blob
    metrics['source_code_sha256']=hashlib.sha256('\n\n'.join(original_code).encode()).hexdigest()
    metrics['source_notebook_unchanged']=source.read_bytes()==raw
    metrics['runtime_seconds']=time.time()-started
    metrics['execution_method']='Plain Python code cells executed sequentially in a fresh Python process; outputs saved as a notebook. No prior interactive state.'
    metrics['instrumentation']='One CPU thread; detached logging every 100 steps; appended diagnostics. Training mathematics unchanged.'
    metrics_path.write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    main()
