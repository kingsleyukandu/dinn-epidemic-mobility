"""Build and sequentially execute a notebook displaying the recorded validation.

Execution uses a fresh ordinary Python process, not a Jupyter kernel. Default
cells load a recorded checkpoint and recompute diagnostics; RETRAIN can be enabled
to perform an independent clean seed-0 fit. All nine validation runs are displayed.
"""
from contextlib import redirect_stdout, redirect_stderr
import io
import base64
from pathlib import Path
import os
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
nb=nbformat.v4.new_notebook()
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12.14'},
    'execution_note':'Ordinary Python cells executed sequentially in a fresh Python process; default mode loads a recorded checkpoint, without retraining.'}
md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
nb.cells=[md(r'''# Two-region SIRD DINN with directional population movement

Kingsley Chimuanya Ukandu | Step 3 | October 11, 2026

This notebook implements the two-region stage through `src/two_region_sird.py` and
displays the recorded synthetic validation. All eight states and the initial state
are known. There are six disease rates and two directional movement rates.

States are fractions of the **combined initial population**. Initial regional
reference populations are $p_1=0.6$, $p_2=0.4$. Incidence is
$\beta_jS_jI_j/p_j$ with fixed reference denominators. Susceptible, infected, and
recovered states move; deceased states stay in the region where recorded.

The state transform enforces nonnegative states, combined population balance, and
the exact initial state. Equations are a soft residual penalty. Deceased monotonicity
is not a hard neural constraint. The experiment does not use real surveillance or
mobility data and does not establish method superiority.'''),
code(r'''from pathlib import Path
import sys, json
import numpy as np
import torch
import matplotlib.pyplot as plt
ROOT = Path.cwd()
if not (ROOT / 'src').is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from two_region_sird import (Configuration, TwoRegionDINN, simulate, train, diagnose,
    STATE_NAMES, RATE_NAMES, SYNTHETIC_RATES)
torch.set_num_threads(1)
RESULTS = ROOT / 'validation' / 'step3' / 'results'
protocol = json.loads((RESULTS / 'protocol.json').read_text())
print('Configuration:', protocol['configuration'])
print('State units:', protocol['model']['state_units'])
print('Validation seeds:', protocol['seeds'])'''),
md('''## Load and verify an illustrative clean checkpoint

Seed 0 is used for the illustration; it was not selected by accuracy. Set
`RETRAIN = True` to reproduce a separate fit with the same fixed budget. The
default execution loads the recorded model and recomputes its diagnostics.'''),
code(r'''RETRAIN = False
times = np.arange(0., 160., 3.)
observations = simulate(times, SYNTHETIC_RATES)
if RETRAIN:
    model, training, history = train(times, observations, seed=0,
        output=ROOT / 'validation' / 'step3' / 'notebook_run')
else:
    folder = RESULTS / 'clean' / 'seed_0'
    training = json.loads((folder / 'training.json').read_text())
    model = TwoRegionDINN(Configuration(**training['configuration']), training['initial_rates'])
    model.load_state_dict(torch.load(folder / 'weights.pt', weights_only=True))
grid = np.linspace(0., 160., 641)
metrics, prediction, truth, residual = diagnose(model, grid)
print('Rate       true           inferred       relative error %')
for name, value in zip(RATE_NAMES, SYNTHETIC_RATES):
    print(f"{name:8s} {value:.9f}    {metrics['parameters'][name]:.9f}    {metrics['relative_errors_percent'][name]:.6f}")
for key in ['trajectory_rmse', 'negative_entries', 'conservation_max_abs_error',
            'initial_max_abs_error', 'residual_max_abs_per_day', 'minimum_D_derivative_per_day']:
    print(key, metrics[key])'''),
code(r'''fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for j, name in enumerate(STATE_NAMES):
    ax = axes.flat[j]
    ax.plot(grid, truth[:, j], 'k--', label='ODE reference')
    ax.plot(grid, prediction[:, j], color='#087f8c', label='DINN seed 0')
    ax.set_title(name)
    ax.set_xlabel('Day')
    ax.set_ylabel('Combined-population fraction')
axes[0, 0].legend(frameon=False)
fig.tight_layout()
plt.show()'''),
md('''## All nine runs and the ODE comparator

The reported ranges are descriptive, not confidence intervals. Each noisy or missing
condition uses one fixed dataset. Conventional calibration is a strong comparator in
these model-correct synthetic experiments.'''),
code(r'''summary = json.loads((RESULTS / 'neural_summary.json').read_text())
print('Data      seed  max disease error %  tau12 error %  tau21 error %  trajectory RMSE')
for row in summary:
    e = row['relative_errors_percent']
    disease = max(e[n] for n in RATE_NAMES[:6])
    print(f"{row['case']:8s}  {row['seed']}       {disease:.6f}         {e['tau12']:.6f}       {e['tau21']:.6f}       {row['trajectory_rmse']:.6g}")
print('\nODE calibration:')
for case in ['clean', 'noisy', 'missing']:
    ode = json.loads((RESULTS / case / 'ode_baseline.json').read_text())
    print(case, 'max rate error %', max(ode['relative_errors_percent'].values()),
        'trajectory RMSE', ode['trajectory_rmse'], 'converged', ode['optimizer_success'])'''),
md('''## Model checks and an identifiability limit

Zero movement reduces to the original SIRD system independently in each region.
Transport without disease agrees with a matrix exponential. Equal regions with
identical epidemics and symmetric movement have zero net movement; changing the
common movement scale then leaves all trajectories unchanged. This prevents its
identification in that setup.'''),
code(r'''checks = json.loads((RESULTS / 'model_checks.json').read_text())
print('Model checks:', checks['status'])
for name, values in checks['results'].items():
    print(name, values)
verification = json.loads((RESULTS / 'verification.json').read_text())
print('\nCheckpoint and dataset verification:', verification['status'])
print('Pilot vs fresh clean seed-0 metrics identical:',
      verification['pilot_vs_fresh_clean_seed0_metrics_identical'])'''),
md('''## Scope

This is a synthetic two-region mobility proof of implementation. Regional populations
are variable; the combined accounted population is conserved. Recovered counts in one
region can decline due to outward movement. Small neural decreases in deceased states
are approximation errors and are retained in diagnostics. A multipatch SIS model,
real-data observation models, uncertainty analysis, and practical forecasting remain
future work. See the Step 3 report and `validation/step3/REPRODUCE.md` for the fixed
protocol and reproduction instructions.''')]
os.chdir(ROOT)
namespace={'__name__':'__main__'}
execution_count=0
for cell in nb.cells:
    if cell.cell_type!='code':continue
    execution_count+=1
    stdout=io.StringIO();stderr=io.StringIO()
    with redirect_stdout(stdout),redirect_stderr(stderr):
        exec(compile(cell.source,'two-region-sird-dinn.ipynb','exec'),namespace)
    cell.execution_count=execution_count
    cell.outputs=[]
    if stdout.getvalue():cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=stdout.getvalue()))
    if stderr.getvalue():cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=stderr.getvalue()))
    for number in plt.get_fignums():
        buffer=io.BytesIO();plt.figure(number).savefig(buffer,format='png',dpi=120,bbox_inches='tight')
        cell.outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode()}))
        plt.close(number)
nbformat.validate(nb)
destination=ROOT/'notebooks/two-region-sird-dinn.ipynb'
nbformat.write(nb,destination)
print('Notebook built and sequentially executed:',destination)
