"""Model identities and independent limiting-case checks for two-region SIRD."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import two_region_sird as m
from sird_validation import simulate as single_simulate, rhs as single_rhs

from project_paths import guard_output
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('runs/checks/two_region'))
args=parser.parse_args()
OUT=guard_output(args.output if args.output.is_absolute() else ROOT/args.output,ROOT)
OUT.mkdir(parents=True, exist_ok=True)
results = {}
rng = np.random.default_rng(314159)
states = rng.dirichlet(np.ones(8), size=200)
rhs = np.array([m.rhs_numpy(0, y, m.SYNTHETIC_RATES) for y in states])
global_balance = float(np.max(np.abs(rhs.sum(axis=1))))
assert global_balance < 1e-14
flow = m.SYNTHETIC_RATES[7]*states[:, [4,5,6]].sum(axis=1)-m.SYNTHETIC_RATES[6]*states[:, [0,1,2]].sum(axis=1)
local_balance = float(np.max(np.abs(rhs[:, :4].sum(axis=1)-flow)))
assert local_balance < 1e-14
results['population_balance'] = {'global_rhs_sum_max_abs': global_balance,
    'regional_rhs_balance_max_abs': local_balance}

minimum_boundary_rhs = 1.
for y in states:
    for j in range(8):
        boundary = y.copy(); boundary[j] = 0; boundary /= boundary.sum()
        minimum_boundary_rhs = min(minimum_boundary_rhs, m.rhs_numpy(0, boundary, m.SYNTHETIC_RATES)[j])
assert minimum_boundary_rhs >= -1e-14
results['boundary_nonnegativity'] = {'tested_boundary_states': 1600, 'minimum_derivative': minimum_boundary_rhs}

generator = m.movement_matrix(*m.SYNTHETIC_RATES[6:])
assert np.max(np.abs(generator.sum(axis=0))) < 1e-14
assert generator[0,1] >= 0 and generator[1,0] >= 0
for j,rate in [(3,2),(7,5)]:
    assert np.max(np.abs(rhs[:,j]-m.SYNTHETIC_RATES[rate]*states[:,j-2])) < 1e-14
results['mobility_convention'] = {'matrix': generator.tolist(), 'deceased_compartments_have_no_movement_terms': True}

t = np.linspace(0, 160, 641)
rates = m.SYNTHETIC_RATES.copy(); rates[6:] = 0
zero = m.simulate(t, rates)
independent_rhs = np.array([np.concatenate([m.REFERENCE[j]*np.asarray(single_rhs(0,
    y[4*j:4*j+4]/m.REFERENCE[j], rates[3*j:3*j+3])) for j in [0,1]]) for y in states])
zero_rhs_error = float(np.max(np.abs(independent_rhs-np.array([m.rhs_numpy(0,y,rates) for y in states]))))
assert zero_rhs_error < 1e-14
independent = np.column_stack([m.REFERENCE[j]*single_simulate(t, rates[3*j:3*j+3],
    y0=m.INITIAL_STATE[4*j:4*j+4]/m.REFERENCE[j]) for j in [0,1]])
error = float(np.max(np.abs(zero-independent)))
assert error < 1e-8
results['zero_mobility'] = {'max_abs_rhs_difference_from_original_SIRD': zero_rhs_error,
    'max_abs_trajectory_difference_from_independent_original_SIRD': error}

transport_rates = np.zeros(8); transport_rates[6:] = m.SYNTHETIC_RATES[6:]
initial = np.array([.55,.03,.01,.01,.30,.02,.07,.01])
transport = m.simulate(t, transport_rates, initial_state=initial)
analytic = []
for day in t:
    x = initial.reshape(2,4).copy()
    x[:,:3] = expm(generator*day) @ x[:,:3]
    analytic.append(x.ravel())
transport_error = float(np.max(np.abs(transport-np.asarray(analytic))))
assert transport_error < 3e-10
assert np.array_equal(transport[:,[3,7]], np.tile(initial[[3,7]], (len(t),1)))
results['transport_only'] = {'max_abs_difference_from_matrix_exponential': transport_error,
    'deceased_counts_constant': True}

sym_ref = np.array([.5,.5]); sym_y = np.tile(np.array([.499995,.000005,0.,0.]),2)
sym_rates = np.array([.5,.2,1/14,.5,.2,1/14,.005,.005])
sym = m.simulate(t, sym_rates, initial_state=sym_y, reference=sym_ref)
sym_fast = sym_rates.copy(); sym_fast[6:] = .03
sym_other = m.simulate(t, sym_fast, initial_state=sym_y, reference=sym_ref)
single = single_simulate(t, sym_rates[:3], y0=np.array([.99999,.00001,0.,0.]))
sym_error = float(np.max(np.abs(sym[:,:4]+sym[:,4:]-single)))
unidentifiable = float(np.max(np.abs(sym-sym_other)))
assert sym_error < 1e-8 and unidentifiable < 1e-8
results['symmetric_limit'] = {'max_abs_aggregate_difference_from_single_SIRD': sym_error,
    'max_abs_difference_with_sixfold_common_movement_rate': unidentifiable,
    'interpretation': 'Equal region sizes and identical epidemic states with symmetric flows leave the common movement scale unidentifiable.'}

seed_y = m.INITIAL_STATE.copy(); seed_y[4] += seed_y[5]; seed_y[5] = 0
seeded = m.simulate(np.array([0.,1.,5.]), m.SYNTHETIC_RATES, initial_state=seed_y)
isolated = m.simulate(np.array([0.,1.,5.]), rates, initial_state=seed_y)
assert seeded[1,5] > 0 and isolated[:,5].max() == 0
results['infection_transport'] = {'region2_initial_infected': 0.,
    'region2_infected_day1_with_movement': float(seeded[1,5]),
    'region2_infected_day5_without_movement': float(isolated[-1,5])}

reference = m.simulate(t, m.SYNTHETIC_RATES)
alternate = m.simulate(t, m.SYNTHETIC_RATES, method='Radau')
solver_error = float(np.max(np.abs(reference-alternate)))
assert solver_error < 1e-8
assert reference.min() > -1e-12 and np.max(np.abs(reference.sum(axis=1)-1)) < 1e-11
assert np.min(np.diff(reference[:,[3,7]],axis=0)) >= -1e-12
results['independent_solver'] = {'max_abs_DOP853_Radau_difference': solver_error,
    'reference_global_balance_max_abs': float(np.max(np.abs(reference.sum(axis=1)-1))),
    'reference_region1_population_range': [float(reference[:,:4].sum(axis=1).min()),float(reference[:,:4].sum(axis=1).max())]}

torch.set_num_threads(1)
trhs = m.rhs_torch(torch.tensor(states), torch.tensor(m.SYNTHETIC_RATES), torch.tensor(m.REFERENCE)).numpy()
rhs_error = float(np.max(np.abs(trhs-rhs)))
assert rhs_error < 1e-14
derivative_errors = []
for seed in [0,1,2]:
    torch.manual_seed(seed)
    model = m.TwoRegionDINN()
    tau = torch.linspace(0,2,1001,dtype=torch.float64).reshape(-1,1)
    y = model(tau).detach().numpy()
    assert y.min() >= 0 and y.max() <= 1
    assert np.max(np.abs(y.sum(axis=1)-1)) < 5e-15
    assert np.array_equal(y[0],m.INITIAL_STATE)
    point = torch.tensor([[.4]],dtype=torch.float64,requires_grad=True)
    u = model(point)
    deriv = torch.cat([torch.autograd.grad(u[:,j:j+1],point,
        torch.ones_like(u[:,j:j+1]),retain_graph=True)[0] for j in range(8)],dim=1)/model.config.horizon
    eps = 1e-4
    numerical = (model(point+eps/model.config.horizon)-model(point-eps/model.config.horizon))/(2*eps)
    derivative_errors.append(float(torch.max(torch.abs(deriv-numerical)).detach()))
    assert derivative_errors[-1] < 1e-9
results['neural_implementation'] = {'numpy_torch_rhs_max_abs_difference': rhs_error,
    'derivative_scaling_max_abs_difference': max(derivative_errors),
    'three_initializations_pass_simplex_and_exact_IC_checks': True}

(OUT/'checks.json').write_text(json.dumps({'status':'PASS',
    'numerical_trajectory_tolerance_population_fraction':1e-8,'results':results},indent=2))
print(json.dumps({'status':'PASS','results':results},indent=2))
