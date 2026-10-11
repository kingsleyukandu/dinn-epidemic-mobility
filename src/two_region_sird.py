"""Two-region SIRD with conservative directional movement of living states.

States are fractions of combined initial population, ordered S1,I1,R1,D1,S2,I2,R2,D2.
Incidence in region j is beta_j*S_j*I_j/p_j, where p_j is its initial population
fraction. These fixed reference denominators preserve the original SIRD model in
the zero-movement limit. They are not current regional population denominators.
One directional movement rate applies equally to S, I, and R. D does not move.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import torch
from torch import nn

STATE_NAMES = ['S1', 'I1', 'R1', 'D1', 'S2', 'I2', 'R2', 'D2']
RATE_NAMES = ['beta1', 'gamma1', 'omega1', 'beta2', 'gamma2', 'omega2', 'tau12', 'tau21']
REFERENCE = np.array([.6, .4])
INITIAL_STATE = np.array([.6-1e-5, 1e-5, 0., 0., .4-1e-6, 1e-6, 0., 0.])
SYNTHETIC_RATES = np.array([.5, .2, 1/14, .38, .16, .035, .015, .010])
INITIAL_RATES = np.array([.7, .12, .1, .5, .10, .06, .02, .02])


def movement_matrix(tau12, tau21):
    """Column-vector convention: entry (destination, origin)."""
    return np.array([[-tau12, tau21], [tau12, -tau21]])


def rhs_numpy(t, state, rates, reference=REFERENCE):
    x = np.asarray(state).reshape(2, 4)
    p = np.asarray(rates)
    beta, gamma, omega = p[[0, 3]], p[[1, 4]], p[[2, 5]]
    infection = beta * x[:, 0] * x[:, 1] / reference
    f = np.column_stack([-infection, infection-(gamma+omega)*x[:, 1],
                         gamma*x[:, 1], omega*x[:, 1]])
    flow = p[7]*x[1, :3] - p[6]*x[0, :3]
    f[0, :3] += flow
    f[1, :3] -= flow
    return f.ravel()


def rhs_torch(state, rates, reference):
    x = state.reshape(-1, 2, 4)
    beta = rates[[0, 3]]; gamma = rates[[1, 4]]; omega = rates[[2, 5]]
    infection = beta*x[:, :, 0]*x[:, :, 1]/reference
    f = torch.stack([-infection, infection-(gamma+omega)*x[:, :, 1],
                     gamma*x[:, :, 1], omega*x[:, :, 1]], dim=2)
    flow = rates[7]*x[:, 1, :3] - rates[6]*x[:, 0, :3]
    flow = torch.cat([flow, torch.zeros_like(x[:, 0, 3:4])], dim=1)
    movement = torch.stack([flow, -flow], dim=1)
    return (f+movement).reshape(-1, 8)


def simulate(times, rates, initial_state=INITIAL_STATE, reference=REFERENCE, method='DOP853'):
    times = np.asarray(times, dtype=float)
    rates = np.asarray(rates, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Times must be a finite, nonempty, strictly increasing, nonnegative vector.')
    if rates.shape != (8,) or not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError('Eight finite nonnegative rates are required.')
    if initial_state.shape != (8,) or not np.isfinite(initial_state).all() or np.any(initial_state < 0):
        raise ValueError('Eight finite nonnegative initial states are required.')
    if reference.shape != (2,) or not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError('Two finite positive reference populations are required.')
    if times[-1] == 0:
        return initial_state[None, :].copy()
    sol = solve_ivp(lambda t, y: rhs_numpy(t, y, rates, reference), (0., float(times[-1])),
                    initial_state, t_eval=times, method=method, rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


@dataclass(frozen=True)
class Configuration:
    horizon: float = 160.
    width: int = 48
    depth: int = 3
    gate_scale_days: float = 3.2
    adam_steps: int = 8000
    lbfgs_steps: int = 1000
    learning_rate: float = 1e-3
    physics_weight: float = 1e-4
    collocation_points: int = 161


class TwoRegionDINN(nn.Module):
    """One simplex across both regions: local populations are not fixed."""
    def __init__(self, config=Configuration(), initial_rates=INITIAL_RATES,
                 initial_state=INITIAL_STATE, reference=REFERENCE):
        super().__init__()
        self.config = config
        if np.any(np.asarray(initial_rates) <= 0):
            raise ValueError('Trainable rate initialization must be positive.')
        if not np.isclose(np.sum(initial_state), 1., atol=1e-14, rtol=0) or np.any(np.asarray(initial_state) < 0):
            raise ValueError('The constrained network requires a nonnegative initial state summing to one.')
        layers = [nn.Linear(1, config.width), nn.Tanh()]
        for _ in range(config.depth-1):
            layers.extend([nn.Linear(config.width, config.width), nn.Tanh()])
        layers.append(nn.Linear(config.width, 8))
        self.net = nn.Sequential(*layers).double()
        # Fixed generic compartment biases, independent of synthetic true rates.
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0., -5., -3., -4., -.4, -5.4, -3.4, -4.4], dtype=torch.float64))
        self.register_buffer('initial_state', torch.tensor(initial_state, dtype=torch.float64))
        self.register_buffer('reference', torch.tensor(reference, dtype=torch.float64))
        self.log_rates = nn.Parameter(torch.log(torch.tensor(initial_rates, dtype=torch.float64)))

    def forward(self, tau):
        if torch.any(tau < 0):
            raise ValueError('This model is defined for nonnegative time.')
        gate = -torch.expm1(-tau*self.config.horizon/self.config.gate_scale_days)
        return (1-gate)*self.initial_state + gate*torch.softmax(self.net(tau), dim=1)

    def rates(self):
        return self.log_rates.exp()


def residual(model, tau):
    states = model(tau)
    derivative = torch.cat([torch.autograd.grad(states[:, j:j+1], tau,
        grad_outputs=torch.ones_like(states[:, j:j+1]), create_graph=True)[0] for j in range(8)], dim=1)
    return derivative - model.config.horizon*rhs_torch(states, model.rates(), model.reference)


def losses(model, t_obs, observations, t_coll):
    data = ((model(t_obs)-observations)**2).mean()
    equations = (residual(model, t_coll)**2).mean()
    return data+model.config.physics_weight*equations, data, equations


def train(times, observations, seed, config=Configuration(), initial_rates=INITIAL_RATES, output=None):
    times = np.asarray(times)
    observations = np.asarray(observations)
    if observations.shape != (len(times), 8) or not np.isfinite(observations).all():
        raise ValueError('Finite observations of all eight states are required.')
    if times.min() < 0 or times.max() > config.horizon:
        raise ValueError('Observation times must lie within the training interval.')
    torch.set_num_threads(1); torch.manual_seed(seed)
    model = TwoRegionDINN(config, initial_rates)
    t_obs = torch.tensor(times/config.horizon, dtype=torch.float64).reshape(-1, 1)
    observations = torch.tensor(observations, dtype=torch.float64)
    t_coll = torch.linspace(0, 1, config.collocation_points, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []; start = time.perf_counter()
    for step in range(config.adam_steps):
        optimizer.zero_grad(); t_coll.grad = None
        loss, data, equation = losses(model, t_obs, observations, t_coll)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite Adam loss')
        if step % 200 == 0 or step == config.adam_steps-1:
            history.append({'stage': 'adam', 'step': step, 'total': float(loss.detach()),
                'data': float(data.detach()), 'equations_tau': float(equation.detach()),
                **dict(zip(RATE_NAMES, model.rates().detach().numpy().tolist()))})
        loss.backward(); optimizer.step()
        if step % 2000 == 0:
            print(f'seed {seed} Adam {step}: objective={float(loss.detach()):.5g}', flush=True)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=config.lbfgs_steps,
        max_eval=2*config.lbfgs_steps, history_size=50, line_search_fn='strong_wolfe',
        tolerance_grad=1e-12, tolerance_change=1e-14)
    evaluations = 0
    def closure():
        nonlocal evaluations
        optimizer.zero_grad(); t_coll.grad = None
        loss, data, equation = losses(model, t_obs, observations, t_coll)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite L-BFGS loss')
        loss.backward(); evaluations += 1
        if evaluations % 20 == 0:
            history.append({'stage': 'lbfgs', 'step': evaluations, 'total': float(loss.detach()),
                'data': float(data.detach()), 'equations_tau': float(equation.detach()),
                **dict(zip(RATE_NAMES, model.rates().detach().numpy().tolist()))})
        return loss
    if config.lbfgs_steps:
        optimizer.step(closure)
    final, data, equation = losses(model, t_obs, observations, t_coll)
    # Report the optimizer's achieved budget; do not assert numerical convergence.
    state = optimizer.state.get(next(iter(model.parameters())), {})
    training = {'seed': seed, 'configuration': asdict(config), 'initial_rates': np.asarray(initial_rates).tolist(),
        'runtime_seconds': time.perf_counter()-start, 'lbfgs_closure_evaluations': evaluations,
        'lbfgs_iterations': int(state.get('n_iter', 0)), 'final_total_loss': float(final.detach()),
        'final_data_mse': float(data.detach()), 'final_equation_mse_tau': float(equation.detach()),
        'observed_scalar_values': observations.numel()}
    if output is not None:
        folder = Path(output); folder.mkdir(parents=True, exist_ok=True)
        (folder/'training.json').write_text(json.dumps(training, indent=2))
        (folder/'loss_history.json').write_text(json.dumps(history, indent=2))
        torch.save(model.state_dict(), folder/'weights.pt')
    return model, training, history


def diagnose(model, times, true_rates=SYNTHETIC_RATES):
    tau = torch.tensor(np.asarray(times)/model.config.horizon, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    states = model(tau).detach().numpy()
    rates = model.rates().detach().numpy()
    res = (residual(model, tau)/model.config.horizon).detach().numpy()
    ref = simulate(times, true_rates)
    initial = model(torch.zeros((1, 1), dtype=torch.float64)).detach().numpy()[0]
    physical_rhs = np.array([rhs_numpy(t, s, rates) for t, s in zip(times, states)])
    derivatives = res+physical_rhs
    metrics = {'parameters': dict(zip(RATE_NAMES, rates.tolist())),
        'relative_errors_percent': dict(zip(RATE_NAMES, (100*np.abs(rates-true_rates)/true_rates).tolist())),
        'trajectory_rmse': float(np.sqrt(np.mean((states-ref)**2))),
        'rmse_by_state': dict(zip(STATE_NAMES, np.sqrt(np.mean((states-ref)**2, axis=0)).tolist())),
        'residual_rmse_per_day': float(np.sqrt(np.mean(res**2))),
        'residual_max_abs_per_day': float(np.max(np.abs(res))),
        'conservation_max_abs_error': float(np.max(np.abs(states.sum(axis=1)-1))),
        'initial_max_abs_error': float(np.max(np.abs(initial-INITIAL_STATE))),
        'minimum_compartment': float(states.min()), 'negative_entries': int((states < 0).sum()),
        'minimum_D_derivative_per_day': float(derivatives[:, [3, 7]].min()),
        'negative_D_derivative_entries': int((derivatives[:, [3, 7]] < 0).sum()),
        'regional_balance_residual_max_abs_per_day': float(np.max(np.abs(res.reshape(-1, 2, 4).sum(axis=2))))}
    return metrics, states, ref, res


def fit_ode(times, observations, initial_rates=INITIAL_RATES):
    start = time.perf_counter(); calls = 0
    def objective(log_rates):
        nonlocal calls
        calls += 1
        return (simulate(times, np.exp(log_rates))-observations).ravel()
    lower = [1e-4]*6+[1e-5]*2
    upper = [3.]*6+[.5]*2
    result = least_squares(objective, np.log(initial_rates), bounds=(np.log(lower), np.log(upper)),
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1500, diff_step=1e-4)
    rates = np.exp(result.x)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    return {'parameters': dict(zip(RATE_NAMES, rates.tolist())),
        'relative_errors_percent': dict(zip(RATE_NAMES, (100*np.abs(rates-SYNTHETIC_RATES)/SYNTHETIC_RATES).tolist())),
        'runtime_seconds': time.perf_counter()-start, 'optimizer_success': bool(result.success),
        'optimizer_message': result.message, 'objective_calls': calls, 'fit_mse': float(np.mean(result.fun**2)),
        'jacobian_singular_values': singular.tolist(),
        'log_rate_jacobian_condition_number': float(singular[0]/singular[-1]),
        'rate_bounds_lower': lower, 'rate_bounds_upper': upper}
