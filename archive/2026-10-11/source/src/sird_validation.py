"""Constrained SIRD DINN and conventional ODE calibration for controlled validation.

The original notebook is preserved. This module is a separate implementation.
Known synthetic truth is used only for data generation and final evaluation.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import torch
from torch import nn

TRUE = np.array([0.5, 0.2, 1/14], dtype=float)
Y0 = np.array([1-1e-6, 1e-6, 0., 0.], dtype=float)
NAMES = ['beta','gamma','omega']


def rhs(t,y,p):
    s,i,r,d=y;beta,gamma,omega=p
    return [-beta*s*i, beta*s*i-(gamma+omega)*i, gamma*i, omega*i]


def simulate(times,parameters,y0=Y0):
    times=np.asarray(times,dtype=float)
    if times[0]<0 or np.any(np.diff(times)<=0):
        raise ValueError('Times must be strictly increasing and nonnegative.')
    if times[-1]==0:return np.asarray(y0)[None,:]
    solution=solve_ivp(lambda t,y:rhs(t,y,parameters),(0,float(times[-1])),y0,
        t_eval=times,method='DOP853',rtol=1e-10,atol=1e-12)
    if not solution.success:raise RuntimeError(solution.message)
    return solution.y.T


@dataclass(frozen=True)
class Configuration:
    horizon:float=160.
    width:int=32
    depth:int=3
    gate_scale_days:float=3.2
    adam_steps:int=8000
    lbfgs_steps:int=600
    learning_rate:float=1e-3
    physics_weight:float=1e-4
    collocation_points:int=161
    dtype:str='float64'


class ConstrainedDINN(nn.Module):
    """Convex mixture of the exact initial state and a neural simplex output.

    g(t)=1-exp(-t/scale); y(t)=(1-g(t))*y0+g(t)*softmax(logits(t)).
    For t>=0, all states are nonnegative, sum to one, and y(0)=y0.
    Only these three properties are hard constraints. The ODE is a soft penalty.
    """
    def __init__(self,config,initial_rates):
        super().__init__();self.config=config
        layers=[nn.Linear(1,config.width),nn.Tanh()]
        for _ in range(config.depth-1):layers.extend([nn.Linear(config.width,config.width),nn.Tanh()])
        layers.append(nn.Linear(config.width,4))
        self.net=nn.Sequential(*layers).double()
        # A fixed generic simplex bias; does not use synthetic true rates.
        with torch.no_grad():self.net[-1].bias.copy_(torch.tensor([0.,-5.,-3.,-4.],dtype=torch.float64))
        self.register_buffer('initial_state',torch.tensor(Y0,dtype=torch.float64))
        self.log_rates=nn.Parameter(torch.log(torch.tensor(initial_rates,dtype=torch.float64)))

    def forward(self,tau):
        if torch.any(tau<0):raise ValueError('The model is defined here for nonnegative time.')
        gate=-torch.expm1(-tau*self.config.horizon/self.config.gate_scale_days)
        return (1-gate)*self.initial_state+gate*torch.softmax(self.net(tau),dim=1)

    def rates(self):return self.log_rates.exp()


def residual(model,tau):
    u=model(tau)
    derivatives=torch.cat([torch.autograd.grad(u[:,j:j+1],tau,
        grad_outputs=torch.ones_like(u[:,j:j+1]),create_graph=True)[0] for j in range(4)],dim=1)
    s,i,r,d=[u[:,j:j+1] for j in range(4)]
    beta,gamma,omega=model.rates()
    f=torch.cat([-beta*s*i,beta*s*i-(gamma+omega)*i,gamma*i,omega*i],dim=1)
    return derivatives-model.config.horizon*f


def losses(model,tau_obs,observations,mask,tau_coll):
    prediction=model(tau_obs)
    data=((prediction-observations)[mask]**2).mean()
    equation=(residual(model,tau_coll)**2).mean()
    return data+model.config.physics_weight*equation,data,equation


def train(times,observations,initial_rates,seed,config=Configuration(),mask=None,output=None):
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=ConstrainedDINN(config,initial_rates)
    tau_obs=torch.tensor(np.asarray(times)/config.horizon,dtype=torch.float64).reshape(-1,1)
    y_obs=torch.tensor(observations,dtype=torch.float64)
    if mask is None:mask=np.ones_like(observations,dtype=bool)
    mask=torch.tensor(mask,dtype=torch.bool)
    tau_coll=torch.linspace(0,1,config.collocation_points,dtype=torch.float64).reshape(-1,1).requires_grad_(True)
    history=[];started=time.perf_counter()
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    for step in range(config.adam_steps):
        optimizer.zero_grad();tau_coll.grad=None
        loss,data,equation=losses(model,tau_obs,y_obs,mask,tau_coll)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();optimizer.step()
        if step%200==0 or step==config.adam_steps-1:
            history.append({'stage':'adam','step':step,'total':float(loss.detach()),
                'data':float(data.detach()),'equations_tau':float(equation.detach()),
                **dict(zip(NAMES,model.rates().detach().numpy().tolist()))})
        if step%2000==0:print(f'seed {seed} Adam {step}: loss={float(loss.detach()):.4g}',flush=True)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=config.lbfgs_steps,
        max_eval=config.lbfgs_steps*2,history_size=50,line_search_fn='strong_wolfe',
        tolerance_grad=1e-12,tolerance_change=1e-14)
    evaluations=0
    def closure():
        nonlocal evaluations
        optimizer.zero_grad();tau_coll.grad=None
        loss,data,equation=losses(model,tau_obs,y_obs,mask,tau_coll)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite L-BFGS loss')
        loss.backward();evaluations+=1
        if evaluations%20==0:
            history.append({'stage':'lbfgs','step':evaluations,'total':float(loss.detach()),
                'data':float(data.detach()),'equations_tau':float(equation.detach()),
                **dict(zip(NAMES,model.rates().detach().numpy().tolist()))})
        return loss
    if config.lbfgs_steps:optimizer.step(closure)
    final,data,equation=losses(model,tau_obs,y_obs,mask,tau_coll)
    training={'config':asdict(config),'seed':seed,'initial_rates':list(initial_rates),
        'runtime_seconds':time.perf_counter()-started,'lbfgs_closure_evaluations':evaluations,
        'final_total_loss':float(final.detach()),'final_data_mse':float(data.detach()),
        'final_equation_mse_tau':float(equation.detach()),'observed_scalar_values':int(mask.sum())}
    if output:
        folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
        (folder/'training.json').write_text(json.dumps(training,indent=2))
        (folder/'loss_history.json').write_text(json.dumps(history,indent=2))
        torch.save(model.state_dict(),folder/'weights.pt')
    return model,training,history


def diagnose(model,times,truth_rates=TRUE):
    tau=torch.tensor(np.asarray(times)/model.config.horizon,dtype=torch.float64).reshape(-1,1).requires_grad_(True)
    states=model(tau).detach().numpy()
    res=(residual(model,tau)/model.config.horizon).detach().numpy()
    rates=model.rates().detach().numpy()
    reference=simulate(times,truth_rates)
    with torch.no_grad():initial=model(torch.zeros((1,1),dtype=torch.float64)).numpy()[0]
    metrics={'parameters':dict(zip(NAMES,rates.tolist())),
        'relative_errors_percent':dict(zip(NAMES,(100*np.abs(rates-truth_rates)/truth_rates).tolist())),
        'trajectory_rmse':float(np.sqrt(np.mean((states-reference)**2))),
        'residual_rmse_per_day':float(np.sqrt(np.mean(res**2))),
        'residual_max_abs_per_day':float(np.max(np.abs(res))),
        'conservation_max_abs_error':float(np.max(np.abs(states.sum(axis=1)-1))),
        'initial_max_abs_error':float(np.max(np.abs(initial-Y0))),
        'minimum_compartment':float(states.min()),'negative_entries':int(np.sum(states<0)),
        'maximum_compartment':float(states.max())}
    return metrics,states,reference,res


def fit_ode(times,observations,initial_rates,mask=None):
    if mask is None:mask=np.ones_like(observations,dtype=bool)
    started=time.perf_counter();calls=0
    def objective(log_rates):
        nonlocal calls
        calls+=1
        return (simulate(times,np.exp(log_rates))-observations)[mask]
    result=least_squares(objective,np.log(initial_rates),bounds=(np.log([1e-4]*3),np.log([3.]*3)),
        xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=1000,diff_step=1e-4)
    rates=np.exp(result.x)
    return {'parameters':dict(zip(NAMES,rates.tolist())),
        'relative_errors_percent':dict(zip(NAMES,(100*np.abs(rates-TRUE)/TRUE).tolist())),
        'runtime_seconds':time.perf_counter()-started,'optimizer_success':bool(result.success),
        'optimizer_message':result.message,'objective_calls':calls,'fit_mse':float(np.mean(result.fun**2)),
        'jacobian_singular_values':np.linalg.svd(result.jac,compute_uv=False).tolist()}


def save_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
