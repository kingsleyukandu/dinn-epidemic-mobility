"""Check the new state transform and its physical-time derivatives."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sird_validation import Configuration,ConstrainedDINN,Y0,TRUE,simulate

torch.set_num_threads(1)
for seed in [0,1,2]:
    torch.manual_seed(seed)
    model=ConstrainedDINN(Configuration(),[.8,.1,.15])
    tau=torch.linspace(0,2,1001,dtype=torch.float64).reshape(-1,1).requires_grad_(True)
    y=model(tau)
    assert float(y.min().detach())>=0
    assert float(y.max().detach())<=1
    assert torch.max(torch.abs(y.sum(dim=1)-1)).item()<5e-15
    assert np.array_equal(y[0].detach().numpy(),Y0)
    point=torch.tensor([[.4]],dtype=torch.float64,requires_grad=True)
    state=model(point)
    derivative=torch.cat([torch.autograd.grad(state[:,j:j+1],point,
        torch.ones_like(state[:,j:j+1]),retain_graph=True)[0] for j in range(4)],dim=1)/160
    eps=1e-4
    physical_difference=(model(point+eps/160)-model(point-eps/160))/(2*eps)
    assert torch.max(torch.abs(derivative-physical_difference)).item()<1e-9
reference=simulate(np.linspace(0,160,641),TRUE)
assert np.max(np.abs(reference.sum(axis=1)-1))<5e-12
assert reference.min()>-1e-12
print('PASS: simplex bounds, population conservation, exact initial state, time derivative scaling, and ODE reference balance.')
