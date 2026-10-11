# Selected source code

Exact source lines; omitted portions are identified by each range. Notebook cell indices are zero-based.

## notebooks/sird-dinn.ipynb (cell 15, lines 1-8)

SHA-256: `3db707e85188125567bf91d3b1d1aed860ec37029142bc6cb5ad36da4c572e20`

```python
for iter in range(iters+1):
    opt.zero_grad()  # clears last step's gradient
    loss_data = mse(model(tau_train), y_obs)
    loss_ic = mse(model(tau0,), y0)
    loss_phys = (physics_residual()**2).mean()
    loss = W_data*loss_data + W_phy*loss_phys + W_ic*loss_ic
    loss.backward()
    opt.step()
```

## src/sird_validation.py (lines 53-59)

SHA-256: `8bb46c3e78678c621c9d3354a32a839649219c406dc77a0037b365cdaa99ae11`

```python
class ConstrainedDINN(nn.Module):
    """Convex mixture of the exact initial state and a neural simplex output.

    g(t)=1-exp(-t/scale); y(t)=(1-g(t))*y0+g(t)*softmax(logits(t)).
    For t>=0, all states are nonnegative, sum to one, and y(0)=y0.
    Only these three properties are hard constraints. The ODE is a soft penalty.
    """
```

## src/sird_validation.py (lines 71-86)

SHA-256: `8bb46c3e78678c621c9d3354a32a839649219c406dc77a0037b365cdaa99ae11`

```python
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
```

## src/two_region_sird.py (lines 20-43)

SHA-256: `d242875312fee7f30293972d43f23203cff932b22bc408258226b031d92a2e67`

```python
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
```

## src/two_region_sird.py (lines 115-129)

SHA-256: `d242875312fee7f30293972d43f23203cff932b22bc408258226b031d92a2e67`

```python
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
```
