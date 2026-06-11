# Disease-Informed Neural Networks for Epidemic Mobility Models: From SIRD Parameter Discovery to SIS Patch Equilibria

## Project Overview

This project is a learning-to-research project focused on understanding how to customize **Physics-Informed Neural Networks (PINNs)** into **Disease-Informed Neural Networks (DINNs)** for epidemic models.

The goal is not simply to run a neural network on epidemic data. The goal is to learn how to design a neural network that respects the governing differential equations of a disease model while also using data to discover unknown disease parameters.

The project develops in three stages:

1. **Basic SIRD-DINN for parameter discovery**
   Build a neural network that learns the trajectory of a basic SIRD epidemic model and recovers hidden parameters such as transmission, recovery, and mortality rates.

2. **SIRD-DINN with population movement**
   Extend the basic SIRD model to two interacting regions where individuals can move between populations.

3. **Transfer to a multipatch SIS model**
   Adapt the DINN/PINN framework to a multipatch SIS epidemic model with mass-action infection and complex endemic equilibrium structure.

The broader aim is to become comfortable customizing DINNs across different epidemic models, especially models involving spatial heterogeneity, population movement, and nonlinear disease dynamics.

---

## What is a PINN?

A **Physics-Informed Neural Network**, or **PINN**, is a neural network trained not only to fit data, but also to satisfy a governing differential equation.

In a standard machine-learning model, the network is usually trained by minimizing prediction error:

```text
prediction - observed data
```

A PINN adds another requirement: the prediction should obey the mathematical law that governs the system.

For a differential equation such as

```text
du/dt = f(u, t, parameters),
```

a neural network can be used to approximate the unknown solution:

```text
NN(t) ≈ u(t).
```

Because neural networks are differentiable, automatic differentiation can compute

```text
dNN(t)/dt.
```

The PINN then penalizes the difference between the neural-network derivative and the right-hand side of the differential equation:

```text
physics residual = neural network derivative - differential equation right-hand side
```

So the network is trained to satisfy both:

1. the observed data,
2. the governing equation.

A typical PINN loss has the form:

```text
total loss = data loss + physics loss + initial/boundary condition loss
```

where:

* **data loss** measures how well the network matches observed data,
* **physics loss** measures how well the network satisfies the differential equation,
* **initial/boundary condition loss** anchors the solution to known starting or boundary values.

This makes PINNs useful for scientific machine learning because they combine data-driven learning with mathematical modeling.

---

## What is a DINN?

A **Disease-Informed Neural Network**, or **DINN**, is a PINN specialized for disease dynamics and compartmental epidemic models.

In epidemic modeling, the governing equations usually describe how people move between disease compartments such as:

```text
S(t): susceptible population
I(t): infected population
R(t): recovered population
D(t): deceased population
```

For a basic SIRD model, a DINN may take time as input and return the epidemic compartments as output:

```text
NN(t) ≈ (S(t), I(t), R(t), D(t)).
```

The DINN is trained to do two things at once:

1. learn the epidemic trajectory,
2. discover the hidden disease parameters that make the trajectory satisfy the epidemic model.

For example, parameters such as:

* transmission rate,
* recovery rate,
* mortality rate,
* movement rate,
* patch-specific transmission rates,
* patch-specific recovery rates,

can be treated as trainable variables.

This means a DINN is not just fitting epidemic curves. It is also learning the parameters inside the governing disease equations.

---

## Why Neural Networks Matter for Disease Dynamics

Neural networks are important in epidemic modeling because real disease data is often incomplete, noisy, sparse, delayed, or affected by changing interventions.

Traditional parameter estimation usually follows this workflow:

```text
choose parameters → solve ODE → compare with data → adjust parameters → solve again
```

This can become expensive and difficult when the model has many compartments, many parameters, or complicated movement between populations.

A DINN changes the workflow:

```text
data + governing equations → learn trajectory + learn parameters
```

This is useful because:

* epidemic data may only be available at discrete time points,
* the ODE residual helps prevent the network from becoming a black-box curve fitter,
* automatic differentiation allows the model to enforce differential equations directly,
* neural networks provide smooth approximations of epidemic curves,
* parameter discovery and trajectory fitting are combined in one optimization problem,
* the same framework can be adapted from simple models to more complex spatial or multipatch models.

In this project, neural networks are used as computational tools for learning, estimating, and exploring epidemic dynamics. They do not replace mathematical analysis; instead, they complement it.

---

## Mathematical Models Used in This Project

### 1. Basic SIRD Model

The first stage of the project uses the classical SIRD model.

The compartments are:

* `S`: susceptible individuals,
* `I`: infectious individuals,
* `R`: recovered individuals,
* `D`: dead/deceased individuals.

The model is

```math
\frac{dS}{dt} = -\frac{\beta}{N}SI,
```

```math
\frac{dI}{dt} = \frac{\beta}{N}SI - \gamma I - \omega I,
```

```math
\frac{dR}{dt} = \gamma I,
```

```math
\frac{dD}{dt} = \omega I.
```

Here:

* `N` is the total population,
* `beta` is the transmission rate,
* `gamma` is the recovery rate,
* `omega` is the mortality rate.

The project may also work with fractions of the population:

```text
s = S/N, i = I/N, r = R/N, d = D/N.
```

Using fractions keeps the numerical values close to order 1, which often makes neural-network training more stable.

In this stage, the neural network learns:

```text
NN(t) ≈ (S(t), I(t), R(t), D(t))
```

or, in fraction form,

```text
NN(t) ≈ (s(t), i(t), r(t), d(t)).
```

The trainable disease parameters are:

* `beta`: transmission rate,
* `gamma`: recovery rate,
* `omega`: mortality rate.

The main goal of this stage is to recover known parameters from synthetic data and verify that the DINN works.

---

### 2. SIRD Model with Population Movement

The second stage extends the SIRD model by introducing two interacting regions.

Instead of one population, there are now two populations:

Region 1:

```text
S1, I1, R1, D1
```

Region 2:

```text
S2, I2, R2, D2
```

The neural network output dimension increases from 4 to 8:

```text
NN(t) ≈ (S1, I1, R1, D1, S2, I2, R2, D2).
```

Movement terms are added to represent transportation between regions.

For example:

* `tau_12`: movement rate from region 1 to region 2,
* `tau_21`: movement rate from region 2 to region 1.

This stage teaches how DINNs scale when the epidemic model includes:

* multiple populations,
* spatial structure,
* transportation,
* additional unknown parameters,
* region-specific disease dynamics.

The goal is to determine whether the DINN can recover both disease parameters and movement parameters from synthetic data.

---

### 3. Multipatch SIS Model with Mass-Action Infection

The final stage transfers the DINN idea to a multipatch SIS epidemic model with mass-action infection.

In an SIS model, individuals move between two disease states:

```text
S → I → S.
```

That means infected individuals can recover and return to the susceptible class.

For patch `i`, the compartments are:

```text
S_i(t): susceptible population in patch i
I_i(t): infected population in patch i
```

The general structure of the equations is:

```text
dS_i/dt = susceptible movement - infection + recovery
```

```text
dI_i/dt = infected movement + infection - recovery
```

For a system with `n` patches, the neural network output size becomes:

```text
2n
```

because each patch has one susceptible and one infected compartment.

The neural network approximates:

```text
NN(t) ≈ (S_1(t), ..., S_n(t), I_1(t), ..., I_n(t)).
```

This stage is connected to the study of endemic equilibria. The goal is not only to fit epidemic trajectories, but also to explore how equilibrium disease levels change as key parameters vary.

Important parameters include:

* susceptible dispersal rate `dS`,
* infected dispersal rate `dI`,
* total population size `N`,
* reproduction number `R0`,
* patch-level transmission rates,
* patch-level recovery rates,
* connectivity matrix between patches.

This stage connects the computational DINN framework with deeper mathematical questions about multiple endemic equilibria, loops, bounded and unbounded branches, and S-shaped bifurcation curves.

---

## Project Execution Plan

## Phase 1: Build the Basic SIRD-DINN

The first phase establishes the baseline DINN workflow.

### Steps

1. Generate synthetic SIRD data using known true parameters.
2. Build a PyTorch neural network with input `t` and output `(S, I, R, D)`.
3. Normalize time and state variables.
4. Define trainable parameters for `beta`, `gamma`, and `omega`.
5. Build the three-part loss:

   * data loss,
   * ODE residual loss,
   * initial condition loss.
6. Train the model using Adam.
7. Compare recovered parameters with the true parameters.
8. Plot true vs predicted epidemic trajectories.

### Expected Outputs

* training loss curve,
* SIRD trajectory plot,
* table of true, initial guess, and recovered parameters.

---

## Phase 2: Extend to SIRD with Transportation

The second phase introduces population movement between two regions.

### Steps

1. Extend the state vector from 4 to 8 compartments.
2. Modify the neural-network output layer.
3. Rewrite the ODE residuals to include transportation terms.
4. Add trainable movement parameters such as `tau_12` and `tau_21`.
5. Train on synthetic two-region epidemic data.
6. Evaluate whether the DINN can recover both disease parameters and movement parameters.
7. Add optional experiments with noisy and missing observations.

### Expected Outputs

* two-region trajectory plots,
* recovered disease and movement parameter table,
* robustness experiments with noisy data,
* comparison of true and learned transportation parameters.

---

## Phase 3: Transfer to the SIS Patch Model

The third phase adapts the DINN framework to the multipatch SIS model.

### Steps

1. Define the multipatch SIS model with mass-action infection.
2. Set the neural-network output size to `2n`, where `n` is the number of patches.
3. Include the connectivity matrix for movement between patches.
4. Implement the SIS ODE residuals.
5. Decide which parameters are known and which are trainable.
6. Train the DINN on synthetic SIS patch data.
7. Explore steady-state or equilibrium behavior.
8. Use a steady-state PINN idea where time derivatives are set to zero to approximate endemic equilibria.
9. Vary parameters such as `dS`, `R0`, or `N`.
10. Visualize the equilibrium structure.
11. Compare computational results with known theoretical behavior.

### Expected Outputs

* SIS patch trajectory plots,
* endemic equilibrium plots,
* bifurcation-style diagrams,
* comparison between numerical DINN/PINN results and mathematical theory.

---

## Evaluation and Validation

The project will be evaluated using both machine-learning and mathematical-modeling criteria.

### 1. Trajectory Fit

Compare the DINN-predicted epidemic curves against the synthetic ground-truth ODE solution.

This includes plots of:

* susceptible population,
* infected population,
* recovered population,
* deceased population.

### 2. Parameter Recovery Error

Compare the learned parameters with the true parameters used to generate synthetic data.

For each parameter, compute:

```text
absolute error = |learned value - true value|
```

and

```text
relative error = |learned value - true value| / |true value|.
```

### 3. Robustness to Noisy Data

Add noise to the synthetic observations and test whether the DINN can still recover meaningful trajectories and parameters.

### 4. Robustness to Missing Observations

Remove some observations and test whether the ODE residual helps the network learn reasonable dynamics from sparse data.

### 5. Comparison with Numerical ODE Solutions

Compare DINN predictions against classical numerical ODE solvers such as `odeint` or `solve_ivp`.

### 6. Comparison with SIS Patch Theory

For the multipatch SIS model, compare the learned or computed equilibrium structures with known theoretical behavior, such as:

* multiple endemic equilibria,
* loops in equilibrium branches,
* bounded and unbounded branches,
* S-shaped bifurcation diagrams,
* changes in disease persistence as movement rates vary.

---

## Why This Project Matters

This project sits at the intersection of:

* scientific machine learning,
* mathematical epidemiology,
* inverse problems,
* neural differential equations,
* disease modeling,
* population mobility,
* computational applied mathematics.

The project is important because it builds a bridge between rigorous mathematical epidemic models and modern data-driven machine-learning methods.

On the mathematical side, epidemic models provide structure, interpretability, and theoretical insight. On the machine-learning side, neural networks provide flexible approximation, automatic differentiation, and data-driven parameter discovery.

By combining both, DINNs offer a way to study disease dynamics without treating the neural network as a black box. The model is still guided by the epidemic equations.

---

## Limitations

This project has several important limitations.

First, synthetic data is easier to work with than real epidemic data. Synthetic data is clean, controlled, and generated from the same equations used in training.

Second, parameter identifiability can be difficult. Different parameter combinations may produce similar epidemic trajectories.

Third, PINN and DINN training can be sensitive to loss weights. If the data loss, physics loss, and initial condition loss are not balanced properly, the model may fail to converge.

Fourth, neural networks may converge to poor local minima or produce inaccurate parameter estimates.

Fifth, real-world disease data often includes reporting delays, undercounting, missing values, changes in testing policy, behavioral changes, and interventions.

Finally, DINNs should complement mathematical analysis, not replace it. For complex epidemic models, theoretical results remain essential for understanding long-term behavior.

---

## Future Work

Possible future extensions include:

* applying the framework to real epidemic data,
* extending the transportation model to more than two regions,
* allowing parameters to vary with time,
* adding uncertainty quantification,
* building Bayesian or ensemble DINNs,
* comparing DINNs with classical least-squares parameter estimation,
* applying the framework to SEIR, SIRS, or vector-borne disease models,
* studying how well steady-state PINNs can recover endemic equilibrium branches,
* testing whether DINNs can detect early signs of backward bifurcation or multiple equilibria.

---

## References

This project is motivated by work on:
0. ICM Satelite conference at MSU
1. Physics-Informed Neural Networks for solving and discovering differential equations.
2. Disease-Informed Neural Networks for compartmental epidemic models.
3. PINN/DINN methods for infectious disease dynamics with travel and transportation.
4. Multipatch SIS epidemic models with mass-action infection and complex endemic equilibrium structure.

Specific references will be added as the project develops.

---

## Author

**Kingsley C. Ukandu**
Applied Mathematics | Scientific Machine Learning | Mathematical Epidemiology | Health Data Analytics

---

## Project Vision

The long-term vision of this project is to develop a transferable workflow for customizing DINNs across epidemic models.

The project begins with a simple SIRD model, grows into a movement-aware epidemic model, and eventually connects to a research-level SIS patch model with complex equilibrium behavior.

The central learning objective is:

```text
Become comfortable taking a disease model, identifying its compartments and equations, building the correct neural-network output structure, writing the ODE residuals, and training the DINN to recover trajectories and parameters.
```

This project is both a coding project and a mathematical modeling project.
