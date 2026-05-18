# Dynamic Portfolio Allocation under Income Uncertainty

This project implements a finite-horizon stochastic dynamic programming model for salaried households in Jharkhand with:

- stochastic labor income (Markov chain)
- stochastic risky-asset returns (Markov regime-switching)
- dynamic portfolio choice each period

## 1) State-transition model

State at time t:

- Wealth: W_t (discretized grid)
- Income state: y_t in {low, middle, high}
- Return regime: z_t in {bear, neutral, bull}

Action at time t:

- Risky portfolio share: alpha_t in [0, 1] (discrete action grid)

Transition:

- Income follows Markov matrix P_y
- Return regime follows Markov matrix P_z
- Wealth evolves as

W_{t+1} = (W_t + s y_t) [alpha_t (1 + r^R_{t+1}) + (1 - alpha_t)(1 + r_f)]

where:

- s is saving rate out of salary
- r_f is risk-free rate
- r^R_{t+1} is risky return implied by next return regime

## 2) Bellman equation (finite horizon)

Terminal value:

V_T(W, y, z) = u(W), where u is CRRA utility.

Backward recursion for t = T-1, ..., 0:

V_t(W, y, z) = max_{alpha in A} beta E[V_{t+1}(W', y', z') | y, z, alpha]

subject to the transition model above.

## 3) Algorithms implemented

- Backward value iteration (finite horizon DP)
- Stage-wise policy iteration (policy evaluation + policy improvement)

The script compares the resulting policies from both methods.

## 4) Simulation + policy plots

The script simulates many individuals over the horizon and saves:

- Policy curves by wealth for each return regime
- Policy heatmap for (wealth x income) at t=0
- Wealth-path summary plot (mean and percentile band)
- Average risky share over time

## Run

1. Install dependencies

pip install -r requirements.txt

2. Run

python dynamic_portfolio_sdp.py

3. See generated files in outputs/.

## Files

- dynamic_portfolio_sdp.py: full model + solver + simulation + plotting
- requirements.txt: minimal dependencies
- outputs/: generated plots and arrays

## Notes for report writing

You can cite the model assumptions as a stylized framework for salaried households in Jharkhand. In your final report/presentation, add:

- sensitivity checks (risk aversion, saving rate, transition matrices)
- calibration notes (if you have district-level salary and return proxy data)
- policy interpretation (how uncertainty shifts risky allocation)
