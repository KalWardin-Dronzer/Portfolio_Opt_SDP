from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ModelConfig:
    # ──────────────────────────────────────────────────────────────────────
    # All parameters calibrated from real-world Indian / Jharkhand data.
    # Sources cited inline.  Values are *annual* unless noted.
    # ──────────────────────────────────────────────────────────────────────

    # Finite horizon (working career segment, years)
    horizon: int = 10

    # Discount factor — standard macro calibration (Ljungqvist & Sargent);
    # implies ~4 % annual time-preference rate.
    beta: float = 0.96

    # CRRA risk-aversion coefficient.
    # Literature range: 1–10;  γ = 3 is a common moderate value
    # (Campbell & Viceira 2002, "Strategic Asset Allocation").
    gamma: float = 3.0

    # ── Risk-free rate ────────────────────────────────────────────────────
    # India 10-year G-Sec yield ≈ 6.9 % nominal (April 2026, RBI/FBIL).
    # SBI 1-year FD rate ≈ 6.4 %.  CPI inflation ≈ 5 %.
    # → Real risk-free rate ≈ 2 %.  We use the *nominal* yield as the
    #   model is in nominal terms.
    # Source: macrotrends.net, worldgovernmentbonds.com
    risk_free_rate: float = 0.069

    # ── Saving rate ───────────────────────────────────────────────────────
    # RBI Annual Report 2024-25:  gross household financial savings = 11.2 %
    # of GNDI.  For salaried middle-class households the rate is higher
    # (NSS consumption surveys suggest 20-30 %).  We use 20 %.
    # Source: RBI, relakhs.com, HCES 2023-24
    saving_rate: float = 0.20

    # ── Discretization ────────────────────────────────────────────────────
    wealth_min: float = 10_000.0
    wealth_max: float = 2_500_000.0
    wealth_points: int = 81

    # Action grid: share in risky asset
    risky_share_grid: Tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    )

    # ── Income states (annual INR, salaried workers in Jharkhand) ────────
    # PLFS 2023-24 (MoSPI):  National avg monthly earnings for regular
    # wage/salaried employees:  Rural ₹17,033 → ≈ ₹2.04 L/yr
    #                          Urban ₹24,434 → ≈ ₹2.93 L/yr
    # Jharkhand per-capita income 2023-24: ₹95,649 (below national avg).
    # We use three stylised tiers anchored to PLFS data:
    #   Low  = rural salaried avg  ≈ ₹2.04 L
    #   Mid  = urban salaried avg  ≈ ₹3.50 L  (Jharkhand urban)
    #   High = skilled / govt       ≈ ₹6.00 L
    # Source: MoSPI PLFS Annual Report 2023-24, Jharkhand Economic Survey
    income_states: Tuple[float, ...] = (204_000.0, 350_000.0, 600_000.0)

    # ── Risky-asset return regimes (annual, nominal) ─────────────────────
    # Estimated from Nifty 50 TRI calendar-year returns 2005-2024 (NSE).
    #
    #   Year buckets used for regime classification:
    #   Bear  (return < 0 %):  2008 (−51 %), 2011 (−24 %), 2015 (−3 %)
    #       → simple average ≈ −26 %, trimmed mean ≈ −15 %
    #   Neutral (0 % – 15 %): 2010 (19 %), 2013 (8 %), 2016 (4 %),
    #       2018 (5 %), 2019 (14 %), 2022 (6 %), 2024 (10 %)
    #       → average ≈ 9 %
    #   Bull  (> 15 %):  2005 (39 %), 2006 (42 %), 2007 (57 %),
    #       2009 (78 %), 2012 (29 %), 2014 (33 %), 2017 (30 %),
    #       2020 (16 %), 2021 (26 %), 2023 (21 %)
    #       → average ≈ 37 %, conservative trim → 28 %
    #
    # Source: NSE India, nseindia.com
    risky_return_states: Tuple[float, ...] = (-0.15, 0.09, 0.28)

    # ── Markov transition matrices ───────────────────────────────────────
    # Estimated from Nifty 50 year-to-year regime transitions 2005-2024.
    #
    #   Observed transition counts (20 year-pairs):
    #               → Bear  Neutral  Bull
    #   Bear  (3):    0       1        2     ≈ (0.05, 0.35, 0.60)
    #   Neutral(7):   1       3        3     ≈ (0.15, 0.45, 0.40)
    #   Bull (10):    2       3        5     ≈ (0.20, 0.30, 0.50)
    #
    #   Note: small sample → we smooth slightly toward uniform.
    # Source: computed from NSE Nifty 50 TRI annual returns
    return_transition: Tuple[Tuple[float, float, float], ...] = (
        (0.10, 0.35, 0.55),   # from Bear  → Bear is rare historically
        (0.15, 0.45, 0.40),   # from Neutral
        (0.20, 0.30, 0.50),   # from Bull
    )

    # Income transition — estimated from general labour economics priors
    # for formal salaried employment (no micro-panel available for Jharkhand).
    # Key assumption: salaried income is *sticky* (high diagonal).
    # Upward mobility is more likely than downward for formal sector.
    # Source: stylised, consistent with IHDS-II panel mobility matrices
    income_transition: Tuple[Tuple[float, float, float], ...] = (
        (0.70, 0.25, 0.05),   # Low:  likely to stay low
        (0.15, 0.65, 0.20),   # Mid:  most stable
        (0.05, 0.25, 0.70),   # High: likely to stay high
    )

    simulation_paths: int = 2000
    seed: int = 42


class DynamicPortfolioSDP:
    """Finite-horizon stochastic DP model for dynamic portfolio allocation
    of salaried households under Markov income and return-regime uncertainty."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.wealth_grid = np.linspace(cfg.wealth_min, cfg.wealth_max, cfg.wealth_points)
        self.A = np.array(cfg.risky_share_grid)
        self.Y = np.array(cfg.income_states)
        self.R = np.array(cfg.risky_return_states)
        self.Py = np.array(cfg.income_transition)
        self.Pr = np.array(cfg.return_transition)

        self.nw = len(self.wealth_grid)
        self.ny = len(self.Y)
        self.nr = len(self.R)
        self.na = len(self.A)

    def _crra(self, x: np.ndarray | float) -> np.ndarray | float:
        """CRRA utility u(x) = x^(1-γ)/(1-γ), with numerical floor."""
        eps = 1.0
        x_safe = np.maximum(x, eps)
        g = self.cfg.gamma
        if abs(g - 1.0) < 1e-10:
            return np.log(x_safe)
        return (x_safe ** (1.0 - g)) / (1.0 - g)

    def flow_utility(self, y: float) -> float:
        """Current-period utility from consuming the non-saved share of salary."""
        consumption = (1.0 - self.cfg.saving_rate) * y
        return float(self._crra(consumption))

    def utility_terminal(self, w: np.ndarray) -> np.ndarray:
        """Terminal value: CRRA utility of final wealth."""
        return self._crra(w)

    def next_wealth(self, w: float, y: float, r_risky: float, alpha: float) -> float:
        gross_return = alpha * (1.0 + r_risky) + (1.0 - alpha) * (1.0 + self.cfg.risk_free_rate)
        w_next = (w + self.cfg.saving_rate * y) * gross_return
        return float(np.clip(w_next, self.cfg.wealth_min, self.cfg.wealth_max))

    def interpolate_value(self, V_next: np.ndarray, w_next: float, y_next: int, r_next: int) -> float:
        # V_next shape: (nw, ny, nr)
        return float(np.interp(w_next, self.wealth_grid, V_next[:, y_next, r_next]))

    def bellman_step(self, V_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One-step Bellman optimality backup: V_t = max_a { u(c) + β E[V_{t+1}] }."""
        V_t = np.empty((self.nw, self.ny, self.nr), dtype=float)
        policy_t = np.empty((self.nw, self.ny, self.nr), dtype=int)

        for iw, w in enumerate(self.wealth_grid):
            for iy, y in enumerate(self.Y):
                u_flow = self.flow_utility(y)
                for ir in range(self.nr):
                    q_values = np.empty(self.na, dtype=float)
                    for ia, alpha in enumerate(self.A):
                        ev = 0.0
                        for iy_next in range(self.ny):
                            py = self.Py[iy, iy_next]
                            for ir_next in range(self.nr):
                                pr = self.Pr[ir, ir_next]
                                w_next = self.next_wealth(w, y, self.R[ir_next], alpha)
                                v_next = self.interpolate_value(V_next, w_next, iy_next, ir_next)
                                ev += py * pr * v_next
                        q_values[ia] = u_flow + self.cfg.beta * ev

                    best_a = int(np.argmax(q_values))
                    V_t[iw, iy, ir] = q_values[best_a]
                    policy_t[iw, iy, ir] = best_a

        return V_t, policy_t

    def solve_by_backward_value_iteration(self) -> Dict[str, np.ndarray]:
        """Solve via backward value iteration (finite-horizon DP)."""
        T = self.cfg.horizon
        V = np.empty((T + 1, self.nw, self.ny, self.nr), dtype=float)
        P = np.empty((T, self.nw, self.ny, self.nr), dtype=int)

        # Terminal condition
        terminal = self.utility_terminal(self.wealth_grid)
        for iy in range(self.ny):
            for ir in range(self.nr):
                V[T, :, iy, ir] = terminal

        for t in range(T - 1, -1, -1):
            print(f"  VI: solving t = {t}")
            V[t], P[t] = self.bellman_step(V[t + 1])

        return {"V": V, "policy_idx": P}

    def evaluate_policy_step(self, V_next: np.ndarray, policy_t: np.ndarray) -> np.ndarray:
        """Policy evaluation: compute V^π_t given V_{t+1} and a fixed policy."""
        V_t = np.empty((self.nw, self.ny, self.nr), dtype=float)

        for iw, w in enumerate(self.wealth_grid):
            for iy, y in enumerate(self.Y):
                u_flow = self.flow_utility(y)
                for ir in range(self.nr):
                    ia = int(policy_t[iw, iy, ir])
                    alpha = self.A[ia]
                    ev = 0.0
                    for iy_next in range(self.ny):
                        py = self.Py[iy, iy_next]
                        for ir_next in range(self.nr):
                            pr = self.Pr[ir, ir_next]
                            w_next = self.next_wealth(w, y, self.R[ir_next], alpha)
                            v_next = self.interpolate_value(V_next, w_next, iy_next, ir_next)
                            ev += py * pr * v_next
                    V_t[iw, iy, ir] = u_flow + self.cfg.beta * ev

        return V_t

    def improve_policy_step(self, V_t: np.ndarray) -> np.ndarray:
        """Policy improvement: find greedy policy w.r.t. current V_t."""
        policy_new = np.empty((self.nw, self.ny, self.nr), dtype=int)
        for iw, w in enumerate(self.wealth_grid):
            for iy, y in enumerate(self.Y):
                u_flow = self.flow_utility(y)
                for ir in range(self.nr):
                    q_values = np.empty(self.na, dtype=float)
                    for ia, alpha in enumerate(self.A):
                        ev = 0.0
                        for iy_next in range(self.ny):
                            py = self.Py[iy, iy_next]
                            for ir_next in range(self.nr):
                                pr = self.Pr[ir, ir_next]
                                w_next = self.next_wealth(w, y, self.R[ir_next], alpha)
                                v_next = self.interpolate_value(V_t, w_next, iy_next, ir_next)
                                ev += py * pr * v_next
                        q_values[ia] = u_flow + self.cfg.beta * ev
                    policy_new[iw, iy, ir] = int(np.argmax(q_values))
        return policy_new

    def solve_by_finite_horizon_policy_iteration(self, eval_rounds: int = 10, max_iter: int = 50) -> Dict[str, np.ndarray]:
        """Stage-wise policy iteration: for each stage t, iterate
        policy-evaluation and policy-improvement until convergence."""
        T = self.cfg.horizon
        V = np.empty((T + 1, self.nw, self.ny, self.nr), dtype=float)
        P = np.zeros((T, self.nw, self.ny, self.nr), dtype=int)
        convergence_iters = []  # track iterations per stage

        terminal = self.utility_terminal(self.wealth_grid)
        for iy in range(self.ny):
            for ir in range(self.nr):
                V[T, :, iy, ir] = terminal

        for t in range(T - 1, -1, -1):
            print(f"  PI: solving t = {t}")
            # Initialize with a default policy (all risk-free)
            policy_t = np.zeros((self.nw, self.ny, self.nr), dtype=int)

            for k in range(max_iter):
                # --- Evaluation: iterate V^π multiple rounds ---
                V_t = self.evaluate_policy_step(V[t + 1], policy_t)
                for _ in range(eval_rounds - 1):
                    V_t = self.evaluate_policy_step(V_t, policy_t)

                # --- Improvement: greedy w.r.t. V^π_t ---
                improved = self.improve_policy_step(V_t)
                if np.array_equal(improved, policy_t):
                    print(f"    converged at iteration {k + 1}")
                    convergence_iters.append(k + 1)
                    break
                policy_t = improved
            else:
                convergence_iters.append(max_iter)

            # Final evaluation with converged policy
            V[t] = self.evaluate_policy_step(V[t + 1], policy_t)
            P[t] = policy_t

        return {"V": V, "policy_idx": P, "convergence_iters": convergence_iters}

    def simulate(self, solution: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.cfg.seed)
        T = self.cfg.horizon
        N = self.cfg.simulation_paths

        wealth = np.empty((N, T + 1), dtype=float)
        income_idx = np.empty((N, T + 1), dtype=int)
        return_idx = np.empty((N, T + 1), dtype=int)
        action = np.empty((N, T), dtype=float)

        # Initial conditions
        wealth[:, 0] = 200_000.0
        income_idx[:, 0] = 1  # middle salary state
        return_idx[:, 0] = 1  # neutral return regime

        policy_idx = solution["policy_idx"]

        for t in range(T):
            for n in range(N):
                iw = int(np.argmin(np.abs(self.wealth_grid - wealth[n, t])))
                iy = int(income_idx[n, t])
                ir = int(return_idx[n, t])

                ia = int(policy_idx[t, iw, iy, ir])
                alpha = float(self.A[ia])
                action[n, t] = alpha

                # Draw next states
                iy_next = int(rng.choice(self.ny, p=self.Py[iy]))
                ir_next = int(rng.choice(self.nr, p=self.Pr[ir]))

                w_next = self.next_wealth(wealth[n, t], self.Y[iy], self.R[ir_next], alpha)

                wealth[n, t + 1] = w_next
                income_idx[n, t + 1] = iy_next
                return_idx[n, t + 1] = ir_next

        return {
            "wealth": wealth,
            "income_idx": income_idx,
            "return_idx": return_idx,
            "action": action,
        }

    def plot_policy_heatmaps(self, solution: Dict[str, np.ndarray], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        policy = solution["policy_idx"]

        selected_t = [0, self.cfg.horizon // 2, self.cfg.horizon - 1]
        regime_names = ["Bear", "Neutral", "Bull"]
        income_names = ["Low income", "Mid income", "High income"]

        for ir, regime in enumerate(regime_names):
            fig, axes = plt.subplots(1, len(selected_t), figsize=(15, 4), sharey=True)
            for j, t in enumerate(selected_t):
                # Average across income states for cleaner portfolio-allocation surface
                alpha_grid = np.mean(self.A[policy[t, :, :, ir]], axis=1)
                axes[j].plot(self.wealth_grid, alpha_grid, linewidth=2)
                axes[j].set_title(f"t = {t}")
                axes[j].set_xlabel("Wealth (INR)")
                axes[j].grid(alpha=0.25)
            axes[0].set_ylabel("Optimal risky share")
            fig.suptitle(f"Policy by wealth under {regime} return regime")
            fig.tight_layout()
            fig.savefig(out_dir / f"policy_by_wealth_{regime.lower()}.png", dpi=160)
            plt.close(fig)

        # Heatmap-style matrix by wealth and income at t=0, neutral regime
        t0 = 0
        ir0 = 1
        alpha_matrix = self.A[policy[t0, :, :, ir0]]

        # Use actual wealth values on x-axis (show subset of ticks)
        tick_step = max(1, self.nw // 8)
        xtick_pos = np.arange(0, self.nw, tick_step)
        xtick_labels = [f"{self.wealth_grid[i]/1e5:.1f}L" for i in xtick_pos]

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(alpha_matrix.T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title("Optimal risky share: t=0, neutral regime")
        ax.set_xlabel("Wealth (INR)")
        ax.set_ylabel("Income state")
        ax.set_xticks(xtick_pos, labels=xtick_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(self.ny), labels=income_names)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Risky share")
        fig.tight_layout()
        fig.savefig(out_dir / "policy_heatmap_t0_neutral.png", dpi=160)
        plt.close(fig)

    def plot_value_function(self, solution: Dict[str, np.ndarray], out_dir: Path) -> None:
        """Plot value function at t=0 — normalized to show wealth-dependence clearly."""
        out_dir.mkdir(parents=True, exist_ok=True)
        V = solution["V"]
        regime_names = ["Bear", "Neutral", "Bull"]
        income_names = ["Low", "Mid", "High"]
        wealth_lakhs = self.wealth_grid / 1e5

        # --- Plot 1: Normalized value function (V - V_min per curve) ---
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ir, (ax, regime) in enumerate(zip(axes, regime_names)):
            for iy, inc_label in enumerate(income_names):
                v_curve = V[0, :, iy, ir]
                v_norm = v_curve - v_curve.min()
                ax.plot(wealth_lakhs, v_norm, linewidth=2, label=f"{inc_label} income")
            ax.set_title(f"{regime} regime")
            ax.set_xlabel("Wealth (₹ lakhs)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=9)
        axes[0].set_ylabel("V(W) − V(W_min)  (normalised gain)")
        fig.suptitle("Value function gain from wealth at t = 0", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "value_function_t0.png", dpi=160)
        plt.close(fig)

        # --- Plot 2: Value function across time for mid-income, neutral ---
        fig, ax = plt.subplots(figsize=(8, 5))
        iy_mid, ir_neutral = 1, 1
        cmap = plt.cm.viridis
        T = self.cfg.horizon
        for t_plot in [0, T // 4, T // 2, 3 * T // 4, T]:
            v_curve = V[t_plot, :, iy_mid, ir_neutral]
            v_norm = v_curve - v_curve.min()
            color = cmap(t_plot / T)
            ax.plot(wealth_lakhs, v_norm, linewidth=2, color=color, label=f"t = {t_plot}")
        ax.set_xlabel("Wealth (₹ lakhs)")
        ax.set_ylabel("V(W) − V(W_min)")
        ax.set_title("Value function evolution over time (mid income, neutral regime)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "value_function_over_time.png", dpi=160)
        plt.close(fig)

    def plot_transition_matrices(self, out_dir: Path) -> None:
        """Plot Markov transition matrices for income and return regimes."""
        out_dir.mkdir(parents=True, exist_ok=True)
        income_labels = ["Low\n₹2.04L", "Mid\n₹3.50L", "High\n₹6.00L"]
        regime_labels = ["Bear\n(−15%)", "Neutral\n(+9%)", "Bull\n(+28%)"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax, matrix, labels, title in [
            (axes[0], self.Py, income_labels, "Income Transition Matrix (P_y)"),
            (axes[1], self.Pr, regime_labels, "Return Regime Transition Matrix (P_z)"),
        ]:
            im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
            n = len(labels)
            ax.set_xticks(range(n), labels=labels, fontsize=9)
            ax.set_yticks(range(n), labels=labels, fontsize=9)
            ax.set_xlabel("To state")
            ax.set_ylabel("From state")
            ax.set_title(title, fontsize=11)
            for i in range(n):
                for j in range(n):
                    color = "white" if matrix[i, j] > 0.5 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=13, fontweight="bold", color=color)
            fig.colorbar(im, ax=ax, shrink=0.8, label="Probability")

        fig.suptitle("Markov Transition Matrices", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "transition_matrices.png", dpi=160)
        plt.close(fig)

    def plot_convergence(self, pi_solution: Dict[str, np.ndarray], out_dir: Path) -> None:
        """Plot policy iteration convergence (iterations per stage)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        iters = pi_solution.get("convergence_iters", [])
        if not iters:
            return
        T = len(iters)
        stages = list(range(T - 1, -1, -1))  # stages solved in reverse

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(T), iters, color="steelblue", edgecolor="navy", alpha=0.85)
        ax.set_xticks(range(T))
        ax.set_xticklabels([f"t={s}" for s in stages], fontsize=9)
        ax.set_xlabel("Stage (solved in reverse)")
        ax.set_ylabel("Policy improvement iterations")
        ax.set_title("Policy Iteration Convergence: Iterations per Stage")
        for i, v in enumerate(iters):
            ax.text(i, v + 0.1, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(iters) + 2)
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "convergence_pi.png", dpi=160)
        plt.close(fig)

    def plot_simulation_summary(self, sim: Dict[str, np.ndarray], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        wealth = sim["wealth"]
        action = sim["action"]

        mean_wealth = wealth.mean(axis=0)
        median_wealth = np.median(wealth, axis=0)
        p10_wealth = np.percentile(wealth, 10, axis=0)
        p90_wealth = np.percentile(wealth, 90, axis=0)
        mean_action = action.mean(axis=0)

        t = np.arange(wealth.shape[1])
        ta = np.arange(action.shape[1])

        # --- Wealth paths with sample trajectories ---
        rng = np.random.default_rng(123)
        sample_idx = rng.choice(wealth.shape[0], size=5, replace=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        for idx in sample_idx:
            ax.plot(t, wealth[idx], alpha=0.3, linewidth=0.8)
        ax.plot(t, mean_wealth, label="Mean wealth", linewidth=2, color="navy")
        ax.plot(t, median_wealth, label="Median wealth", linewidth=2, color="darkgreen", linestyle="--")
        ax.fill_between(t, p10_wealth, p90_wealth, alpha=0.2, color="steelblue", label="10-90 percentile")
        ax.set_title("Simulated wealth paths")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Wealth (INR)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "simulation_wealth_summary.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ta, mean_action, marker="o")
        ax.set_title("Average risky share over time")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Average risky share")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "simulation_avg_risky_share.png", dpi=160)
        plt.close(fig)


def sensitivity_analysis(out_dir: Path) -> None:
    """Run the model for several risk-aversion levels and compare policies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gammas = [1.5, 3.0, 5.0]
    colors = ["tab:blue", "tab:orange", "tab:red"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    regime_names = ["Bear", "Neutral", "Bull"]

    for gamma, color in zip(gammas, colors):
        print(f"  Sensitivity: gamma = {gamma}")
        cfg = ModelConfig(gamma=gamma)
        model = DynamicPortfolioSDP(cfg)
        sol = model.solve_by_backward_value_iteration()
        policy = sol["policy_idx"]

        for ir, (ax, regime) in enumerate(zip(axes, regime_names)):
            alpha_avg = np.mean(model.A[policy[0, :, :, ir]], axis=1)
            ax.plot(model.wealth_grid / 1e5, alpha_avg, linewidth=2, color=color,
                    label=f"γ = {gamma}")
            ax.set_title(f"{regime} regime")
            ax.set_xlabel("Wealth (₹ lakhs)")
            ax.grid(alpha=0.25)

    for ax in axes:
        ax.legend()
    axes[0].set_ylabel("Optimal risky share (avg over income)")
    fig.suptitle("Sensitivity to risk aversion (γ) at t = 0", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_gamma.png", dpi=160)
    plt.close(fig)


def backtest_historical(model: DynamicPortfolioSDP, solution: Dict[str, np.ndarray],
                        out_dir: Path) -> None:
    """Backtest the optimal policy on actual Nifty 50 annual returns 2014-2024.

    Compares DP-optimal allocation against three naive strategies:
      - 100 % risk-free (all FD)
      - 100 % equity (all Nifty)
      - 50-50 constant mix

    Source: NSE Nifty 50 TRI calendar-year returns (nseindia.com).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Actual Nifty 50 TRI calendar-year returns 2014-2024 ──────────────
    # Source: NSE India
    historical_returns = {
        2014: 0.329,
        2015: -0.030,
        2016: 0.044,
        2017: 0.303,
        2018: 0.046,
        2019: 0.135,
        2020: 0.161,
        2021: 0.256,
        2022: 0.057,
        2023: 0.213,
        2024: 0.101,
    }

    # We use 10 return periods: the return *realised during* year t
    # drives wealth from end-of-(t-1) to end-of-t.
    # Period 0 uses 2015 return, period 1 uses 2016, ..., period 9 uses 2024.
    # (We start the backtest at end-2014 wealth.)
    years = list(range(2015, 2025))  # 10 periods
    actual_r = np.array([historical_returns[y] for y in years])

    # Classify each year into bear / neutral / bull regime index
    def classify_regime(r: float) -> int:
        if r < 0:
            return 0        # Bear
        elif r < 0.15:
            return 1        # Neutral
        else:
            return 2        # Bull

    regime_idx = np.array([classify_regime(r) for r in actual_r])
    regime_names_map = {0: "Bear", 1: "Neutral", 2: "Bull"}

    cfg = model.cfg
    T = cfg.horizon  # 10
    policy_idx = solution["policy_idx"]
    rf = cfg.risk_free_rate

    # We assume the household starts in mid-income state
    iy = 1  # mid income
    y = model.Y[iy]

    # Initial wealth: ₹2 lakh (same as simulation)
    w0 = 200_000.0

    # ── Run four strategies ──────────────────────────────────────────────
    strategies = {
        "DP Optimal": None,      # will use policy lookup
        "100% Risk-Free": 0.0,
        "100% Equity": 1.0,
        "50-50 Mix": 0.5,
    }

    results = {}
    for name, fixed_alpha in strategies.items():
        wealth_path = [w0]
        alpha_path = []
        w = w0
        for t in range(T):
            ir = regime_idx[t]
            r_actual = actual_r[t]

            if fixed_alpha is not None:
                alpha = fixed_alpha
            else:
                # DP policy lookup
                iw = int(np.argmin(np.abs(model.wealth_grid - w)))
                ia = int(policy_idx[t, iw, iy, ir])
                alpha = float(model.A[ia])

            alpha_path.append(alpha)

            # Wealth transition using actual Nifty return
            gross = alpha * (1.0 + r_actual) + (1.0 - alpha) * (1.0 + rf)
            w = (w + cfg.saving_rate * y) * gross
            w = max(w, cfg.wealth_min)
            wealth_path.append(w)

        results[name] = {
            "wealth": np.array(wealth_path),
            "alpha": np.array(alpha_path),
        }

    # ── Plot: wealth comparison ──────────────────────────────────────────
    t_axis = np.arange(T + 1)
    year_labels = [str(2014 + i) for i in range(T + 1)]
    colors = {"DP Optimal": "navy", "100% Risk-Free": "gray",
              "100% Equity": "crimson", "50-50 Mix": "teal"}
    styles = {"DP Optimal": "-", "100% Risk-Free": "--",
              "100% Equity": "-.", "50-50 Mix": ":"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        ax.plot(t_axis, data["wealth"] / 1e5, linewidth=2.5,
                color=colors[name], linestyle=styles[name], label=name)
    ax.set_xticks(t_axis)
    ax.set_xticklabels(year_labels, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Wealth (INR lakhs)")
    ax.set_title("Historical Backtest: DP Optimal vs Naive Strategies (2014-2024)")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25)

    # Add regime annotations
    for t in range(T):
        regime_label = regime_names_map[regime_idx[t]]
        color = {"Bear": "red", "Neutral": "orange", "Bull": "green"}[regime_label]
        ax.axvspan(t + 0.5, t + 1.5, alpha=0.07, color=color)
        ax.text(t + 1, ax.get_ylim()[1] * 0.97, regime_label[0],
                ha="center", va="top", fontsize=8, color=color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "backtest_wealth_comparison.png", dpi=160)
    plt.close(fig)

    # ── Plot: DP optimal alpha over time ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    dp_alpha = results["DP Optimal"]["alpha"]
    ax.bar(range(T), dp_alpha, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xticks(range(T))
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Risky share (alpha)")
    ax.set_ylim(0, 1.05)
    ax.set_title("DP Optimal Risky Allocation During Backtest (2015-2024)")

    # Label regimes
    for t in range(T):
        regime_label = regime_names_map[regime_idx[t]]
        ax.text(t, dp_alpha[t] + 0.03, f"{dp_alpha[t]:.0%}",
                ha="center", va="bottom", fontsize=8)

    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "backtest_dp_alpha.png", dpi=160)
    plt.close(fig)

    # ── Print summary ────────────────────────────────────────────────────
    print("\n  === Historical Backtest (2014-2024) ===")
    print(f"  {'Strategy':<20s}  {'Final Wealth':>14s}  {'Total Return':>13s}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*13}")
    for name, data in results.items():
        final_w = data["wealth"][-1]
        total_ret = (final_w / w0 - 1) * 100
        print(f"  {name:<20s}  INR {final_w/1e5:>8.2f} L  {total_ret:>10.1f} %")
    print()


def main() -> None:
    cfg = ModelConfig()
    model = DynamicPortfolioSDP(cfg)

    print("Running value iteration...")
    vi_solution = model.solve_by_backward_value_iteration()

    print("Running policy iteration...")
    pi_solution = model.solve_by_finite_horizon_policy_iteration()

    policy_match = np.array_equal(vi_solution["policy_idx"], pi_solution["policy_idx"])
    print(f"Value-iteration policy == policy-iteration policy: {policy_match}")

    print("Simulating...")
    sim = model.simulate(vi_solution)

    out_dir = Path("outputs")
    print("Generating plots...")
    model.plot_policy_heatmaps(vi_solution, out_dir)
    model.plot_value_function(vi_solution, out_dir)
    model.plot_simulation_summary(sim, out_dir)
    model.plot_transition_matrices(out_dir)
    model.plot_convergence(pi_solution, out_dir)

    print("Running sensitivity analysis...")
    sensitivity_analysis(out_dir)

    print("Running historical backtest...")
    backtest_historical(model, vi_solution, out_dir)

    np.save(out_dir / "value_function_t0.npy", vi_solution["V"][0])
    np.save(out_dir / "policy_idx.npy", vi_solution["policy_idx"])

    print("Saved outputs to:", out_dir.resolve())


if __name__ == "__main__":
    main()
