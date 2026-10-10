"""
Long-only, fully invested mean-variance optimization.
"""
import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Daily returns are around 1e-3 and variances around 1e-4; the objective is scaled up so that
# the solver's tolerances are small compared with it
OBJECTIVE_SCALE = 1e4


def optimize(expected_returns, covariance, risk_aversion):
    """Maximize  w.mu - risk_aversion / 2 x w.Sigma.w   subject to  sum(w) = 1, w >= 0."""
    w = cp.Variable(len(expected_returns))
    objective = expected_returns.values @ w - risk_aversion / 2 * cp.quad_form(w, cp.psd_wrap(covariance.values))
    problem = cp.Problem(cp.Maximize(OBJECTIVE_SCALE * objective), [cp.sum(w) == 1, w >= 0])
    problem.solve(solver=cp.CLARABEL)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"optimization failed: {problem.status}")
    weights = np.clip(w.value, 0, None)
    return pd.Series(weights / weights.sum(), index=expected_returns.index)


def optimize_original(expected_returns, covariance, return_weight):
    """The original implementation: minimize  w.Sigma.w - return_weight x w.mu  with scipy's default
    settings, starting from equal weights (kept as it is, to see what it returns)."""
    n = len(expected_returns)
    mu, sigma = expected_returns.values, covariance.values
    result = minimize(
        lambda x: x @ sigma @ x - return_weight * (x @ mu),
        np.ones(n) / n,
        constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'ineq', 'fun': lambda x: x}),
        bounds=tuple((0, 1) for _ in range(n)),
    )
    weights = np.clip(result.x, 0, None)
    return pd.Series(weights / weights.sum(), index=expected_returns.index)
