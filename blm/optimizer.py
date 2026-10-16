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


def optimize_with_tracking_error(expected_returns, covariance, risk_aversion, benchmark, max_tracking_error):
    """optimize() with a limit on the ex-ante tracking error against the benchmark weights:
    (w - b) Sigma (w - b) <= max_tracking_error^2 (in the unit of the covariance, e.g. daily).

    Returns (weights, solver status); the status is kept so that a run can record whether every date was solved.
    """
    w = cp.Variable(len(expected_returns))
    sigma = cp.psd_wrap(covariance.values)
    objective = expected_returns.values @ w - risk_aversion / 2 * cp.quad_form(w, sigma)
    constraints = [cp.sum(w) == 1, w >= 0,
                   cp.quad_form(w - benchmark.reindex(expected_returns.index).values, sigma) <= max_tracking_error ** 2]
    problem = cp.Problem(cp.Maximize(OBJECTIVE_SCALE * objective), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"optimization failed: {problem.status}")
    weights = np.clip(w.value, 0, None)
    return pd.Series(weights / weights.sum(), index=expected_returns.index), problem.status
