"""
Statistics of the event study (設計.md, 9): OLS with standard errors clustered by event date,
a permutation control that shuffles a score among the releases of the same date, and Holm's correction.

Written with numpy only, so that every number can be checked by hand (tests/test_events_evaluate.py).
"""
import numpy as np
import pandas as pd
from scipy import stats


def design(df, x_cols, fixed_effects=()):
    """Regressors: a constant, x_cols and dummies of the fixed effects (first level dropped)."""
    parts = [pd.Series(1.0, index=df.index, name="const"), df[x_cols].astype(float)]
    for fe in fixed_effects:
        parts.append(pd.get_dummies(df[fe].astype(str), prefix=fe, drop_first=True, dtype=float))
    return pd.concat(parts, axis=1)


def ols_clustered(y, X, clusters):
    """OLS with cluster-robust standard errors (CR1: G/(G-1) * (N-1)/(N-K)). Returns a frame per regressor."""
    y = np.asarray(y, dtype=float)
    Xm = np.asarray(X, dtype=float)
    n, k = Xm.shape
    xtx_inv = np.linalg.pinv(Xm.T @ Xm)
    beta = xtx_inv @ Xm.T @ y
    resid = y - Xm @ beta
    groups = pd.Series(np.asarray(clusters)).astype(str)
    meat = np.zeros((k, k))
    for _, idx in groups.groupby(groups).groups.items():
        s = Xm[list(idx)].T @ resid[list(idx)]
        meat += np.outer(s, s)
    g = groups.nunique()
    factor = g / (g - 1) * (n - 1) / (n - k)
    cov = factor * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=g - 1)
    names = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(k)]
    return pd.DataFrame({"coef": beta, "se": se, "t": t, "p": p}, index=names).assign(n=n, clusters=g)


def regress(df, y, x_cols, fixed_effects=(), cluster="event_date"):
    """Drop rows with a missing value, then OLS with standard errors clustered by `cluster`."""
    data = df.dropna(subset=[y, *x_cols, *fixed_effects, cluster])
    return ols_clustered(data[y], design(data, x_cols, fixed_effects), data[cluster])


def permutation_test(df, y, focal, x_cols, fixed_effects=(), cluster="event_date", n=1000, seed=0):
    """Shuffle the focal score among the releases of the same event date and refit n times.
    Returns the observed coefficient, the shuffled ones and the one-sided p value (share of shuffles >= observed)."""
    data = df.dropna(subset=[y, focal, *x_cols, *fixed_effects, cluster]).reset_index(drop=True)
    X = design(data, [focal, *x_cols], fixed_effects)
    observed = ols_clustered(data[y], X, data[cluster]).loc[focal, "coef"]
    rng = np.random.default_rng(seed)
    groups = [np.asarray(idx) for idx in data.groupby(cluster).indices.values()]
    shuffled = []
    col = X.columns.get_loc(focal)
    values = X[focal].to_numpy()
    Xm = X.to_numpy(dtype=float).copy()
    xtx_inv_cache = None
    for _ in range(n):
        v = values.copy()
        for idx in groups:
            v[idx] = rng.permutation(v[idx])
        Xm[:, col] = v
        xtx_inv_cache = np.linalg.pinv(Xm.T @ Xm)
        shuffled.append((xtx_inv_cache @ Xm.T @ data[y].to_numpy(dtype=float))[col])
    shuffled = np.array(shuffled)
    return {"observed": observed, "shuffled": shuffled, "p_one_sided": (1 + np.sum(shuffled >= observed)) / (n + 1)}


def holm(p_values):
    """Holm's step-down adjusted p values, in the order given."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted
