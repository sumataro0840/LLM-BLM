"""
Black-Litterman posterior returns with one absolute view per stock (P = identity).
"""
import numpy as np
import pandas as pd


def posterior_returns(prior, covariance, view_mean, view_variance, tau):
    """prior + tau Sigma (tau Sigma + Omega)^-1 (Q - prior), with Omega = diag(view_variance).

    A stock whose answers agree (small variance) is moved close to its view;
    one whose answers are scattered stays close to the prior.
    """
    tau_sigma = tau * covariance.values
    omega = np.diag(view_variance.values)
    adjustment = tau_sigma @ np.linalg.solve(tau_sigma + omega, (view_mean - prior).values)
    return prior + pd.Series(adjustment, index=prior.index)


def posterior_returns_original(prior, covariance, view_mean, view_variance, tau):
    """The same posterior written as in the original implementation (pseudo-inverses)."""
    inv_tau_sigma = np.linalg.pinv(tau * covariance.values)
    inv_omega = np.linalg.pinv(np.diag(view_variance.values))
    m = np.linalg.pinv(inv_tau_sigma + inv_omega)
    return pd.Series(m @ (inv_tau_sigma @ prior.values + inv_omega @ view_mean.values), index=prior.index)
