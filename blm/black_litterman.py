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


def posterior_returns_general(prior, covariance, pick, views, omega, tau):
    """prior + tau Sigma P' (P tau Sigma P' + Omega)^-1 (q - P prior) for any view matrix.

    pick: P (K x N, DataFrame with the stocks as columns), views: q (K), omega: Omega (K x K, a full matrix).
    """
    tau_sigma = tau * covariance.values
    p = pick.values
    gap = views.values - p @ prior.values
    adjustment = tau_sigma @ p.T @ np.linalg.solve(p @ tau_sigma @ p.T + omega, gap)
    return prior + pd.Series(adjustment, index=prior.index)


def posterior_returns_inverse_form(prior, covariance, pick, views, omega, tau):
    """The same posterior in the textbook form [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 prior + P' Omega^-1 q]
    (used to check posterior_returns_general; it needs tau Sigma and Omega to be invertible)."""
    inv_tau_sigma = np.linalg.inv(tau * covariance.values)
    inv_omega = np.linalg.inv(omega)
    p = pick.values
    precision = inv_tau_sigma + p.T @ inv_omega @ p
    mean = np.linalg.solve(precision, inv_tau_sigma @ prior.values + p.T @ inv_omega @ views.values)
    return pd.Series(mean, index=prior.index)


def posterior_returns_proportional(prior, covariance, pick, views, adoption):
    """Posterior with Omega proportional to the views' own covariance, Omega = tau (1/s - 1) P Sigma P'.

    Then tau cancels and every view direction moves by the same share s (the adoption rate):
    prior + s Sigma P' (P Sigma P')^+ (q - P prior). s = 0 keeps the prior, the limit of an infinite Omega.
    """
    if adoption <= 0:
        return prior.copy()
    sigma = covariance.values
    p = pick.values
    gap = views.values - p @ prior.values
    adjustment = adoption * sigma @ p.T @ np.linalg.pinv(p @ sigma @ p.T) @ gap
    return prior + pd.Series(adjustment, index=prior.index)
