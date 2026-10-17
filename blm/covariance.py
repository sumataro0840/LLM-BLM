"""
Covariance of daily returns.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.covariance import LedoitWolf


def shrunk_covariance(returns):
    """Ledoit-Wolf shrinkage: with more stocks than days the sample covariance cannot be inverted,
    so it is pulled toward a simple structure by an amount chosen from the data."""
    estimator = LedoitWolf().fit(returns.values)
    return pd.DataFrame(estimator.covariance_, index=returns.columns, columns=returns.columns)


def sample_covariance(returns):
    return returns.cov()


def marcenko_pastur_pdf(x, variance, ratio):
    """Density of the eigenvalues of a correlation matrix of pure noise (Marcenko-Pastur), ratio = stocks / days.
    With more stocks than days only the continuous part is returned; it integrates to days / stocks."""
    lo, hi = variance * (1 - np.sqrt(ratio)) ** 2, variance * (1 + np.sqrt(ratio)) ** 2
    x = np.asarray(x, dtype=float)
    inside = (x > lo) & (x < hi)
    pdf = np.zeros_like(x)
    pdf[inside] = np.sqrt((hi - x[inside]) * (x[inside] - lo)) / (2 * np.pi * variance * ratio * x[inside])
    return pdf


def fit_noise_variance(eigenvalues, ratio, bandwidth=0.01):
    """Variance of the noise that makes the Marcenko-Pastur density closest to the observed eigenvalues
    (kernel density of the positive eigenvalues; Lopez de Prado, Machine Learning for Asset Managers, ch. 2)."""
    positive = eigenvalues[eigenvalues > 1e-10]
    share = len(positive) / len(eigenvalues)               # mass of the continuous part (days / stocks when below 1)
    # compare only where noise can be: below the upper edge for the largest possible noise variance (1);
    # a grid up to the market eigenvalue (~100) would leave the noise bulk with a handful of points
    widest_edge = (1 + np.sqrt(ratio)) ** 2
    grid = np.linspace(positive.min(), min(positive.max(), widest_edge), 1000)
    kde = share * gaussian_kde(positive, bw_method=bandwidth / positive.std(ddof=1))(grid)

    def loss(v):
        return np.sum((marcenko_pastur_pdf(grid, v[0], ratio) - kde) ** 2)

    fit = minimize(loss, x0=[0.5], bounds=[(1e-5, 1 - 1e-5)])
    return float(fit.x[0])


def denoised_covariance(returns):
    """Covariance with the noise eigenvalues of the correlation matrix replaced by their average
    (constant residual eigenvalue method of Lopez de Prado, ch. 2):
    eigenvalues above the Marcenko-Pastur upper edge are kept as signal, the others are averaged,
    so the trace is kept and the matrix becomes invertible even with more stocks than days."""
    sd = returns.std()
    corr = returns.corr().values
    values, vectors = np.linalg.eigh(corr)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    ratio = returns.shape[1] / returns.shape[0]
    variance = fit_noise_variance(values, ratio)
    edge = variance * (1 + np.sqrt(ratio)) ** 2
    n_signal = max(1, int(np.sum(values > edge)))
    cleaned = values.copy()
    cleaned[n_signal:] = values[n_signal:].sum() / (len(values) - n_signal)
    corr1 = vectors @ np.diag(cleaned) @ vectors.T
    d = np.sqrt(np.diag(corr1))
    corr1 = corr1 / np.outer(d, d)
    cov = corr1 * np.outer(sd.values, sd.values)
    out = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    out.attrs.update(n_signal=n_signal, noise_variance=variance, edge=edge)
    return out
