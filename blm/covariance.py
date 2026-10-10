"""
Covariance of daily returns.
"""
import pandas as pd
from sklearn.covariance import LedoitWolf


def shrunk_covariance(returns):
    """Ledoit-Wolf shrinkage: with more stocks than days the sample covariance cannot be inverted,
    so it is pulled toward a simple structure by an amount chosen from the data."""
    estimator = LedoitWolf().fit(returns.values)
    return pd.DataFrame(estimator.covariance_, index=returns.columns, columns=returns.columns)


def sample_covariance(returns):
    return returns.cov()
