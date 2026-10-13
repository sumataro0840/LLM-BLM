"""
Strategies: each one takes the inputs of a decision date and returns the weights of the universe.
"""
import pandas as pd

from blm import config
from blm.black_litterman import posterior_returns, posterior_returns_original
from blm.covariance import sample_covariance, shrunk_covariance
from blm.equilibrium import capm_returns_original, implied_returns
from blm.optimizer import optimize, optimize_original


def market_cap(inputs):
    """Market-cap weights: the S&P500 as it is (the benchmark)."""
    return inputs.market_weights


def equal_weight(inputs):
    universe = inputs.market_weights.index
    return pd.Series(1 / len(universe), index=universe)


def mean_variance(inputs):
    """Expected returns = average of the past returns (no LLM)."""
    covariance = shrunk_covariance(inputs.returns)
    expected = inputs.returns.mean() - inputs.risk_free_daily
    return optimize(expected, covariance, config.RISK_AVERSION)


def black_litterman(inputs):
    """Market equilibrium corrected by the LLM views."""
    covariance = shrunk_covariance(inputs.returns)
    prior = implied_returns(covariance, inputs.market_weights, config.RISK_AVERSION)
    views = inputs.view_mean - inputs.risk_free_daily      # the LLM predicts returns, the prior is in excess returns
    expected = posterior_returns(prior, covariance, views, inputs.view_variance, config.TAU)
    return optimize(expected, covariance, config.RISK_AVERSION)


def black_litterman_original(inputs):
    """The calculation of the original implementation, on the same universe and views:
    covariance from the decision month only, yearly risk-free rate subtracted from daily returns,
    and scipy's optimizer with its default settings."""
    covariance = sample_covariance(inputs.month_returns)
    prior = capm_returns_original(inputs.month_returns, inputs.market_weights, config.ORIGINAL_RISK_FREE_RATE)
    expected = posterior_returns_original(prior, covariance, inputs.view_mean, inputs.view_variance, config.TAU)
    return optimize_original(expected, covariance, config.ORIGINAL_RETURN_WEIGHT)


def mean_variance_original(inputs):
    """The mean-variance baseline of the original implementation: mean and covariance of the
    decision month only, and scipy's optimizer with its default settings."""
    covariance = sample_covariance(inputs.month_returns)
    return optimize_original(inputs.month_returns.mean(), covariance, config.ORIGINAL_RETURN_WEIGHT)


COMMON_PARAMS = {'cov_window_days': config.COV_WINDOW_DAYS, 'min_history_ratio': config.MIN_HISTORY_RATIO}

# name -> (function, uses the LLM views, coefficients stored with the portfolio)
STRATEGIES = {
    'market_cap': (market_cap, False, {}),
    'equal_weight': (equal_weight, False, {}),
    'mean_variance': (mean_variance, False, {'risk_aversion': config.RISK_AVERSION, 'covariance': 'ledoit_wolf'}),
    'black_litterman': (black_litterman, True, {'tau': config.TAU, 'risk_aversion': config.RISK_AVERSION,
                                                'covariance': 'ledoit_wolf'}),
    'mean_variance_original': (mean_variance_original, False, {
        'return_weight': config.ORIGINAL_RETURN_WEIGHT, 'covariance': 'sample (decision month)'}),
    'black_litterman_original': (black_litterman_original, True, {
        'tau': config.TAU, 'risk_free_rate': config.ORIGINAL_RISK_FREE_RATE,
        'return_weight': config.ORIGINAL_RETURN_WEIGHT, 'covariance': 'sample (decision month)'}),
}
