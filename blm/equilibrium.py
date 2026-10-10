"""
Market equilibrium returns (the prior of Black-Litterman).
"""


def implied_returns(covariance, market_weights, risk_aversion):
    """Excess returns that make the market-cap weights optimal: pi = delta x Sigma x w_market."""
    return risk_aversion * covariance @ market_weights


def capm_returns_original(returns, market_weights, risk_free_rate):
    """The original implementation: beta x mean(market return - risk_free_rate),
    where the yearly risk-free rate is subtracted from daily returns without conversion."""
    market = (returns * market_weights).sum(axis=1)
    beta = returns.apply(lambda x: x.cov(market)) / market.var()
    return beta * (market - risk_free_rate).mean()
