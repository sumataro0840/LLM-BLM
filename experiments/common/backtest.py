"""
Monthly backtest with trading costs: weights are decided on each decision date and held to the end of the next month.

The weights held before trading are last month's weights moved by their returns ("drifted"),
so the turnover counts what has to be traded, including stocks that left the universe.
"""
import numpy as np
import pandas as pd

COST_ONE_WAY = 0.001   # 0.1% of the traded amount (the second version of the paper)


def run(strategy, all_data, cost=COST_ONE_WAY, lagged=False):
    """strategy(date, data, history) -> (weights Series over the universe, dict of extra figures).

    history: list of the rows already produced (so a strategy can only see the past).
    lagged: buy one trading day after the decision date (sensitivity to the trade at the decision close).
    Returns one row per decision date.
    """
    rows, held = [], None
    for d in sorted(all_data):
        x = all_data[d]
        # the strategy never sees the outcomes (fields named next_*)
        weights, extra = strategy(d, {k: v for k, v in x.items() if not k.startswith('next_')}, rows)
        weights = weights.reindex(x['universe']).fillna(0.0)
        returns = (x['next_return_lagged'] if lagged else x['next_return']).reindex(weights.index)
        # the portfolio starts from the benchmark
        before = x['market_weights'] if held is None else held
        first_day = 0.0
        if lagged:
            # until the trade one day later, last month's holdings are kept and earn that day's return
            r1 = x['next_daily'].iloc[0].reindex(before.index).fillna(0.0)
            first_day = float((before * r1).sum())
            moved = before * (1 + r1)
            before = moved / moved.sum()
        # a stock without a price at the end of the period (missing data) is counted as 0% rather than dropped
        gross = (1 + first_day) * (1 + float((weights * returns.fillna(0.0)).sum())) - 1
        index = weights.index.union(before.index)
        traded = float((weights.reindex(index).fillna(0) - before.reindex(index).fillna(0)).abs().sum())
        net = gross - cost * traded
        grown = weights * (1 + returns.fillna(0.0))
        held = grown / grown.sum() if grown.sum() > 0 else weights
        bench = x['market_weights']
        rows.append({
            'date': d, 'gross': gross, 'net': net, 'turnover': traded / 2,
            'effective_n': float(1 / (weights ** 2).sum()),
            'active_share': float(0.5 * (weights - bench).abs().sum()),
            'active_beta': float(((weights - bench) * x['beta'].reindex(weights.index)).sum()),
            'max_active_weight': float((weights - bench).abs().max()),
            'risk_free': x['next_risk_free'],
            'weights': weights,
            **extra,
        })
    return pd.DataFrame(rows).set_index('date')


def market_cap_strategy(d, x, history):
    return x['market_weights'], {}
