"""
The backtest counts drifted holdings, turnover and costs as intended, and hides the outcomes from strategies.
"""
import unittest

import pandas as pd

from experiments.common.backtest import run


def toy():
    idx = ['a', 'b']
    day = lambda r: pd.DataFrame([r], columns=idx)
    base = {'universe': idx, 'market_weights': pd.Series([0.5, 0.5], index=idx), 'beta': pd.Series([1.0, 1.0], index=idx),
            'next_risk_free': 0.0}
    return {
        '2024-01-31': {**base, 'next_return': pd.Series([0.10, 0.0], index=idx), 'next_return_lagged': pd.Series([0.10, 0.0], index=idx),
                       'next_daily': day([0.0, 0.0])},
        '2024-02-29': {**base, 'next_return': pd.Series([0.0, 0.0], index=idx), 'next_return_lagged': pd.Series([0.0, 0.0], index=idx),
                       'next_daily': day([0.02, 0.0])},
    }


class TestBacktest(unittest.TestCase):
    def test_drift_turnover_and_cost(self):
        data = toy()
        res = run(lambda d, x, h: (pd.Series([0.5, 0.5], index=x['universe']), {}), data, cost=0.001)
        self.assertAlmostEqual(res.gross.iloc[0], 0.05)
        self.assertAlmostEqual(res.turnover.iloc[0], 0.0)       # starts from the benchmark
        # after the first month the holdings drifted to 0.55/1.05 and 0.5/1.05; back to 50/50
        drifted = 0.55 / 1.05
        self.assertAlmostEqual(res.turnover.iloc[1], abs(0.5 - drifted))
        self.assertAlmostEqual(res.net.iloc[1], res.gross.iloc[1] - 0.001 * 2 * abs(0.5 - drifted))

    def test_strategies_do_not_see_outcomes(self):
        seen = []
        run(lambda d, x, h: (seen.append(sorted(x)) or pd.Series([0.5, 0.5], index=x['universe']), {}), toy())
        self.assertFalse(any(k.startswith('next_') for keys in seen for k in keys))

    def test_lagged_trade_keeps_old_holdings_for_a_day(self):
        data = toy()
        # all in 'a' in January, all in 'b' from February: in February 'a' still earns its first-day 2%
        weights = {'2024-01-31': [1.0, 0.0], '2024-02-29': [0.0, 1.0]}
        res = run(lambda d, x, h: (pd.Series(weights[d], index=x['universe']), {}), data, cost=0.0, lagged=True)
        self.assertAlmostEqual(res.gross.iloc[1], 0.02)


if __name__ == '__main__':
    unittest.main()
