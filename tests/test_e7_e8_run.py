"""
E7 / E8 on synthetic data only: the views of a date never use a release whose day 0 is later, the scores of a day
are standardized with earlier days only, and the BLM tilts toward a stock with a positive view.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e7 = _load('e7_run', 'experiments/E7_monthly_blm/run.py')
e8 = _load('e8_run', 'experiments/E8_daily_blm/run.py')


def events():
    return pd.DataFrame({'accession': ['a', 'b', 'c', 'd', 'e'], 'symbol': ['X', 'X', 'Y', 'Z', 'Z'],
                         'event_date': ['2024-01-10', '2024-01-31', '2024-02-01', '2023-09-01', '2024-01-20'],
                         'accepted_at': ['t1', 't2', 't3', 't4', 't5'], 'llm_composite': [1.0, 2.0, 3.0, 4.0, 5.0]})


class TestE7(unittest.TestCase):
    def test_latest_scores_use_only_the_past_window(self):
        s = e7.latest_scores(events(), '2024-01-31', {'X', 'Y', 'Z'})
        self.assertEqual(s.to_dict(), {'Z': 5.0, 'X': 2.0})       # Y is later; Z's 2023-09 release is too old

    def test_blm_tilts_toward_positive_view(self):
        rng = np.random.default_rng(0)
        names = [f's{i}' for i in range(30)]
        r = pd.DataFrame(rng.normal(0, 0.01, (300, 30)), columns=names)
        cov = r.cov()
        w_mkt = pd.Series(1 / 30, index=names)
        z = pd.Series([2.0, -2.0], index=['s0', 's1'])
        w, status = e7.blm_weights(cov, w_mkt, r.std(), z, 0.03)
        self.assertIn(status, ('optimal', 'optimal_inaccurate'))
        self.assertGreater(w['s0'], w_mkt['s0'])
        self.assertLess(w['s1'], w_mkt['s1'])


class TestE8(unittest.TestCase):
    def test_scores_standardized_with_earlier_days_only(self):
        ev = pd.DataFrame({'symbol': [f's{i}' for i in range(e8.MIN_PAST + 2)],
                           'event_date': ['2024-01-01'] * e8.MIN_PAST + ['2024-02-01', '2024-03-01'],
                           'llm_composite': list(np.arange(e8.MIN_PAST, dtype=float)) + [1e6, 7.0]})
        z = e8.standardized_scores(ev, '2024-02-01')
        past = np.arange(e8.MIN_PAST, dtype=float)
        self.assertAlmostEqual(float(z.iloc[0]), (1e6 - past.mean()) / past.std(ddof=1))
        self.assertIsNone(e8.standardized_scores(ev, '2024-01-01'))   # not enough earlier events

    def test_no_views_keeps_the_benchmark(self):
        names = ['a', 'b']
        cov = pd.DataFrame([[1e-4, 0], [0, 1e-4]], index=names, columns=names)
        bench = pd.Series([0.6, 0.4], index=names)
        w, status = e8.daily_weights(cov, pd.Series(0.01, index=names), bench, None)
        self.assertEqual(status, 'no views')
        pd.testing.assert_series_equal(w, bench)



e9 = _load('e9_run', 'experiments/E9_monthly_blm_long/run.py')


class TestE9(unittest.TestCase):
    def test_te_limit_uses_only_past_months(self):
        self.assertEqual(e9.limit_for([]), e9.TARGET)
        few = [{'te_realized': 0.045, 'te_pred': 0.03}] * (e9.MIN_RATIO_MONTHS - 1)
        self.assertEqual(e9.limit_for(few), e9.TARGET)                      # not enough months yet
        hist = [{'te_realized': 0.06, 'te_pred': 0.03}] * 10 + [{'te_realized': 0.045, 'te_pred': 0.03}] * 36
        self.assertAlmostEqual(e9.limit_for(hist), e9.TARGET / 1.5)          # only the last 36 months count


if __name__ == '__main__':
    unittest.main()
