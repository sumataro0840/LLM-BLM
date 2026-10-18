"""
E5's rolling ridge on synthetic data only: a quarter's prediction never uses a release whose window had not closed
before the quarter began, and a linear signal in the vectors is recovered out of sample.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    'e5_run', Path(__file__).resolve().parents[1] / 'experiments' / 'E5_chrono_embeddings' / 'run.py')
e5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e5)


def synthetic(n=4000, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2015-01-01', '2020-12-31')
    accepted = np.sort(rng.choice(days, n))
    frame = pd.DataFrame({'accession': [f'a{i}' for i in range(n)],
                          'accepted_et': pd.DatetimeIndex(accepted).strftime('%Y-%m-%dT16:05:00'),
                          'window_end': (pd.DatetimeIndex(accepted) + pd.offsets.BDay(2)).strftime('%Y-%m-%d')})
    X = rng.normal(size=(n, dim)).astype(np.float32)
    beta = rng.normal(size=dim) / np.sqrt(dim)
    frame['abn_r_0_1'] = 0.01 * (X @ beta) + rng.normal(0, 0.02, n)
    return frame, X


class TestE5(unittest.TestCase):
    def test_recovers_signal_out_of_sample(self):
        frame, X = synthetic()
        pred, info = e5.rolling_ridge(frame, X)
        ok = pred.notna().to_numpy()
        self.assertGreater(ok.mean(), 0.8)
        self.assertGreater(np.corrcoef(pred[ok], frame.abn_r_0_1[ok])[0, 1], 0.3)

    def test_no_future_reactions(self):
        """Changing every reaction from a quarter on must not change the predictions of that quarter or earlier."""
        frame, X = synthetic(seed=1)
        pred, _ = e5.rolling_ridge(frame, X)
        cut = '2018-07-01'
        later = frame.window_end >= cut          # windows not closed before the 2018Q3 start
        changed = frame.copy()
        changed.loc[later, 'abn_r_0_1'] = np.random.default_rng(9).normal(size=later.sum())
        pred2, _ = e5.rolling_ridge(changed, X)
        until = (frame.accepted_et.str[:10] < '2018-10-01').to_numpy()   # through 2018Q3
        np.testing.assert_allclose(pred[until].to_numpy(), pred2[until].to_numpy(), equal_nan=True)
        self.assertFalse(np.allclose(pred[~until].to_numpy(), pred2[~until].to_numpy(), equal_nan=True))

    def test_too_few_training_releases_gives_no_prediction(self):
        frame, X = synthetic()
        pred, info = e5.rolling_ridge(frame, X)
        first = pd.PeriodIndex(frame.accepted_et.str[:10], freq='Q') == pd.Period('2015Q1')
        self.assertTrue(pred[first].isna().all())
        self.assertTrue(all(i['n_train'] >= e5.MIN_TRAIN for i in info))


if __name__ == '__main__':
    unittest.main()
