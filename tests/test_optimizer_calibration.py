"""
The optimizers return the benchmark without views and respect the tracking-error limit;
the calibration slope recovers a known slope and only uses what it is given.
"""
import unittest

import numpy as np
import pandas as pd

from blm.calibration import adoption_rate, calibration_slope
from blm.optimizer import optimize, optimize_with_tracking_error
from tests.test_black_litterman import random_problem


class TestOptimizer(unittest.TestCase):
    def test_equilibrium_returns_give_market_weights(self):
        covariance, weights, prior, _ = random_problem()
        w = optimize(prior, covariance, 2.5)
        self.assertLess(np.max(np.abs(w - weights)), 1e-4)

    def test_tracking_error_limit(self):
        covariance, weights, prior, rng = random_problem()
        tilted = prior + rng.normal(0, 0.002, len(prior))
        te = 0.03 / np.sqrt(252)
        w, status = optimize_with_tracking_error(tilted, covariance, 2.5, weights, te)
        active = (w - weights).values
        self.assertLessEqual(np.sqrt(active @ covariance.values @ active), te * 1.001)
        self.assertIn(status, ('optimal', 'optimal_inaccurate'))
        # a very loose limit gives the unconstrained answer
        loose, _ = optimize_with_tracking_error(tilted, covariance, 2.5, weights, 10.0)
        self.assertLess(np.max(np.abs(loose - optimize(tilted, covariance, 2.5))), 1e-4)


class TestCalibration(unittest.TestCase):
    def test_recovers_the_slope(self):
        rng = np.random.default_rng(1)
        dates = np.repeat(np.arange(24), 400)
        forecast = rng.normal(0, 1, len(dates)) + np.repeat(rng.normal(0, 3, 24), 400)   # a level that changes by month
        outcome = 0.1 * forecast + rng.normal(0, 1, len(dates)) + np.repeat(rng.normal(0, 3, 24), 400)
        slope, se, n = calibration_slope(forecast, outcome, dates)
        self.assertEqual(n, 24)
        self.assertAlmostEqual(slope, 0.1, delta=0.03)   # the monthly levels are taken out and do not bias it
        self.assertGreater(se, 0)

    def test_adoption_rate_rules(self):
        self.assertEqual(adoption_rate(0.2, n_dates=2), 0.0)            # too few outcomes
        self.assertEqual(adoption_rate(-0.1, n_dates=10), 0.0)          # wrong direction
        self.assertEqual(adoption_rate(2.0, n_dates=10), 0.99)          # capped
        self.assertEqual(adoption_rate(0.05, se=0.06, n_dates=10), 0.0)  # t-value below 1: not used
        self.assertAlmostEqual(adoption_rate(0.1, se=0.05, n_dates=10), 0.1 * 0.75)


if __name__ == '__main__':
    unittest.main()
