"""
The Black-Litterman formulas agree with each other and keep their units and dimensions.

    python -m unittest discover tests
"""
import unittest

import numpy as np
import pandas as pd

from blm.black_litterman import (posterior_returns, posterior_returns_general, posterior_returns_inverse_form,
                                 posterior_returns_proportional)
from blm.equilibrium import implied_returns


def random_problem(n=40, days=300, seed=0):
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 0.01, days)
    returns = pd.DataFrame(np.outer(factor, rng.uniform(0.5, 1.5, n)) + rng.normal(0, 0.015, (days, n)),
                           columns=[f's{i}' for i in range(n)])
    covariance = returns.cov()
    weights = pd.Series(rng.uniform(1, 10, n), index=covariance.index)
    weights /= weights.sum()
    prior = implied_returns(covariance, weights, 2.5)
    return covariance, weights, prior, rng


class TestPosterior(unittest.TestCase):
    def test_update_form_equals_textbook_form(self):
        covariance, _, prior, rng = random_problem()
        n = len(prior)
        for k in (n, 10, 1):   # absolute views on every stock, 10 random portfolios, one view
            pick = pd.DataFrame(np.eye(n) if k == n else rng.normal(size=(k, n)), columns=prior.index)
            views = pd.Series(pick.values @ prior.values + rng.normal(0, 0.001, k))
            omega = np.diag(rng.uniform(1e-6, 1e-5, k))
            a = posterior_returns_general(prior, covariance, pick, views, omega, 0.025)
            b = posterior_returns_inverse_form(prior, covariance, pick, views, omega, 0.025)
            self.assertEqual(a.shape, (n,))
            self.assertLess(np.max(np.abs(a - b)) / np.max(np.abs(a)), 1e-8)

    def test_existing_function_is_the_identity_case(self):
        covariance, _, prior, rng = random_problem()
        views = prior + rng.normal(0, 0.001, len(prior))
        variance = pd.Series(rng.uniform(1e-6, 1e-5, len(prior)), index=prior.index)
        a = posterior_returns(prior, covariance, views, variance, 0.025)
        b = posterior_returns_general(prior, covariance, pd.DataFrame(np.eye(len(prior)), columns=prior.index),
                                      pd.Series(views.values), np.diag(variance.values), 0.025)
        self.assertLess(np.max(np.abs(a - b)), 1e-12)

    def test_proportional_omega_moves_every_view_by_the_same_share(self):
        covariance, _, prior, rng = random_problem()
        n, s, tau = len(prior), 0.07, 0.025
        pick = pd.DataFrame(rng.normal(size=(8, n)), columns=prior.index)
        views = pd.Series(pick.values @ prior.values + rng.normal(0, 0.001, 8))
        omega = tau * (1 / s - 1) * pick.values @ covariance.values @ pick.values.T
        a = posterior_returns_proportional(prior, covariance, pick, views, s)
        b = posterior_returns_general(prior, covariance, pick, views, omega, tau)
        self.assertLess(np.max(np.abs(a - b)), 1e-12)
        # with P = I the posterior is prior + s (q - prior)
        identity = pd.DataFrame(np.eye(n), columns=prior.index)
        q = prior + rng.normal(0, 0.001, n)
        c = posterior_returns_proportional(prior, covariance, identity, pd.Series(q.values), s)
        self.assertLess(np.max(np.abs(c - (prior + s * (q - prior)))), 1e-12)

    def test_zero_adoption_keeps_the_prior(self):
        covariance, _, prior, rng = random_problem()
        identity = pd.DataFrame(np.eye(len(prior)), columns=prior.index)
        out = posterior_returns_proportional(prior, covariance, identity, pd.Series(prior.values + 0.01), 0.0)
        self.assertTrue(np.allclose(out, prior))

    def test_large_omega_keeps_the_prior_small_omega_takes_the_views(self):
        covariance, _, prior, rng = random_problem()
        views = prior + 0.002
        far = posterior_returns(prior, covariance, views, pd.Series(1e3, index=prior.index), 0.025)
        near = posterior_returns(prior, covariance, views, pd.Series(1e-14, index=prior.index), 0.025)
        self.assertLess(np.max(np.abs(far - prior)), 1e-8)
        self.assertLess(np.max(np.abs(near - views)), 1e-6)


if __name__ == '__main__':
    unittest.main()
