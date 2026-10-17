"""
Marcenko-Pastur denoising: the noise density integrates to the right mass, pure noise is recognised as noise,
planted factors are kept, and the result is a valid covariance even with more stocks than days.
"""
import unittest

import numpy as np
import pandas as pd
from scipy.integrate import quad

from blm.covariance import denoised_covariance, fit_noise_variance, marcenko_pastur_pdf


def factor_returns(n_stocks, n_days, n_factors, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0, 1, (n_stocks, n_factors))
    # same factor variance for every stock, so the noise share is the same (the Marcenko-Pastur assumption);
    # with uneven shares the noise bulk spreads past the edge and extra eigenvalues are kept
    loadings *= np.sqrt(n_factors) / np.linalg.norm(loadings, axis=1, keepdims=True)
    factors = rng.normal(0, 0.01, (n_days, n_factors))
    noise = rng.normal(0, 0.01, (n_days, n_stocks))
    return pd.DataFrame(factors @ loadings.T + noise)


class TestDenoise(unittest.TestCase):
    def test_density_mass(self):
        for ratio, mass in [(0.2, 1.0), (2.0, 0.5)]:
            lo, hi = (1 - np.sqrt(ratio)) ** 2, (1 + np.sqrt(ratio)) ** 2
            total = quad(lambda x: marcenko_pastur_pdf(np.array([x]), 1.0, ratio)[0], lo, hi, limit=200)[0]
            self.assertAlmostEqual(total, mass, places=3)

    def test_pure_noise_variance_near_one(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2000, 400))
        values = np.linalg.eigvalsh(np.corrcoef(x, rowvar=False))
        self.assertAlmostEqual(fit_noise_variance(values, 400 / 2000), 1.0, delta=0.1)

    def test_keeps_planted_factors(self):
        out = denoised_covariance(factor_returns(200, 1000, 3))
        self.assertEqual(out.attrs['n_signal'], 3)

    def test_valid_when_more_stocks_than_days(self):
        r = factor_returns(400, 252, 2, seed=2)
        out = denoised_covariance(r)
        self.assertGreater(np.linalg.eigvalsh(out.values).min(), 0)            # invertible
        np.testing.assert_allclose(np.diag(out.values), r.var().values, rtol=1e-8)   # variances kept
        self.assertTrue(np.allclose(out.values, out.values.T))


if __name__ == '__main__':
    unittest.main()
