"""
Clustered standard errors match a direct computation, the permutation keeps shuffles inside a date,
Holm's correction and the reaction windows are right (synthetic data only: no real outcomes are looked at).
"""
import unittest

import numpy as np
import pandas as pd

from events.evaluate import holm, ols_clustered, permutation_test, regress
from events.returns import reactions, window_return


class TestClustered(unittest.TestCase):
    def test_matches_direct_formula(self):
        rng = np.random.default_rng(0)
        n, g = 120, 12
        clusters = np.repeat(np.arange(g), n // g)
        shock = rng.normal(0, 1, g)[clusters]          # a common shock per date
        x = rng.normal(0, 1, n)
        y = 0.5 * x + shock + rng.normal(0, 1, n)
        X = pd.DataFrame({"const": 1.0, "x": x})
        out = ols_clustered(y, X, clusters)
        # direct computation
        Xm = X.to_numpy()
        beta = np.linalg.solve(Xm.T @ Xm, Xm.T @ y)
        u = y - Xm @ beta
        bread = np.linalg.inv(Xm.T @ Xm)
        meat = sum(np.outer(Xm[clusters == c].T @ u[clusters == c], Xm[clusters == c].T @ u[clusters == c]) for c in range(g))
        cov = g / (g - 1) * (n - 1) / (n - 2) * bread @ meat @ bread
        np.testing.assert_allclose(out.coef.to_numpy(), beta)
        np.testing.assert_allclose(out.se.to_numpy(), np.sqrt(np.diag(cov)))
        self.assertEqual(out.clusters.iloc[0], g)

    def test_clustered_is_wider_with_common_shocks(self):
        rng = np.random.default_rng(1)
        n, g = 400, 20
        clusters = np.repeat(np.arange(g), n // g)
        x = rng.normal(0, 1, g)[clusters] + 0.1 * rng.normal(0, 1, n)   # the score also moves by date
        y = rng.normal(0, 1, g)[clusters] + rng.normal(0, 1, n)
        X = pd.DataFrame({"const": 1.0, "x": x})
        clustered = ols_clustered(y, X, clusters).loc["x", "se"]
        each = ols_clustered(y, X, np.arange(n)).loc["x", "se"]
        self.assertGreater(clustered, 1.5 * each)

    def test_regress_with_fixed_effects_and_missing(self):
        df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6, np.nan], "x": [1, 2, 3, 5, 4, 6, 7],
                           "sector": list("aabbcca"), "event_date": list("1122334")})
        out = regress(df, "y", ["x"], ["sector"])
        self.assertEqual(out.n.iloc[0], 6)
        self.assertIn("sector_b", out.index)


class TestPermutation(unittest.TestCase):
    def test_signal_is_found_and_noise_is_not(self):
        rng = np.random.default_rng(2)
        n = 600
        dates = np.repeat(np.arange(60), 10)
        score = rng.normal(0, 1, n)
        df = pd.DataFrame({"event_date": dates, "score": score, "control": rng.normal(0, 1, n)})
        df["y"] = 0.3 * df.score + rng.normal(0, 1, n)
        df["noise"] = rng.normal(0, 1, n)
        strong = permutation_test(df, "y", "score", ["control"], n=200, seed=0)
        weak = permutation_test(df, "noise", "score", ["control"], n=200, seed=0)
        self.assertLess(strong["p_one_sided"], 0.01)
        self.assertGreater(weak["p_one_sided"], 0.05)

    def test_shuffles_stay_inside_dates(self):
        # when every date has one value only, shuffling inside dates changes nothing
        df = pd.DataFrame({"event_date": np.repeat(np.arange(30), 5), "score": np.repeat(np.arange(30.0), 5)})
        df["y"] = df.score + np.random.default_rng(3).normal(0, 1, len(df))
        out = permutation_test(df, "y", "score", [], n=20)
        np.testing.assert_allclose(out["shuffled"], out["observed"])


class TestHolm(unittest.TestCase):
    def test_known_values(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class TestReactions(unittest.TestCase):
    def setUp(self):
        self.days = [f"2024-01-{d:02d}" for d in range(2, 12)]
        self.closes = pd.DataFrame({"AAA": np.linspace(100, 109, 10), "XLK": np.full(10, 50.0),
                                    "^GSPC": np.linspace(1000, 1009, 10)}, index=self.days)

    def test_window_runs_from_the_close_before(self):
        # event on day 5 (2024-01-07): window (0, 1) = close of day 4 -> close of day 6
        r = window_return(self.closes, "AAA", self.days, 5, (0, 1))
        self.assertAlmostEqual(r, 106 / 104 - 1)

    def test_abnormal_against_sector_etf_or_market(self):
        rel = pd.DataFrame({"accession": ["a", "b"], "symbol": ["AAA", "AAA"], "event_date": ["2024-01-07", None]})
        out = reactions(rel, self.closes, self.days, {"AAA": "XLK"}).set_index("accession")
        self.assertAlmostEqual(out.loc["a", "abn_r_0_1"], 106 / 104 - 1)     # the ETF did not move
        self.assertEqual(out.loc["a", "benchmark"], "XLK")
        self.assertTrue(np.isnan(out.loc["b", "abn_r_0_1"]))
        no_etf = reactions(rel.head(1), self.closes, self.days, {"AAA": None}).iloc[0]
        self.assertEqual(no_etf.benchmark, "^GSPC")
        self.assertAlmostEqual(no_etf.abn_r_0_1, (106 / 104 - 1) - (1006 / 1004 - 1))


if __name__ == "__main__":
    unittest.main()
