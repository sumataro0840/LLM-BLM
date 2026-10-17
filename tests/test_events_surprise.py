"""
The surprise uses the quarter the release reports, a year-earlier quarter, and only history filed before the release.
"""
import unittest

import numpy as np
import pandas as pd

from events.surprise import surprise_of, with_year_ago


def quarters(n=14, filed_lag=30):
    ends = pd.date_range("2021-03-31", periods=n, freq="QE")
    rng = np.random.default_rng(1)
    eps = 1 + 0.05 * np.arange(n) + rng.normal(0, 0.1, n)
    return pd.DataFrame({"symbol": "AAA", "period_end": ends.strftime("%Y-%m-%d"),
                         "filed_date": (ends + pd.Timedelta(days=filed_lag)).strftime("%Y-%m-%d"),
                         "eps": eps, "revenue": 100 * 1.02 ** np.arange(n)})


class TestSurprise(unittest.TestCase):
    def setUp(self):
        self.q = with_year_ago(quarters())

    def test_reports_the_latest_quarter_before_the_release(self):
        # released 20 days after the quarter ended 2024-06-30, before its 10-Q (filed 30 days after)
        s = surprise_of("2024-07-20", self.q)
        self.assertEqual(s["period_end"], "2024-06-30")
        row = self.q[self.q.period_end == "2024-06-30"].iloc[0]
        self.assertAlmostEqual(s["eps"], row.eps)
        self.assertAlmostEqual(s["eps_lag4"], self.q[self.q.period_end == "2023-06-30"].eps.iloc[0])

    def test_history_only_from_filings_before_the_release(self):
        s = surprise_of("2024-07-20", self.q)
        # quarters filed before 2024-07-20 and ending before 2024-06-30, with a year-earlier value: up to 2024-03-31
        known = self.q[(self.q.filed_date < "2024-07-20") & (self.q.period_end < "2024-06-30")].eps_diff.dropna().tail(8)
        target = self.q[self.q.period_end == "2024-06-30"].iloc[0]
        self.assertEqual(s["n_history"], len(known))
        self.assertAlmostEqual(s["sue"], target.eps_diff / known.std(ddof=1))

    def test_too_little_history_gives_no_sue(self):
        s = surprise_of("2022-07-20", self.q)
        self.assertEqual(s["period_end"], "2022-06-30")
        self.assertTrue(np.isnan(s["sue"]))

    def test_no_quarter_within_100_days(self):
        self.assertIsNone(surprise_of("2021-01-10", self.q)["period_end"])


if __name__ == "__main__":
    unittest.main()
