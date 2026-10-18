"""
Alpha Vantage answers: quarters are parsed with missing values as None, the allowance message is not cached,
share-class symbols are converted, and releases are paired with the reported date nearest to their acceptance.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from events import estimates

ANSWER = {"symbol": "IBM", "quarterlyEarnings": [
    {"fiscalDateEnding": "2026-06-30", "reportedDate": "2026-07-22", "reportedEPS": "2.93", "estimatedEPS": "2.80",
     "surprise": "0.13", "surprisePercentage": "4.64", "reportTime": "post-market"},
    {"fiscalDateEnding": "2026-03-31", "reportedDate": "2026-04-21", "reportedEPS": "1.60", "estimatedEPS": "None",
     "surprise": "None", "surprisePercentage": "None", "reportTime": ""},
]}


class TestEstimates(unittest.TestCase):
    def test_parse(self):
        rows = estimates.parse("IBM", ANSWER)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows.estimated_eps[0], 2.80)
        self.assertTrue(pd.isna(rows.estimated_eps[1]))      # saved as NULL
        self.assertEqual(rows.report_time[0], "post-market")
        self.assertTrue(pd.isna(rows.report_time[1]))        # unknown time -> NULL
        self.assertTrue(estimates.parse("X", {}).empty)

    def test_provider_symbol(self):
        self.assertEqual(estimates.provider_symbol("BRK-B"), "BRK.B")

    def test_limit_message_not_cached(self):
        class Res:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, *a): return self.body
        with tempfile.TemporaryDirectory() as d, mock.patch.object(estimates.config, "ESTIMATES_DIR", Path(d)), \
                mock.patch("urllib.request.urlopen", return_value=Res(json.dumps({"Information": "limit"}).encode())):
            with self.assertRaises(estimates.LimitReached):
                estimates.fetch("IBM", "k")
            self.assertFalse((Path(d) / "IBM.json").exists())

    def test_match(self):
        est = estimates.parse("IBM", ANSWER)
        rel = pd.DataFrame({"accession": ["a", "b", "c"], "symbol": ["IBM", "IBM", "IBM"],
                            "accepted_et": ["2026-07-22T16:05:00", "2026-04-22T08:00:00", "2025-01-10T08:00:00"],
                            "session": ["post", "pre", "pre"]})
        m = estimates.match_releases(rel, est).set_index("accession")
        self.assertTrue(m.loc["a", "matched"])
        self.assertEqual(m.loc["a", "days_from_acceptance"], 0)
        self.assertEqual(m.loc["b", "days_from_acceptance"], -1)
        self.assertFalse(m.loc["c", "matched"])


if __name__ == "__main__":
    unittest.main()
