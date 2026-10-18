"""
E3's analysis on synthetic data only: the analyst variables are prepared as written, a planted LLM effect survives the
analyst control, and an LLM score that only mirrors the analyst surprise loses its coefficient once that is controlled.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    'e3_run', Path(__file__).resolve().parents[1] / 'experiments' / 'E3_analyst_surprise' / 'run.py')
e3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e3)
from tests.test_e2_run import synthetic


def with_surprise(n=20000, llm_own=0.002, seed=3):
    d = synthetic(n=n, effect=0.0, seed=seed)
    rng = np.random.default_rng(seed)
    surprise = rng.standard_t(3, n) * 10                     # heavy tails, in percent
    surprise[rng.random(n) < 0.03] = np.nan                    # a few releases without an estimate
    d['surprise_pct'] = surprise
    d['surprise'] = surprise / 100
    s = np.nan_to_num((surprise - np.nanmean(surprise)) / np.nanstd(surprise))
    own = rng.normal(size=n)
    d['llm_composite'] = own + s                               # the LLM reads the surprise and something else
    d['abn_r_0_1'] = llm_own * own + 0.004 * s + rng.normal(0, 0.03, n)
    return d.reset_index(drop=True)


class TestE3(unittest.TestCase):
    def test_prepare(self):
        A = with_surprise(n=3000)
        d = e3.prepare(A, 'llm_composite', 'abn_r_0_1')
        self.assertEqual(d.analyst_missing.sum(), A.surprise_pct.isna().sum())
        self.assertEqual(d.analyst_surprise.isna().sum(), 0)
        self.assertAlmostEqual(d.analyst_surprise[d.analyst_missing == 0].mean(), 0, delta=0.05)
        r = e3.prepare(A, 'llm_composite', 'abn_r_0_1', rank=True)
        self.assertTrue(r.analyst_surprise.between(0, 1).all())

    def test_own_information_survives(self):
        A = with_surprise(llm_own=0.002)
        d = e3.prepare(A, 'llm_composite', 'abn_r_0_1')
        m1 = e3.fit(d, 'llm_composite', e3.M1)
        self.assertLess(m1['p_one_sided'], 0.01)
        self.assertGreater(m1['others']['analyst_surprise']['t'], 3)

    def test_mirror_only_is_absorbed(self):
        A = with_surprise(llm_own=0.0)
        d = e3.prepare(A, 'llm_composite', 'abn_r_0_1')
        m0 = e3.fit(d, 'llm_composite', e3.BASE)
        m1 = e3.fit(d, 'llm_composite', e3.M1)
        self.assertLess(m0['p_one_sided'], 0.01)                # without the analyst control the score looks useful
        self.assertGreater(m1['p_one_sided'], 0.01)             # with it, the score adds nothing


if __name__ == '__main__':
    unittest.main()
