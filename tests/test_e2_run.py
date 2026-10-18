"""
E2's analysis on synthetic data only (the real reactions are computed once, after the pre-registration):
a planted effect is recovered in basis points, a score with no effect is not supported, the preparation
winsorizes, standardizes and fills as written, and the TF-IDF scorer only learns from windows that closed earlier.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'E2_event_reaction'))
import run as e2

from events.scorers.tfidf import TfidfScorer


def synthetic(n=3000, effect=0.002, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-02', periods=150).strftime('%Y-%m-%d')
    d = pd.DataFrame({'event_date': rng.choice(dates, n)})
    d['year_quarter'] = pd.PeriodIndex(d.event_date, freq='Q').astype(str)
    d['sector'] = rng.choice(list('ABCDE'), n)
    score = rng.normal(size=n)
    d['llm_composite'] = score * 2 + 1                       # standardized inside prepare
    d['placebo'] = rng.normal(size=n)
    for col in ['sue', 'revenue_surprise', 'dictionary_net_tone', 'finbert_net_tone', 'tfidf_pred', 'abn_pre']:
        d[col] = rng.normal(size=n)
    d['log_mcap'] = rng.normal(23, 1, n)
    d.loc[rng.random(n) < 0.1, 'sue'] = np.nan
    day_shock = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    d['abn_r_0_1'] = effect * score + 0.004 * d.sue.fillna(0) + day_shock[d.event_date].values + rng.normal(0, 0.03, n)
    return d


class TestE2(unittest.TestCase):
    def test_recovers_planted_effect(self):
        out = e2.fit(synthetic(n=20000), 'llm_composite', y='abn_r_0_1', permute=False)
        self.assertAlmostEqual(out['coef_bp_per_sd'], 20, delta=3 * out['se_bp'])
        self.assertLess(out['p_one_sided'], 0.01)
        self.assertGreater(out['r2_gain'], 0)
        self.assertIn('revenue_surprise_missing', out['controls_dropped'])   # never missing in this data

    def test_no_effect_not_supported(self):
        with mock.patch.object(e2, 'N_PERMUTATIONS', 200):
            out = e2.fit(synthetic(), 'placebo', y='abn_r_0_1', permute=True)
        self.assertGreater(out['p_one_sided'], 0.05)
        self.assertGreater(out['permutation_p_one_sided'], 0.05)

    def test_prepare(self):
        d = synthetic()
        d.loc[0, 'abn_r_0_1'] = 5.0                           # an outlier
        p = e2.prepare(d.assign(y=d.abn_r_0_1), 'llm_composite')
        self.assertLess(p.y.max(), 1.0)                       # winsorized
        self.assertAlmostEqual(p.llm_composite.mean(), 0, places=8)
        self.assertAlmostEqual(p.llm_composite.std(), 1, places=8)
        self.assertEqual(p.sue.isna().sum(), 0)               # filled
        self.assertEqual(p.sue_missing.sum(), d.sue.isna().sum())
        r = e2.prepare(d.assign(y=d.abn_r_0_1), 'llm_composite', rank=True)
        self.assertTrue(r.y.between(0, 1).all())

    def test_holm_order(self):
        np.testing.assert_allclose(e2.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class TestTfidfUsesOnlyEarlierWindows(unittest.TestCase):
    def test_training_set(self):
        n = 1200
        rng = np.random.default_rng(1)
        days = pd.bdate_range('2020-01-01', periods=n).strftime('%Y-%m-%d')
        words = np.where(rng.random(n) < 0.5, 'growth raised strong', 'decline lowered weak')
        y = np.where(words == 'growth raised strong', 0.02, -0.02) + rng.normal(0, 0.005, n)
        rel = pd.DataFrame({'accession': [f'a{i}' for i in range(n)], 'accepted_et': [f'{d}T08:00:00' for d in days],
                            'text_path': [f'{w} filler text number {i}' for i, w in enumerate(words)]})
        react = pd.DataFrame({'abn_r_0_1': y, 'window_end': days}, index=rel.accession)
        seen = []

        class Spy(TfidfScorer):
            pass

        with mock.patch('events.scorers.tfidf.texts.read', side_effect=lambda p, m=None: p), \
             mock.patch('events.scorers.tfidf.MIN_DF', 1):
            out = {r.accession: r.values for r in Spy(reactions=react).score(rel)}
        first_q_start = pd.Period(days[0], 'Q').start_time.strftime('%Y-%m-%d')
        # releases of the first quarters have too few earlier windows -> no prediction
        self.assertIsNone(out['a0']['predicted_reaction'])
        last = rel.accession.iloc[-1]
        q_start = pd.Period(days[-1], 'Q').start_time.strftime('%Y-%m-%d')
        self.assertEqual(out[last]['n_train'], int((react.window_end < q_start).sum()))
        good = [out[a]['predicted_reaction'] for a, w in zip(rel.accession, words)
                if w == 'growth raised strong' and out[a]['predicted_reaction'] is not None]
        bad = [out[a]['predicted_reaction'] for a, w in zip(rel.accession, words)
               if w != 'growth raised strong' and out[a]['predicted_reaction'] is not None]
        self.assertGreater(np.mean(good), np.mean(bad))


if __name__ == '__main__':
    unittest.main()
