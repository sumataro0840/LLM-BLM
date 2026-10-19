"""
The fix experiments on synthetic data: the cleaning rule, the keep rule, the sector aggregation and the paired t.
"""
import unittest

import numpy as np
import pandas as pd

from experiments.F_fixes import common as C
from experiments.F_fixes import develop as DV
from experiments.F_fixes.clean import clean


class TestClean(unittest.TestCase):
    def test_removes_fragments_and_boilerplate(self):
        text = '\n'.join(['Acme Reports Record Quarter', '· Revenue rose 12 percent to $3.1 billion, ahead of plan.',
                          '2,676', '(0.9', ')', 'U.S.', 'Acme raises full-year guidance to $2.80 to $2.90 per share.',
                          'About Acme:', 'Acme is a global maker of widgets serving customers in 40 countries.',
                          'This release contains forward-looking statements within the meaning of the act.',
                          'Media: Jane Doe, (224) 668-0791', 'Acme will webcast its call at 8 a.m.'])
        c = clean(text)
        self.assertIn('Revenue rose 12 percent', c)
        self.assertIn('raises full-year guidance', c)
        for gone in ['2,676', '(0.9', 'U.S.', 'global maker of widgets', 'forward-looking', '668-0791', 'webcast']:
            self.assertNotIn(gone, c)


class TestKeepRule(unittest.TestCase):
    def test_passes_needs_t_and_both_halves(self):
        ok = {'delta': 0.01, 't': 1.5}
        self.assertTrue(DV.passes(ok, {'delta': 0.0}, {'delta': 0.002}))
        self.assertFalse(DV.passes({'delta': 0.01, 't': 0.8}, ok, ok))         # t below 1
        self.assertFalse(DV.passes(ok, {'delta': -0.001}, ok))                  # worse in one half

    def test_choose_prefers_simpler_when_best_is_not_clearly_better(self):
        table = {('cur', 'a'): 0.01, ('cur', 'b'): 0.012, ('a', 'b'): 0.001}
        t_of = {('cur', 'a'): 2.0, ('cur', 'b'): 2.5, ('a', 'b'): 0.3}

        def compare(x, y):
            full = {'delta': table[(x, y)], 't': t_of[(x, y)]}
            return full, {'delta': 0.0}, {'delta': 0.0}
        self.assertEqual(DV.choose(('current', 'cur'), [('A', 'a'), ('B', 'b')], compare, [])[0], 'A')
        t_of[('a', 'b')] = 1.5                                                   # now the best is clearly better
        self.assertEqual(DV.choose(('current', 'cur'), [('A', 'a'), ('B', 'b')], compare, [])[0], 'B')

    def test_choose_keeps_current_when_nothing_passes(self):
        compare = lambda x, y: ({'delta': -0.01, 't': -1.0}, {'delta': 0}, {'delta': 0})
        self.assertEqual(DV.choose(('current', 'cur'), [('A', 'a')], compare, []), ('current', 'cur'))


class TestSector(unittest.TestCase):
    def test_cap_weighted_view_and_decay(self):
        members = pd.DataFrame({'symbol': [f's{i}' for i in range(40)], 'sector': ['X'] * 20 + ['Y'] * 20,
                                'weight': [1.0] * 40})
        p = pd.DataFrame({'symbol': members.symbol, 'event_date': ['2020-01-20'] * 20 + ['2019-12-01'] * 20,
                          'accepted_at': '', 'score': [1.0] * 20 + [-1.0] * 20})
        v = C.sector_scores(p, 'score', '2020-01-31', members, 'latest92_cap')
        self.assertGreater(v['X'], 0)
        self.assertLess(v['Y'], 0)
        self.assertAlmostEqual(v['X'], -v['Y'])
        w = C.sector_scores(p, 'score', '2020-01-31', members, 'hl30_cap')      # Y is older: shrunk more
        self.assertGreater(abs(w['X']), abs(w['Y']))


class TestPairedT(unittest.TestCase):
    def test_cluster_t(self):
        rng = np.random.default_rng(0)
        clusters = np.repeat(np.arange(200), 5)
        self.assertLess(abs(C.cluster_t_of_mean(rng.normal(0, 1, 1000), clusters)), 3)
        self.assertGreater(C.cluster_t_of_mean(rng.normal(0.3, 1, 1000), clusters), 3)


if __name__ == '__main__':
    unittest.main()
