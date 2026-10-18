"""
E4's evaluation on synthetic data only: when the fine-tuned model already holds what the LLM score holds, the LLM
coefficient disappears once the prediction is added; when the LLM score has more, it stays.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

from tests.test_e3_run import with_surprise

_spec = importlib.util.spec_from_file_location(
    'e4_run', Path(__file__).resolve().parents[1] / 'experiments' / 'E4_finbert_finetune' / 'run.py')
e4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e4)


def frame(llm_extra, seed=5):
    A = with_surprise(llm_own=0.0, seed=seed)
    rng = np.random.default_rng(seed)
    news = rng.normal(size=len(A))
    extra = rng.normal(size=len(A))
    A['ft_pred'] = news + rng.normal(0, 0.3, len(A))              # the fine-tuned model sees the news
    A['llm_composite'] = news + llm_extra * extra + rng.normal(0, 0.3, len(A))
    A['abn_r_0_1'] = 0.003 * news + 0.003 * llm_extra * extra + rng.normal(0, 0.03, len(A))
    d = e4.e3.prepare(A, 'llm_composite', 'abn_r_0_1')
    d['ft_pred'] = e4.e2.standardize(A['ft_pred'])
    return d


class TestE4(unittest.TestCase):
    def test_llm_absorbed_by_finetuned(self):
        r = e4.fit(frame(llm_extra=0.0), 'llm_composite', e4.e3.M1 + ['ft_pred'])
        self.assertGreater(r['p_one_sided'], 0.01)

    def test_llm_with_more_information_stays(self):
        r = e4.fit(frame(llm_extra=1.0), 'llm_composite', e4.e3.M1 + ['ft_pred'])
        self.assertLess(r['p_one_sided'], 0.001)
        self.assertGreater(r['others']['ft_pred']['t'], 3)


if __name__ == '__main__':
    unittest.main()
