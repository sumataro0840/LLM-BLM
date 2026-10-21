"""
Exploratory (not pre-registered; written after final.py's results were seen): the pre-registered logistic regression
standardizes heavy-tailed features (the analyst surprise has values such as -500 %), so its AUC (0.593 in split B)
is below that of the analyst surprise alone (0.612). Here every feature is replaced by its rank within the training
set (the empirical distribution of the training rows, applied to the test rows) before the same rolling logistic
regression, and the AUC gain of each score is recomputed on the final periods.

    python experiments/F_fixes/explore_rank_logit.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.F_fixes import common as C
from experiments.F_fixes import develop as DV

OUT = Path(__file__).resolve().parent / 'results'


def rank_matrix(train, test, score_cols):
    cols = C.LOGIT_FEATURES + list(score_cols)
    Xtr, Xte = [], []
    for c in cols:
        a = train[c].astype(float)
        b = test[c].astype(float)
        ref = np.sort(a.dropna().to_numpy())
        to_rank = lambda v: pd.Series(np.searchsorted(ref, v.to_numpy(), side='right') / len(ref), index=v.index).where(v.notna())
        ra, rb = to_rank(a).fillna(0.5), to_rank(b).fillna(0.5)
        Xtr.append(ra - 0.5)
        Xte.append(rb - 0.5)
        if c in C.LOGIT_FEATURES and a.isna().std() > 0:
            Xtr.append(a.isna().astype(float))
            Xte.append(b.isna().astype(float))
    return np.column_stack(Xtr), np.column_stack(Xte)


def rolling_rank_logit(p, score_cols=()):
    from sklearn.linear_model import LogisticRegression
    q = C.quarters(p)
    pred = pd.Series(np.nan, index=p.accession.to_numpy())
    for quarter in sorted(q.unique()):
        start = quarter.start_time.strftime('%Y-%m-%d')
        train = p[(p.window_end < start) & p.abn_r_0_1.notna()].dropna(subset=list(score_cols))
        test = p[q == quarter].dropna(subset=list(score_cols))
        if len(train) < 500 or not len(test):
            continue
        Xtr, Xte = rank_matrix(train, test, score_cols)
        model = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, (train.abn_r_0_1 > 0).to_numpy())
        pred[test.accession.to_numpy()] = model.predict_proba(Xte)[:, 1]
    return pred


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from experiments.common import data as D
    p = DV.panel_with_scores(D.connect())
    b = pd.read_csv(C.CACHE / 'bert_ft_B.csv').set_index('accession').bert_ft
    p['bert_B'] = p.accession.map(b)
    base = rolling_rank_logit(p)
    scores = {'chrono': ['chrono_base'], 'dictionary': ['dictionary_net_tone'], 'finbert': ['finbert_net_tone'],
              'tfidf': ['tfidf_pred'], 'nlp': ['dictionary_net_tone', 'finbert_net_tone', 'tfidf_pred'],
              'chrono_plus_nlp': ['chrono_base', 'dictionary_net_tone', 'finbert_net_tone', 'tfidf_pred'],
              'bert_ft': ['bert_B'], 'gpt4o': ['llm_composite']}
    out = {}
    for name, cols in scores.items():
        pr = rolling_rank_logit(p, cols)
        for period, first in (('B_final', C.final_start('B')), ('gpt_period', C.GPT_START)):
            F = p[p.filing_date >= first]
            G = F.assign(_a=F.accession.map(base), _b=F.accession.map(pr))
            out.setdefault(period, {})[name] = C.auc_diff_ci(G, '_a', '_b')
        print(name, {k: round(v[name]['diff'], 4) for k, v in out.items()}, flush=True)
    (OUT / 'explore_rank_logit.json').write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
