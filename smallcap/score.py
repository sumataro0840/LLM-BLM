"""
Scores of the small-cap releases that use no stock reaction (設計_小型株.md), on this PC's GPU. Each step skips what is
already done, so an interrupted run continues. Run it detached with smallcap/start_score.ps1.

    python -m smallcap.score dictionary finbert chrono_raw chrono_clean      (or: all, in this order)
    python -m smallcap.score llm      # gpt-4o-mini, releases filed from 2023-11-01 (paid API; not part of all)

dictionary, finbert: the scorers of run_events.py with the settings of the S&P500 runs (score runs 2 and 3 of the main
database), stored in the small-cap database. ChronoGPT, vintage 2015 as in experiment 11: the mean of the last layer of
the raw text (E5's vector) and the three vectors of the cleaned text (mean of the last layer, last token, mean of the
26th layer), under data/cache/smallcap/ (outside Git).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import run_events  # noqa: E402
from events import texts  # noqa: E402
from experiments.F_fixes import score as F  # noqa: E402
from smallcap.build import open_db  # noqa: E402

OUT = REPO / 'data' / 'cache' / 'smallcap'
NOTE = 'small caps (設計_小型株.md)'


def releases(conn):
    return pd.read_sql('SELECT accession, filing_date, text_path, accepted_et FROM earnings_releases '
                       'WHERE text_path IS NOT NULL ORDER BY filing_date, accession', conn)


def official(conn, method):
    run_events.cmd_score(conn, argparse.Namespace(method=method, start=None, end=None, limit=None, note=NOTE,
                                                  i_have_preregistered=False))


def chrono_raw(rel):
    def vec(chrono, model, tok, r):
        return {'mean_last': chrono.embed(model, tok, texts.read(r.text_path, F.RAW_CHARS)).float().numpy()}
    F._run_chrono(rel, OUT / 'chrono_raw_v2015', vec, 'chrono_raw')


def chrono_clean(rel):
    F.OUT = OUT                      # step_chrono writes under F.OUT / chrono_clean_v2015
    F.step_chrono(rel)


STEPS = ['dictionary', 'finbert', 'chrono_raw', 'chrono_clean']


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    names = sys.argv[1:]
    if not names or any(n not in STEPS + ['all', 'llm'] for n in names):
        print(__doc__)
        return
    names = STEPS if 'all' in names else names
    OUT.mkdir(parents=True, exist_ok=True)
    conn = open_db()
    rel = releases(conn)
    print(f'{len(rel)} releases with a text', flush=True)
    for name in names:
        t0 = time.time()
        if name in ('dictionary', 'finbert', 'llm'):
            official(conn, name)
        elif name == 'chrono_raw':
            chrono_raw(rel)
        else:
            chrono_clean(rel)
        print(f'{name} done ({time.time() - t0:.0f}s)', flush=True)
    print('all done', flush=True)


if __name__ == '__main__':
    main()
