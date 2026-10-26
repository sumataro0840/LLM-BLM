"""
The replication sample (R1): a copy of data/smallcap.sqlite with the population, filings, prices and floats kept,
the release tables emptied, and 1,000 new companies drawn (seed 1) from the eligible companies NOT in the original
sample. The original database is only read.

    python -m smallcap.replica            # creates data/smallcap_rep.sqlite (refuses if it exists)
    python -m smallcap.replica g3         # G3 (やること.md): data/smallcap_g3.sqlite with every eligible company in
                                          # neither the original nor the replication sample (seed 2)

Then, with SMALLCAP_DB=data/smallcap_rep.sqlite (and SMALLCAP_CACHE=data/cache/smallcap_rep for the scores):
    python -m smallcap.build releases / fundamentals / surprise
    python -m smallcap.score all ; python -m smallcap.score llm
"""
import random
import shutil
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from smallcap.build import eligible, now_utc  # noqa: E402

SRC = REPO / 'data' / 'smallcap.sqlite'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'rep'
DST = REPO / 'data' / f'smallcap_{TAG}.sqlite'
SEED, N = {'rep': (1, 1000), 'g3': (2, 10 ** 6)}[TAG]
EXCLUDE = {'rep': [SRC], 'g3': [SRC, REPO / 'data' / 'smallcap_rep.sqlite']}[TAG]
EMPTIED = ['earnings_releases', 'release_scores', 'release_score_calls', 'release_score_runs', 'release_surprises',
           'financial_facts', 'financial_items', 'sc_sample']


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if DST.exists():
        raise SystemExit(f'{DST} があります（作り直すなら消してから）')
    original = set()                     # the companies of the samples already used
    for path in EXCLUDE:
        src = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        original |= {r[0] for r in src.execute('SELECT cik FROM sc_sample')}
        src.close()
    shutil.copyfile(SRC, DST)
    conn = sqlite3.connect(DST)
    for t in EMPTIED:
        conn.execute(f'DELETE FROM {t}')
    conn.execute(f"DELETE FROM sqlite_sequence WHERE name IN ({','.join('?' * len(EMPTIED))})", EMPTIED)
    conn.commit()
    e = eligible(conn)
    pool = sorted(set(e.cik) - original)
    pick = set(random.Random(SEED).sample(pool, min(N, len(pool))))
    s = e[e.cik.isin(pick)]
    conn.executemany('INSERT INTO sc_sample VALUES (?, ?, ?, ?)',
                     [(int(r.cik), r.symbol, SEED, now_utc()) for r in s.itertuples()])
    conn.commit()
    overlap = len(set(s.cik) & original)
    print(f'{len(e)} eligible, {len(original)} in the samples already used, {len(pool)} left, {len(s)} drawn '
          f'(seed {SEED}); overlap with the original: {overlap}')
    conn.execute('VACUUM')
    conn.close()


if __name__ == '__main__':
    main()
