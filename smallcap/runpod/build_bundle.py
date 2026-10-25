"""
Bundle for the ChronoGPT scoring of a small-cap sample on a rented GPU (RunPod), without the database or any secret.
Same layout and checks as experiments/F_fixes/runpod/build_bundle.py.

    python smallcap/runpod/build_bundle.py --db data/smallcap_rep.sqlite --name rep          # the replication sample
    python smallcap/runpod/build_bundle.py --db data/smallcap.sqlite --limit 20 --name test  # a check against the PC

Writes data/cache/runpod/smallcap_<name>.tar.gz (outside Git) with:
  LLM-BLM/  the code the two ChronoGPT steps need, the release texts, the release list (data/cache/smallcap_releases.csv,
            the same rows and order as smallcap/score.py's releases()) and a .env that only names a dummy database
  LLM/chrono/chrono.py   ChronoGPT's loader (the weights are downloaded on the pod and checked against meta.json)
  run.sh, score_chrono.py
"""
import argparse
import shutil
import sqlite3
import sys
import tarfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
OUT = REPO / 'data' / 'cache' / 'runpod'
CODE = ['events/__init__.py', 'events/config.py', 'events/texts.py', 'events/scorers/__init__.py',
        'events/scorers/chrono.py', 'experiments/__init__.py', 'experiments/F_fixes/__init__.py',
        'experiments/F_fixes/score.py', 'experiments/F_fixes/clean.py']


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--db', required=True)
    a.add_argument('--name', required=True)
    a.add_argument('--limit', type=int)
    args = a.parse_args()
    from events import config
    conn = sqlite3.connect(f'file:{REPO / args.db}?mode=ro', uri=True)
    rel = pd.read_sql('SELECT accession, filing_date, text_path, accepted_et FROM earnings_releases '
                      'WHERE text_path IS NOT NULL ORDER BY filing_date, accession', conn)
    if args.limit:
        rel = rel.head(args.limit)
    stage = OUT / f'bundle_{args.name}'
    if stage.exists():
        shutil.rmtree(stage)
    repo = stage / 'LLM-BLM'
    for f in CODE:
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, repo / f)
    for p in rel.text_path:
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / p, repo / p)
    (repo / 'data' / 'cache').mkdir(parents=True, exist_ok=True)
    rel.to_csv(repo / 'data' / 'cache' / 'smallcap_releases.csv', index=False)
    (repo / '.env').write_text('DB_PATH=data/unused.sqlite\n', encoding='utf-8')     # events/config.py needs a name
    chrono = stage / 'LLM' / 'chrono'
    chrono.mkdir(parents=True)
    shutil.copy2(config.CHRONO_DIR / 'chrono.py', chrono / 'chrono.py')
    shutil.copy2(config.CHRONO_DIR / 'models' / '2015' / 'meta.json', stage / 'chrono_2015_meta_expected.json')
    for f in ('run.sh', 'score_chrono.py'):
        shutil.copy2(HERE / f, stage / f)
    target = OUT / f'smallcap_{args.name}.tar.gz'
    with tarfile.open(target, 'w:gz') as tar:
        tar.add(stage, arcname='bundle')
    print(f'{len(rel)} releases, {target} ({target.stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
