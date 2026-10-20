"""
Build the bundle that scores the fix experiments on a rented GPU (RunPod) without the database or any secret:

    python experiments/F_fixes/runpod/build_bundle.py

Writes data/cache/runpod/fixes_bundle.tar.gz (outside Git) with:
  LLM-BLM/  the code that score.py needs, the release texts it reads, the Loughran-McDonald dictionary, the release
            list (data/cache/fixes_releases.csv) and a .env that only names a dummy database path
  LLM/chrono/chrono.py   ChronoGPT's loader (the weights are downloaded on the pod and checked against meta.json)
  run.sh    installs the pinned libraries, downloads ChronoGPT vintage 2015, runs the scoring steps and packs the
            results into /workspace/fixes_out.tar.gz
No .env value of this PC, no API key and no database is copied.
"""
import shutil
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OUT = REPO / 'data' / 'cache' / 'runpod'
CODE = ['events/__init__.py', 'events/config.py', 'events/texts.py', 'events/scorers/__init__.py',
        'events/scorers/dictionary.py', 'events/scorers/finbert.py', 'events/scorers/chrono.py',
        'experiments/__init__.py', 'experiments/F_fixes/__init__.py', 'experiments/F_fixes/score.py',
        'experiments/F_fixes/clean.py']


def main():
    from events import config
    from experiments.F_fixes import score as S
    stage = OUT / 'bundle'
    if stage.exists():
        shutil.rmtree(stage)
    repo = stage / 'LLM-BLM'
    for rel in CODE:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    lm = config.LM_DICTIONARY.relative_to(REPO)
    (repo / lm).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.LM_DICTIONARY, repo / lm)
    rel = S.releases()
    for p in rel.text_path:
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / p, repo / p)
    (repo / 'data' / 'cache' / 'fixes').mkdir(parents=True, exist_ok=True)
    rel.to_csv(repo / 'data' / 'cache' / 'fixes_releases.csv', index=False)
    (repo / '.env').write_text('DB_PATH=data/unused.sqlite\n', encoding='utf-8')     # events/config.py needs a name
    chrono = stage / 'LLM' / 'chrono'
    chrono.mkdir(parents=True)
    shutil.copy2(config.CHRONO_DIR / 'chrono.py', chrono / 'chrono.py')
    shutil.copy2(config.CHRONO_DIR / 'models' / '2015' / 'meta.json', stage / 'chrono_2015_meta_expected.json')
    shutil.copy2(Path(__file__).with_name('run.sh'), stage / 'run.sh')
    target = OUT / 'fixes_bundle.tar.gz'
    with tarfile.open(target, 'w:gz') as tar:
        tar.add(stage, arcname='bundle')
    print(f'{len(rel)} releases, {target} ({target.stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
