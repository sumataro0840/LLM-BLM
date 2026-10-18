"""
E5 step 1: ChronoGPT's internal representation of every release text (no stock reactions are used here).

Each release is read by the vintage trained on text up to the end of the year before its acceptance (at most 2024),
the same rule as the zero-shot scores (events/scorers/chrono.py). The vector is the mean over tokens of the
normalized last layer (chrono.embed, 1,536 numbers) of the opening of the release (the first 1,792 tokens).
One file per vintage in data/cache/chrono_embeddings/ (outside Git); a vintage already written is skipped,
so an interrupted run continues.

    python experiments/E5_chrono_embeddings/embed.py
"""
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from events import config, texts
from events.scorers.chrono import LATEST_VINTAGE, _chrono
from marketdb.db import events as db
from marketdb.db.connection import connect

OUT = REPO / 'data' / 'cache' / 'chrono_embeddings'
MAX_CHARS = 9000          # more than fits; chrono.embed keeps the first 1,792 tokens


def params(chrono):
    code = inspect.getsource(chrono.embed) + repr(chrono.CONTEXT)
    return {'model': 'manelalab/chrono-gpt-instruct-v1', 'vector': 'mean over tokens of the normalized last layer',
            'max_chars': MAX_CHARS, 'context_tokens': chrono.CONTEXT,
            'vintage': f'end of the year before the acceptance date, at most {LATEST_VINTAGE}',
            'embed_code_sha256': hashlib.sha256(code.encode()).hexdigest()}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    chrono = _chrono()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'params.json').write_text(json.dumps(params(chrono), indent=1), encoding='utf-8')
    rel = db.load_releases(connect(config.DB_PATH), with_text=True)
    rel['vintage'] = [min(chrono.vintage_for(a[:10]), LATEST_VINTAGE) for a in rel.accepted_et]
    print(f'{len(rel)} releases, vintages {sorted(rel.vintage.unique())}', flush=True)
    t0 = time.time()
    model = None
    for year, group in rel.groupby('vintage'):
        path = OUT / f'v{year}.npz'
        if path.exists():
            print(f'  vintage {year}: already done', flush=True)
            continue
        model = None                       # free the previous vintage before loading the next (8 GB GPU)
        model, tok = chrono.load(year)
        vectors = np.zeros((len(group), 1536), dtype=np.float32)
        for i, p in enumerate(group.text_path):
            vectors[i] = chrono.embed(model, tok, texts.read(p, MAX_CHARS)).numpy()
            if (i + 1) % 500 == 0:
                print(f'  vintage {year}: {i + 1}/{len(group)} ({time.time() - t0:.0f}s)', flush=True)
        np.savez_compressed(path, accession=group.accession.to_numpy(), vectors=vectors.astype(np.float16))
        print(f'  vintage {year}: {len(group)} done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
