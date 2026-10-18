"""
E5 step 1: ChronoGPT's internal representation of the release texts (no stock reactions are used here).

A ridge regression trained on earlier releases is only meaningful when the training and the predicted releases are
read by the same model, so each evaluation block uses one vintage for every release (preregistration.md, 3):
    block B  vintage 2015 (text up to 2015-12-31)  releases filed up to 2023-10-31 (sample B is 2016-01 .. 2023-10)
    block A  vintage 2022 (text up to 2022-12-31)  all releases (sample A is 2023-11 .. 2026-08)
The vintage ends before every release it predicts, so neither block looks ahead.
Releases: E2's base sample with a main-window reaction (only whether it exists is used, not its value).
Vector: the mean over tokens of the normalized last layer (chrono.embed, 1,536 numbers) of the first 1,792 tokens.
Saved in chunks to data/cache/chrono_embeddings/v{vintage}/ (outside Git), so an interrupted run continues.

    python experiments/E5_chrono_embeddings/embed.py            # block B, then block A
"""
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from events import config, texts
from events.scorers.chrono import _chrono
from marketdb.db import events as db
from marketdb.db.connection import connect

OUT = REPO / 'data' / 'cache' / 'chrono_embeddings'
PANEL = REPO / 'experiments' / 'E2_event_reaction' / 'results' / 'panel.csv'
BLOCKS = [(2015, '2023-10-31'), (2022, '2026-12-31')]
MAX_CHARS = 9000          # more than fits; chrono.embed keeps the first 1,792 tokens
CHUNK = 1000


def releases():
    p = pd.read_csv(PANEL, usecols=['accession', 'filing_date', 'abn_r_0_1'])
    p = p[p.abn_r_0_1.notna()]
    rel = db.load_releases(connect(config.DB_PATH), with_text=True)[['accession', 'text_path']]
    return p.merge(rel, on='accession').sort_values(['filing_date', 'accession']).reset_index(drop=True)


def params(chrono, vintage, until, n):
    code = inspect.getsource(chrono.embed) + repr(chrono.CONTEXT)
    return {'model': 'manelalab/chrono-gpt-instruct-v1', 'vintage': vintage, 'filed_until': until, 'n_releases': n,
            'vector': 'mean over tokens of the normalized last layer', 'max_chars': MAX_CHARS,
            'context_tokens': chrono.CONTEXT, 'embed_code_sha256': hashlib.sha256(code.encode()).hexdigest()}


def load(vintage):
    """(accessions, float32 matrix) of a finished block."""
    d = OUT / f'v{vintage}'
    parts = [np.load(f, allow_pickle=True) for f in sorted(d.glob('part*.npz'))]
    return (np.concatenate([p['accession'] for p in parts]),
            np.concatenate([p['vectors'] for p in parts]).astype(np.float32))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    chrono = _chrono()
    rel = releases()
    t0 = time.time()
    for vintage, until in BLOCKS:
        group = rel[rel.filing_date <= until].reset_index(drop=True)
        d = OUT / f'v{vintage}'
        d.mkdir(parents=True, exist_ok=True)
        (d / 'params.json').write_text(json.dumps(params(chrono, vintage, until, len(group)), indent=1), encoding='utf-8')
        todo = [s for s in range(0, len(group), CHUNK) if not (d / f'part{s // CHUNK:03d}.npz').exists()]
        print(f'vintage {vintage}: {len(group)} releases, {len(todo)} chunks to do', flush=True)
        if not todo:
            continue
        model = None                       # free the previous vintage before loading the next (8 GB GPU)
        model, tok = chrono.load(vintage)
        for s in todo:
            part = group.iloc[s:s + CHUNK]
            vectors = np.stack([chrono.embed(model, tok, texts.read(p, MAX_CHARS)).float().numpy()
                                for p in part.text_path])
            np.savez_compressed(d / f'part{s // CHUNK:03d}.npz', accession=part.accession.to_numpy(),
                                vectors=vectors.astype(np.float16))
            print(f'  vintage {vintage}: {s + len(part)}/{len(group)} ({time.time() - t0:.0f}s)', flush=True)
        del model


if __name__ == '__main__':
    main()
