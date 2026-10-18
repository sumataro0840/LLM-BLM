"""
EODHD (eodhd.com): daily prices that include delisted US stocks, which yfinance does not have.
The key is EODHD_API_KEY in LLM-BLM/.env. The data are licensed for personal use: the cache stays outside Git
(data/cache/eodhd/).

    python marketdb/sources/eodhd.py probe      # 3 calls: are the members without prices in EODHD's delisted list?
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from marketdb import config  # noqa: F401  (loads .env)

BASE = 'https://eodhd.com/api'
CACHE = REPO / 'data' / 'cache' / 'eodhd'


def _key():
    key = os.environ.get('EODHD_API_KEY')
    if not key:
        raise SystemExit('EODHD_API_KEY を LLM-BLM/.env に書いてください')
    return key


def get(path, **params):
    params.update(api_token=_key(), fmt='json')
    url = f'{BASE}/{path}?{urllib.parse.urlencode(params)}'
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def symbol_list(delisted):
    """Every US symbol (delisted=True: the delisted ones), cached: Code, Name, Exchange, Type, Isin ..."""
    path = CACHE / f'symbols_US_{"delisted" if delisted else "active"}.json'
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        data = get('exchange-symbol-list/US', **({'delisted': 1} if delisted else {}))
        path.write_text(json.dumps(data), encoding='utf-8')
    return json.loads(path.read_text(encoding='utf-8'))


def eod(symbol, start, end):
    """Daily prices of SYMBOL.US (date, open, high, low, close, adjusted_close, volume)."""
    return get(f'eod/{symbol}.US', **{'from': start, 'to': end})


def probe():
    from experiments.common import data as D
    c = D.connect()
    members = {r[0] for r in c.execute('SELECT DISTINCT symbol FROM index_membership')}
    priced = {r[0] for r in c.execute('SELECT DISTINCT symbol FROM prices')}
    missing = sorted(members - priced)
    delisted = {s['Code']: s for s in symbol_list(True)}
    active = {s['Code'] for s in symbol_list(False)}
    found = [s for s in missing if s.replace('-', '.') in delisted or s in delisted]
    print(f'members without prices: {len(missing)}; in the delisted list: {len(found)}; '
          f'in the active list (ticker reused?): {sum(s in active for s in missing)}')
    print('not found:', [s for s in missing if s not in found])
    for s in ('ALXN', 'CELG')[:1]:
        rows = eod(s, '2020-01-01', '2021-12-31')
        print(s, delisted.get(s, {}).get('Name'), f'{len(rows)} days', rows[:1] if rows else '')


if __name__ == '__main__':
    if sys.argv[1:] == ['probe']:
        probe()
    else:
        print(__doc__)
