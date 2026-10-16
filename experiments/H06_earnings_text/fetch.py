"""
H06 step 1: earnings press releases (8-K Item 2.02, exhibit 99) from SEC EDGAR, with their filing dates.

    SEC_USER_AGENT="<name> <contact e-mail>" python experiments/H06_earnings_text/fetch.py [test]

SEC asks for a contact in the User-Agent and at most 10 requests per second; the contact is passed in the
environment and is not stored. Texts go to data/cache/edgar/ (outside Git); the list of filings to
results/filings.csv.
"""
import html
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import data as D

OUT = Path(__file__).resolve().parent / 'results'
TEXTS = D.REPO / 'data' / 'cache' / 'edgar'
FIRST, LAST = '2023-11-01', '2026-08-30'     # filings after the model's knowledge cutoff, before the last decision date
MAX_PER_SECOND = 8
_lock, _last = threading.Lock(), [0.0]


def get(url):
    """GET with the SEC rate limit (shared across threads) and up to 4 retries."""
    for attempt in range(5):
        with _lock:
            wait = _last[0] + 1 / MAX_PER_SECOND - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': os.environ['SEC_USER_AGENT'], 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f'failed: {url}')


def filings_of(cik):
    """8-K filings with Item 2.02 between FIRST and LAST (older pages are read when needed)."""
    sub = json.loads(get(f'https://data.sec.gov/submissions/CIK{cik:010d}.json'))
    pages = [sub['filings']['recent']]
    for extra in sub['filings'].get('files', []):
        if extra['filingTo'] >= FIRST:
            pages.append(json.loads(get(f"https://data.sec.gov/submissions/{extra['name']}")))
    rows = []
    for p in pages:
        for form, date, acc, items, accepted, primary in zip(p['form'], p['filingDate'], p['accessionNumber'], p['items'],
                                                            p['acceptanceDateTime'], p['primaryDocument']):
            if form == '8-K' and '2.02' in (items or '') and FIRST <= date <= LAST:
                rows.append({'cik': cik, 'filing_date': date, 'accession': acc, 'accepted': accepted, 'primary': primary})
    return rows


EXHIBIT = re.compile(r'ex[-_]?99|exhibit[-_]?99|99[-_.]?1', re.I)


def choose_exhibit(items, primary):
    """The press release: an htm/txt file named like exhibit 99(.1); otherwise the largest other document."""
    docs = [i for i in items if i['name'].lower().endswith(('.htm', '.html', '.txt')) and not i['name'].startswith(('0', 'R'))
            and 'index' not in i['name'].lower() and i['name'] != primary]
    named = [i for i in docs if EXHIBIT.search(i['name'])]
    pool = named or docs
    if not pool:
        return None, 'none'
    size = lambda i: int(i['size']) if str(i.get('size', '')).isdigit() else 0
    # exhibit 99.1 before 99.2 (the second one is often a slide deck or a supplement)
    pool.sort(key=lambda i: (0 if re.search(r'99[-_.]?0?1(?!\d)', i['name']) else 1, -size(i)))
    return pool[0]['name'], 'named' if named else 'largest'


def to_text(raw):
    """HTML -> plain text: scripts and tags removed, entities decoded, white space collapsed."""
    s = raw.decode('utf-8', errors='replace')
    s = re.sub(r'(?is)<(script|style).*?</\1>', ' ', s)
    s = re.sub(r'(?i)<br\s*/?>|</(p|div|tr|li|h\d)>', '\n', s)
    s = re.sub(r'(?s)<[^>]+>', ' ', s)
    s = html.unescape(s).replace('\xa0', ' ')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n\s*\n+', '\n', s)
    return s.strip()


def fetch_one(row):
    acc = row['accession'].replace('-', '')
    path = TEXTS / str(row['cik']) / f'{acc}.txt'
    if path.exists():
        return {**row, 'file': str(path.relative_to(D.REPO)), 'chars': path.stat().st_size, 'how': 'cached'}
    index = get(f"https://www.sec.gov/Archives/edgar/data/{row['cik']}/{acc}/index.json")
    if index is None:
        return {**row, 'file': None, 'chars': 0, 'how': 'no index'}
    name, how = choose_exhibit(json.loads(index)['directory']['item'], row['primary'])
    if name is None:
        return {**row, 'file': None, 'chars': 0, 'how': how}
    raw = get(f"https://www.sec.gov/Archives/edgar/data/{row['cik']}/{acc}/{name}")
    text = to_text(raw) if raw else ''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return {**row, 'document': name, 'file': str(path.relative_to(D.REPO)), 'chars': len(text), 'how': how}


def main(test=False):
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM llm_views WHERE run_id = 1")]
    ciks = pd.read_sql("SELECT symbol, cik FROM tickers WHERE cik IS NOT NULL", conn).set_index('symbol').cik
    ciks = ciks.reindex(symbols).dropna().astype(int)
    print(f'{len(symbols)} symbols, {len(ciks)} with a CIK', flush=True)
    if test:
        ciks = ciks.head(3)
    t0 = time.time()
    with ThreadPoolExecutor(4) as pool:
        lists = list(pool.map(filings_of, sorted(set(ciks.values))))
    filings = [f for lst in lists for f in lst]
    print(f'{len(filings)} filings ({time.time() - t0:.0f}s)', flush=True)
    with ThreadPoolExecutor(4) as pool:
        done = list(pool.map(fetch_one, filings))
    df = pd.DataFrame(done)
    sym_of = {}
    for s, c in ciks.items():
        sym_of.setdefault(c, []).append(s)
    df['symbols'] = df.cik.map(lambda c: ','.join(sym_of.get(c, [])))
    df.to_csv(OUT / ('filings_test.csv' if test else 'filings.csv'), index=False)
    print(df.how.value_counts().to_dict(), 'median chars', df.chars.median(), f'({time.time() - t0:.0f}s)')


MAIN_TYPES = ('EX-99.1', 'EX-99', 'EX-99.01')


def doc_type(path):
    """The exhibit type written at the top of a saved document (e.g. EX-99.1)."""
    m = re.match(r'\s*(EX-[\d.A-Z]+)', path.read_text(encoding='utf-8')[:100])
    return m.group(1) if m else None


def refine():
    """Where the chosen file is not the press release itself (EX-99.2 supplements, EX-10 contracts),
    look through the other documents of the filing for the EX-99.1."""
    df = pd.read_csv(OUT / 'filings.csv')
    fixed = 0
    for i, row in df[df.file.notna()].iterrows():
        path = D.REPO / row.file
        if doc_type(path) in MAIN_TYPES:
            continue
        acc = row.accession.replace('-', '')
        items = json.loads(get(f"https://www.sec.gov/Archives/edgar/data/{row.cik}/{acc}/index.json"))['directory']['item']
        docs = [it['name'] for it in items if it['name'].lower().endswith(('.htm', '.html', '.txt'))
                and it['name'] != row.get('document') and it['name'] != row.primary and not it['name'].startswith(('0', 'R'))
                and 'index' not in it['name'].lower()]
        for name in docs:
            text = to_text(get(f"https://www.sec.gov/Archives/edgar/data/{row.cik}/{acc}/{name}") or b'')
            if re.match(r'\s*(EX-99\.1|EX-99|EX-99\.01)\b', text[:100]):
                path.write_text(text, encoding='utf-8')
                df.loc[i, ['document', 'chars', 'how']] = [name, len(text), 'refined']
                fixed += 1
                break
        else:
            df.loc[i, 'how'] = df.loc[i, 'how'] + '_not_ex991'
    df.to_csv(OUT / 'filings.csv', index=False)
    print('refined', fixed, df.how.value_counts().to_dict())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'refine':
        refine()
    else:
        main(test=len(sys.argv) > 1 and sys.argv[1] == 'test')
