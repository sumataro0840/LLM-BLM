"""
Fetchers of the additional data (設計_追加データ.md, stage 1). Each returns a DataFrame in the shape of its table
(without fetched_at). The User-Agent is USER_AGENT of .env (data.sec.gov and Wikimedia ask for one).
"""
import io
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import pandas as pd

from marketdb.config import USER_AGENT, WIKIDATA_SPARQL_URL

_lock, _last = threading.Lock(), {}


def get(url, per_second=8.0, tries=5):
    """GET with a per-host rate limit and retries; None for 403/404 (a file that does not exist)."""
    host = urllib.parse.urlparse(url).netloc
    for attempt in range(tries):
        with _lock:
            wait = _last.get(host, 0.0) + 1 / per_second - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[host] = time.time()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f'failed: {url}')


# ---------- SEC filings index ----------

def filings_of(cik, first='2014-01-01'):
    """Every filing of a CIK from `first` on (form, items, acceptance time), from data.sec.gov/submissions."""
    sub = json.loads(get(f'https://data.sec.gov/submissions/CIK{int(cik):010d}.json'))
    pages = [sub['filings']['recent']]
    for extra in sub['filings'].get('files', []):
        if extra['filingTo'] >= first:
            pages.append(json.loads(get(f"https://data.sec.gov/submissions/{extra['name']}")))
    rows = []
    for p in pages:
        for acc, form, date, accepted, items, primary in zip(p['accessionNumber'], p['form'], p['filingDate'],
                                                            p['acceptanceDateTime'], p['items'], p['primaryDocument']):
            if date >= first:
                rows.append({'accession': acc, 'cik': int(cik), 'form': form, 'filing_date': date,
                             'accepted_at': accepted or None, 'items': items or None, 'primary_doc': primary or None})
    return pd.DataFrame(rows)


# ---------- Fama-French ----------

FRENCH = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip'
FRENCH_SETS = [('F-F_Research_Data_5_Factors_2x3_daily', 'daily'), ('F-F_Momentum_Factor_daily', 'daily'),
               ('F-F_Research_Data_5_Factors_2x3', 'monthly'), ('F-F_Momentum_Factor', 'monthly')]


def parse_french(text, freq):
    """The first table of a French CSV (daily: YYYYMMDD rows; monthly: YYYYMM rows, before the annual table),
    percent -> decimal. Returns long rows (date, factor, value)."""
    lines = text.splitlines()
    head = next(i for i, l in enumerate(lines) if l.strip().startswith(',') and any(c.isalpha() for c in l))
    factors = [c.strip() for c in lines[head].split(',')[1:]]
    width = 8 if freq == 'daily' else 6
    rows = []
    for line in lines[head + 1:]:
        parts = [p.strip() for p in line.split(',')]
        if not parts[0].isdigit() or len(parts[0]) != width:
            if rows:
                break                   # the end of the first table (the monthly files continue with annual rows)
            continue
        d = parts[0]
        if freq == 'daily':
            date = f'{d[:4]}-{d[4:6]}-{d[6:]}'
        else:
            date = (pd.Period(f'{d[:4]}-{d[4:]}', 'M').end_time).strftime('%Y-%m-%d')
        for f, v in zip(factors, parts[1:]):
            if v and float(v) > -99:
                rows.append({'date': date, 'factor': f, 'value': float(v) / 100})
    return pd.DataFrame(rows)


def french(name, freq):
    raw = get(FRENCH.format(name=name), per_second=2)
    z = zipfile.ZipFile(io.BytesIO(raw))
    text = z.read(z.namelist()[0]).decode('latin-1')
    return parse_french(text, freq).assign(dataset=name, freq=freq)


# ---------- FRED ----------

FRED = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}'
# series -> how its value becomes public (design, section 2)
FRED_SERIES = {
    'DGS3MO': 'daily', 'DGS2': 'daily', 'DGS10': 'daily', 'T10Y2Y': 'daily', 'T10Y3M': 'daily',
    'BAA10Y': 'daily', 'BAMLH0A0HYM2': 'daily', 'DTWEXBGS': 'daily', 'DCOILWTICO': 'daily', 'VIXCLS': 'daily',
    'T5YIE': 'daily', 'ICSA': 'weekly',
    'CPIAUCSL': 'monthly', 'UNRATE': 'monthly', 'INDPRO': 'monthly', 'PAYEMS': 'monthly', 'UMCSENT': 'monthly',
}


def available_date(dates, kind):
    d = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if kind == 'daily':
        out = d + pd.Timedelta(days=1)
    elif kind == 'weekly':
        out = d + pd.Timedelta(days=6)                 # weekly claims: week ending Saturday, published Thursday
    else:
        out = d + pd.offsets.MonthEnd(2)               # monthly: end of the following month (conservative)
    return out.strftime('%Y-%m-%d')


def fred(series_id, first='2010-01-01'):
    df = pd.read_csv(io.BytesIO(get(FRED.format(sid=series_id), per_second=2)))
    df.columns = ['date', 'value']
    df = df[df.date >= first].copy()
    df['value'] = pd.to_numeric(df.value, errors='coerce')
    df['available_date'] = list(available_date(df.date, FRED_SERIES[series_id]))
    return df.assign(series_id=series_id)


# ---------- FINRA short volume ----------

FINRA = 'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt'


def finra_symbol(s):
    return s.replace('.', '-').replace('/', '-')


def short_volume_day(day, symbols):
    """One day's consolidated file, rows of `symbols` only (FINRA's 'BRK.B' -> 'BRK-B'); None if no file."""
    raw = get(FINRA.format(d=day.replace('-', '')), per_second=4)
    if raw is None:
        return None
    df = pd.read_csv(io.BytesIO(raw), sep='|', dtype={'Date': str, 'Symbol': str})
    df = df[df.Date.str.fullmatch(r'\d{8}', na=False)]          # the last line is a record count
    df['symbol'] = df.Symbol.map(finra_symbol)
    df = df[df.symbol.isin(symbols)]
    return pd.DataFrame({'symbol': df.symbol, 'date': day, 'short_volume': df.ShortVolume.astype(float),
                         'short_exempt_volume': df.ShortExemptVolume.astype(float),
                         'total_volume': df.TotalVolume.astype(float)})


# ---------- Wikipedia page views ----------

def wikipedia_articles(ciks):
    """{cik: English Wikipedia article title} from Wikidata (property P5531 = CIK), one query."""
    values = ' '.join(f'"{c:010d}"' for c in ciks)
    query = ('SELECT ?cik ?article WHERE { VALUES ?cik { ' + values + ' } ?item wdt:P5531 ?cik . '
             '?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }')
    raw = get(f'{WIKIDATA_SPARQL_URL}?format=json&query={urllib.parse.quote(query)}', per_second=1)
    out = {}
    for b in json.loads(raw)['results']['bindings']:
        title = urllib.parse.unquote(b['article']['value'].rsplit('/wiki/', 1)[1])
        out.setdefault(int(b['cik']['value']), title)
    return out


WIKIMEDIA = ('https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/'
             '{title}/daily/{start}/{end}')


def pageviews(title, start='20150701', end='20260930'):
    raw = get(WIKIMEDIA.format(title=urllib.parse.quote(title.replace(' ', '_'), safe=''), start=start, end=end),
              per_second=10)
    if raw is None:
        return pd.DataFrame(columns=['article', 'date', 'views'])
    items = json.loads(raw)['items']
    return pd.DataFrame({'article': title, 'date': [f"{i['timestamp'][:4]}-{i['timestamp'][4:6]}-{i['timestamp'][6:8]}"
                                                    for i in items], 'views': [i['views'] for i in items]})
