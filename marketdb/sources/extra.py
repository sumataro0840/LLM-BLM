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


def get(url, per_second=8.0, tries=5, agent=None):
    """GET with a per-host rate limit and retries; None for 403/404 (a file that does not exist)."""
    host = urllib.parse.urlparse(url).netloc
    attempt, waits = 0, 0
    while attempt < tries:
        with _lock:
            wait = _last.get(host, 0.0) + 1 / per_second - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[host] = time.time()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': agent or USER_AGENT, 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if e.code == 429 and waits < 40:        # rate limited: wait as asked, without using up the tries
                waits += 1
                time.sleep(float(e.headers.get('retry-after') or 10) + 1)
                continue
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 ** attempt)
        attempt += 1
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
    # keep_default_na=False: a symbol such as 'NA' must not become a missing value
    df = pd.read_csv(io.BytesIO(raw), sep='|', dtype=str, keep_default_na=False)
    df = df[df.Date.str.fullmatch(r'\d{8}', na=False)]          # the last line is a record count
    df['symbol'] = df.Symbol.map(finra_symbol)
    df = df[df.symbol.isin(symbols)]
    num = lambda c: pd.to_numeric(df[c], errors='coerce')
    return pd.DataFrame({'symbol': df.symbol, 'date': day, 'short_volume': num('ShortVolume'),
                         'short_exempt_volume': num('ShortExemptVolume'), 'total_volume': num('TotalVolume')})


# ---------- SEC insider transactions (www.sec.gov needs SEC_USER_AGENT with a contact) ----------

INSIDER = 'https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip'


def sec_agent():
    import os
    agent = os.environ.get('SEC_USER_AGENT')
    if not agent:
        raise SystemExit('SEC_USER_AGENT を LLM-BLM/.env に書いてください（www.sec.gov の求め）')
    return agent


def _date(s):
    return pd.to_datetime(s, format='%d-%b-%Y', errors='coerce').dt.strftime('%Y-%m-%d')


def insider_quarter(year, quarter, symbols_of_cik):
    """Non-derivative transactions of one quarter's data set, for issuers in symbols_of_cik ({cik: [symbols]}).
    None if the file does not exist (yet)."""
    raw = get(INSIDER.format(y=year, q=quarter), per_second=1, agent=sec_agent())
    if raw is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(raw))
    read = lambda name, cols: pd.read_csv(z.open(name), sep='\t', dtype=str, usecols=cols, on_bad_lines='skip')
    sub = read('SUBMISSION.tsv', ['ACCESSION_NUMBER', 'FILING_DATE', 'DOCUMENT_TYPE', 'ISSUERCIK'])
    sub['cik'] = pd.to_numeric(sub.ISSUERCIK, errors='coerce')
    sub = sub[sub.cik.isin(list(symbols_of_cik))]
    own = read('REPORTINGOWNER.tsv', ['ACCESSION_NUMBER', 'RPTOWNERCIK', 'RPTOWNER_RELATIONSHIP', 'RPTOWNER_TITLE'])
    own = own[own.ACCESSION_NUMBER.isin(sub.ACCESSION_NUMBER)]
    own = own.groupby('ACCESSION_NUMBER').agg(owner_cik=('RPTOWNERCIK', 'first'),
                                              owner_relationship=('RPTOWNER_RELATIONSHIP',
                                                                  lambda s: ';'.join(sorted(set(s.dropna())))),
                                              owner_title=('RPTOWNER_TITLE', 'first'))
    tr = read('NONDERIV_TRANS.tsv', ['ACCESSION_NUMBER', 'NONDERIV_TRANS_SK', 'TRANS_DATE', 'TRANS_CODE',
                                     'TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'TRANS_ACQUIRED_DISP_CD',
                                     'SHRS_OWND_FOLWNG_TRANS', 'DIRECT_INDIRECT_OWNERSHIP'])
    tr = tr.merge(sub, on='ACCESSION_NUMBER').join(own, on='ACCESSION_NUMBER')
    filing = _date(tr.FILING_DATE)
    out = pd.DataFrame({
        'accession': tr.ACCESSION_NUMBER, 'trans_sk': tr.NONDERIV_TRANS_SK, 'issuer_cik': tr.cik.astype(int),
        'form': tr.DOCUMENT_TYPE, 'filing_date': filing,
        'available_date': (pd.to_datetime(filing) + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d'),
        'trans_date': _date(tr.TRANS_DATE), 'code': tr.TRANS_CODE, 'acquired_disposed': tr.TRANS_ACQUIRED_DISP_CD,
        'shares': pd.to_numeric(tr.TRANS_SHARES, errors='coerce'),
        'price': pd.to_numeric(tr.TRANS_PRICEPERSHARE, errors='coerce'),
        'shares_after': pd.to_numeric(tr.SHRS_OWND_FOLWNG_TRANS, errors='coerce'),
        'direct_indirect': tr.DIRECT_INDIRECT_OWNERSHIP, 'owner_cik': tr.owner_cik,
        'owner_relationship': tr.owner_relationship, 'owner_title': tr.owner_title})
    out = out[out.filing_date.notna()]
    # one row per symbol of the issuer (two share classes of one company get the same transactions)
    out['symbol'] = out.issuer_cik.map(symbols_of_cik)
    return out.explode('symbol')


# ---------- SEC 13F data sets ----------

THIRTEEN_F_PAGE = 'https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets'


def thirteen_f_files():
    """Links of every 13F data set (quarterly up to 2023, ranges of filing dates from 2024)."""
    import re
    page = get(THIRTEEN_F_PAGE, per_second=1, agent=sec_agent()).decode('utf-8', 'replace')
    return sorted(set(re.findall(r'href="([^"]+form13f[^"]*\.zip)"', page)))


def thirteen_f_parts(link):
    """Sum of shares, value and filers per CUSIP and period in one data set (rules of holdings_13f_parts)."""
    raw = get('https://www.sec.gov' + link, per_second=1, agent=sec_agent())
    z = zipfile.ZipFile(io.BytesIO(raw))
    sub = pd.read_csv(z.open('SUBMISSION.tsv'), sep='\t', dtype=str)
    sub = sub[sub.SUBMISSIONTYPE == '13F-HR'].copy()
    sub['filed'] = pd.to_datetime(sub.FILING_DATE, format='%d-%b-%Y', errors='coerce')
    sub['period'] = pd.to_datetime(sub.PERIODOFREPORT, format='%d-%b-%Y', errors='coerce')
    sub = sub[sub.filed <= sub.period + pd.Timedelta(days=45)]
    info = pd.read_csv(z.open('INFOTABLE.tsv'), sep='\t', dtype=str, on_bad_lines='skip',
                       usecols=['ACCESSION_NUMBER', 'NAMEOFISSUER', 'CUSIP', 'VALUE', 'SSHPRNAMT', 'SSHPRNAMTTYPE',
                                'PUTCALL'])
    info = info[(info.SSHPRNAMTTYPE == 'SH') & info.PUTCALL.isna()]
    info = info.merge(sub[['ACCESSION_NUMBER', 'filed', 'period']], on='ACCESSION_NUMBER')
    info['cusip'] = info.CUSIP.str.strip().str.upper()
    info = info[info.cusip.str.fullmatch(r'[0-9A-Z]{9}', na=False)]
    info['shares'] = pd.to_numeric(info.SSHPRNAMT, errors='coerce')
    info['value'] = pd.to_numeric(info.VALUE, errors='coerce') * \
        (info.filed < pd.Timestamp('2023-01-03')).map({True: 1000.0, False: 1.0})
    info['period'] = info.period.dt.strftime('%Y-%m-%d')
    g = info.groupby(['cusip', 'period'])
    out = g.agg(n_filers=('ACCESSION_NUMBER', 'nunique'), shares=('shares', 'sum'), value=('value', 'sum'),
                issuer_name=('NAMEOFISSUER', 'first')).reset_index()
    return out.assign(source_file=link.rsplit('/', 1)[1])


def openfigi_tickers(cusips):
    """{cusip: (figi ticker, name)} for US equities from OpenFIGI's mapping API (no key: 10 per request,
    25 requests a minute)."""
    out = {}
    cusips = list(cusips)
    for i in range(0, len(cusips), 10):
        body = json.dumps([{'idType': 'ID_CUSIP', 'idValue': c} for c in cusips[i:i + 10]]).encode()
        for attempt in range(6):
            time.sleep(2.5)
            req = urllib.request.Request('https://api.openfigi.com/v3/mapping', data=body,
                                         headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    res = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(30)
                    continue
                raise
        else:
            raise RuntimeError('OpenFIGI kept refusing (429)')
        for c, r in zip(cusips[i:i + 10], res):
            data = [d for d in r.get('data', []) if d.get('exchCode') == 'US'] or r.get('data', [])
            if data:
                out[c] = (data[0].get('ticker'), data[0].get('name'))
    return out


# ---------- Wikipedia page views ----------

def wikipedia_articles(ciks, batch=80):
    """{cik: English Wikipedia article title} from Wikidata (property P5531 = CIK), 80 CIKs per query
    (one query with all of them makes the URL too long)."""
    out = {}
    ciks = list(ciks)
    for i in range(0, len(ciks), batch):
        values = ' '.join(f'"{c:010d}"' for c in ciks[i:i + batch])
        query = ('SELECT ?cik ?article WHERE { VALUES ?cik { ' + values + ' } ?item wdt:P5531 ?cik . '
                 '?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }')
        raw = get(f'{WIKIDATA_SPARQL_URL}?format=json&query={urllib.parse.quote(query)}', per_second=0.5)
        if raw is None:
            continue
        for b in json.loads(raw)['results']['bindings']:
            title = urllib.parse.unquote(b['article']['value'].rsplit('/wiki/', 1)[1])
            out.setdefault(int(b['cik']['value']), title)
    return out


WIKIMEDIA = ('https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/'
             '{title}/daily/{start}/{end}')


def pageviews(title, start='20150701', end='20260930'):
    raw = get(WIKIMEDIA.format(title=urllib.parse.quote(title.replace(' ', '_'), safe=''), start=start, end=end),
              per_second=1)
    if raw is None:
        return pd.DataFrame(columns=['article', 'date', 'views'])
    items = json.loads(raw)['items']
    return pd.DataFrame({'article': title, 'date': [f"{i['timestamp'][:4]}-{i['timestamp'][4:6]}-{i['timestamp'][6:8]}"
                                                    for i in items], 'views': [i['views'] for i in items]})
