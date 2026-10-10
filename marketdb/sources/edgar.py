"""
Financial statements from SEC EDGAR (XBRL company facts) and company id (CIK) lookup.
"""
import json
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

import pandas as pd

from marketdb.config import (EDGAR_FACTS_URL, EDGAR_SUBMISSIONS_URL, SP500_URL, USER_AGENT, WIKIDATA_SPARQL_URL)


def _get_json(url, accept=None):
    headers = {'User-Agent': USER_AGENT, **({'Accept': accept} if accept else {})}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        return json.load(response)


def load_wikipedia_ciks():
    """{symbol: cik} of the current S&P500 constituents (symbols as written on Wikipedia)."""
    table = pd.read_html(SP500_URL, storage_options={'User-Agent': USER_AGENT})[0]
    return {s: int(c) for s, c in zip(table['Symbol'], table['CIK'])}


def lookup_wikidata_ciks(symbols):
    """{symbol: [(cik, company name), ...]} for stock exchange tickers known to Wikidata (one request)."""
    values = ' '.join(f'"{s}"' for s in symbols)
    query = (f'SELECT ?t ?cik ?itemLabel WHERE {{ VALUES ?t {{ {values} }} ?item p:P414 ?st . ?st pq:P249 ?t . '
             f'?item wdt:P5531 ?cik . SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}')
    result = _get_json(f'{WIKIDATA_SPARQL_URL}?format=json&query={urllib.parse.quote(query)}', 'application/sparql-results+json')
    found = {}
    for b in result['results']['bindings']:
        candidate = (int(b['cik']['value']), b['itemLabel']['value'])
        if candidate not in found.setdefault(b['t']['value'], []):
            found[b['t']['value']].append(candidate)
    return found


def choose_cik(symbol, security, candidates):
    """Pick one of several (cik, name): the filer that lists the ticker, otherwise the closest company name."""
    if len({c for c, _ in candidates}) == 1:
        return candidates[0][0]
    best, best_score = None, -1.0
    for cik in dict.fromkeys(c for c, _ in candidates):
        try:
            sub = _get_json(EDGAR_SUBMISSIONS_URL.format(cik=f'{cik:010d}'))
        except Exception:
            continue
        if symbol in sub.get('tickers', []):
            return cik
        score = SequenceMatcher(None, security.lower(), sub.get('name', '').lower()).ratio()
        if score > best_score:
            best, best_score = cik, score
    return best


def load_company_facts(cik, start):
    """Values reported for periods ending on or after `start`, as first filed.

    Returns rows of (taxonomy, tag, unit, label, period_start, period_end, value, form, filed_date).
    The same period is reported again in later filings; only the earliest filing is kept.
    """
    data = _get_json(EDGAR_FACTS_URL.format(cik=f'{cik:010d}'))
    first = {}
    for taxonomy, tags in data.get('facts', {}).items():
        for tag, body in tags.items():
            for unit, rows in body.get('units', {}).items():
                for r in rows:
                    end = r['end']
                    if end < start or r.get('val') is None:
                        continue
                    key = (taxonomy, tag, unit, r.get('start') or end, end)
                    if key not in first or r['filed'] < first[key][3]:
                        first[key] = (body.get('label'), float(r['val']), r['form'], r['filed'])
    return [(*key[:3], label, key[3], key[4], value, form, filed) for key, (label, value, form, filed) in first.items()]
