"""
How much of the small-cap band is lost when only companies with prices today are kept, why the lost ones left, and
how they did before they left (survivorship bias of the small-cap data; design: 設計_小型株.md).
No stock reaction to a release is computed here.

    python -m smallcap.bias          -> smallcap/results/bias.json

A company-year of the band is "relevant" when the company filed an earnings 8-K (Item 2.02) in that year or the next.
It is "kept" when its ticker of today has a price in that period, and "lost" otherwise. The reason a lost company left
is read from its EDGAR filings (first match): bankruptcy (8-K Item 1.03), acquired (merger proxy, tender offer reply,
8-K Item 5.01), delisted or deregistered (Form 25 / 15), listed now but no price from yfinance, still filing (another
market or a new name), other. The public float of the year before (dei:EntityPublicFloat) stands in for the price.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DB = REPO / 'data' / 'smallcap.sqlite'
OUT = Path(__file__).resolve().parent / 'results'


def load():
    c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    band = pd.read_sql('SELECT b.year, b.cik, f.public_float FROM sc_band b JOIN sc_float f USING (cik, year) '
                       'WHERE b.in_band = 1', c)
    floats = pd.read_sql('SELECT cik, year, public_float FROM sc_float WHERE valid = 1', c)
    comp = pd.read_sql('SELECT * FROM sc_companies', c)
    fil = pd.read_sql('SELECT cik, filing_date FROM sc_filings', c)
    sym = pd.read_sql('SELECT symbol, cik FROM tickers WHERE cik IS NOT NULL', c)
    px = pd.read_sql("SELECT symbol, MIN(date) AS first_px, MAX(date) AS last_px FROM prices GROUP BY symbol", c)
    return band, floats, comp, fil, sym.merge(px, on='symbol', how='left')


def reason(r):
    if pd.notna(r.last_bankruptcy):
        return 'bankruptcy'
    if pd.notna(r.last_merger):
        return 'acquired'
    if pd.notna(r.last_25) or pd.notna(r.last_15):
        return 'delisted_or_deregistered'
    if pd.notna(r.ticker):
        return 'listed_but_no_price'
    if pd.notna(r.last_filing) and r.last_filing >= '2026-01-01':
        return 'still_filing'
    return 'other'


def tables():
    """(band company-years with relevant / kept / reason / float change, the last band year of each lost company)."""
    band, floats, comp, fil, sym = load()
    fil['year'] = fil.filing_date.str[:4].astype(int)
    earn_years = fil.groupby('cik').year.apply(set).to_dict()
    band['relevant'] = [any(y in earn_years.get(c, ()) for y in (yr, yr + 1)) for c, yr in zip(band.cik, band.year)]
    px_of = sym.groupby('cik').agg(first_px=('first_px', 'min'), last_px=('last_px', 'max'))
    band = band.join(px_of, on='cik')
    start, end = band.year.astype(str) + '-07-01', (band.year + 2).astype(str) + '-06-30'
    band['kept'] = band.first_px.notna() & (band.first_px <= end) & (band.last_px >= start)
    comp['reason'] = comp.apply(reason, axis=1)
    band = band.merge(comp[['cik', 'reason', 'last_filing']], on='cik', how='left')
    band.loc[band.kept, 'reason'] = 'kept'
    rel = band[band.relevant].copy()

    # float change from the year before (log), for the change of the stock before a company is lost
    prev = floats.assign(year=floats.year + 1).rename(columns={'public_float': 'float_prev'})
    rel = rel.merge(prev, on=['cik', 'year'], how='left')
    rel['dlog'] = np.log(rel.public_float / rel.float_prev)
    # the last band year of each lost company (the year closest to its exit)
    lost = rel[~rel.kept]
    last_year = lost.sort_values('year').groupby('cik').tail(1)
    return band, rel, last_year


def main():
    band, rel, last_year = tables()

    def describe(d):
        x = d.dlog.dropna()
        return {'n': int(len(d)), 'n_with_change': int(len(x)), 'median_change': float(np.expm1(x.median())) if len(x) else None,
                'mean_log_change': float(x.mean()) if len(x) else None,
                'share_down_40pct': float((x < np.log(0.6)).mean()) if len(x) else None}
    by_year = rel.groupby('year').apply(lambda d: pd.Series({
        'band': int(len(band[band.year == d.name])), 'relevant': int(len(d)), 'kept': int(d.kept.sum()),
        'lost': int((~d.kept).sum()), 'lost_share': float((~d.kept).mean())})).reset_index()
    reasons = rel[~rel.kept].groupby('reason').size().sort_values(ascending=False)
    reasons_companies = last_year.groupby('reason').size().sort_values(ascending=False)
    summary = {
        'band_company_years': int(len(band)), 'relevant_company_years': int(len(rel)),
        'kept_company_years': int(rel.kept.sum()), 'lost_company_years': int((~rel.kept).sum()),
        'lost_share': float((~rel.kept).mean()),
        'companies_relevant': int(rel.cik.nunique()), 'companies_kept_ever': int(rel[rel.kept].cik.nunique()),
        'companies_lost': int(last_year.cik.nunique()),
        'by_year': by_year.to_dict('records'),
        'lost_company_years_by_reason': {k: int(v) for k, v in reasons.items()},
        'lost_companies_by_reason': {k: int(v) for k, v in reasons_companies.items()},
        'float_change_kept': describe(rel[rel.kept]),
        'float_change_lost_all_years': describe(rel[~rel.kept]),
        'float_change_lost_last_year_by_reason': {k: describe(d) for k, d in last_year.groupby('reason')},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / 'bias.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'by_year'}, ensure_ascii=False, indent=1))
    print(by_year.to_string(index=False))


if __name__ == '__main__':
    main()
