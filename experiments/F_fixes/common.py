"""
Shared pieces of the fix experiments (資料集/03_方針と文献調査/今後の計画/修正の実験方針.md):
the release panel, the firm-level and sector-level metrics, the rolling models trained on earlier releases only,
and the sector Black-Litterman portfolio.

Nothing here chooses anything: develop.py applies the keep rule, final.py runs the final comparison.
"""
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from events import config, texts
from events.evaluate import design
from experiments.F_fixes import score as S
from experiments.F_fixes.clean import clean

CACHE = S.OUT
PANEL = REPO / 'experiments' / 'E2_event_reaction' / 'results' / 'panel.csv'

# ---------- periods (修正の実験方針.md, 2) ----------
START = '2016-01-01'
END = '2026-08-31'
SPLITS = {'B': '2019-12-31', 'A': '2020-12-31', 'C': '2022-12-31'}        # last day of development
HALF = {'B': '2017-12-31', 'A': '2018-12-31', 'C': '2019-12-31'}          # last day of the first half
GPT_START = '2023-11-01'
NUM_CONTROLS = ['sue', 'revenue_surprise', 'sue_missing', 'revenue_surprise_missing', 'abn_pre', 'log_mcap',
                'analyst_surprise', 'analyst_missing']
FIXED = ['sector', 'year_quarter']


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e2 = _load('e2_run', 'experiments/E2_event_reaction/run.py')
e3 = _load('e3_run', 'experiments/E3_analyst_surprise/run.py')
e5 = _load('e5_run', 'experiments/E5_chrono_embeddings/run.py')


def final_start(split):
    return (pd.Timestamp(SPLITS[split]) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')


# ---------- the release panel ----------

def base_panel(conn):
    """E5's releases (E2's base sample with a main-window reaction and a text) with E2's variables, the matched
    analyst surprise and the acceptance time. One row per release, sorted by filing date."""
    from events import estimates
    from marketdb.db import events as db
    p = pd.read_csv(PANEL)
    p = p[p.abn_r_0_1.notna()]
    rel = db.load_releases(conn, with_text=True)[['accession', 'accepted_et', 'text_path']]
    p = p.merge(rel, on='accession')
    est = db.load_estimates(conn, sorted(p.symbol.unique()))
    m = estimates.match_releases(p, est)
    p = p.merge(m[['accession', 'fiscal_date_ending']], on='accession', how='left')
    p = p.merge(est[['symbol', 'fiscal_date_ending', 'surprise', 'surprise_pct']], on=['symbol', 'fiscal_date_ending'],
                how='left')
    return p.sort_values(['filing_date', 'accession']).reset_index(drop=True)


def window(p, start, end):
    return p[p.filing_date.between(start, end)]


# ---------- firm-level metric (3.2) ----------

def _prepared(A, col):
    """E3's preparation (E2's winsorizing and standardizing within A, analyst variables) with col as the focal score."""
    return e3.prepare(A.assign(_score=A[col]), '_score', 'abn_r_0_1')


def _resid(d, xs):
    X = design(d, xs, FIXED).to_numpy(float)
    y = d['y'].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def cluster_t_of_mean(values, clusters):
    """t of the mean of values with the standard error clustered by clusters."""
    v = np.asarray(values, float)
    dev = pd.Series(v - v.mean()).groupby(np.asarray(clusters)).sum().to_numpy()
    g, n = len(dev), len(v)
    se = np.sqrt((dev ** 2).sum() * g / (g - 1)) / n
    return float(v.mean() / se) if se > 0 else 0.0


def firm_compare(A, base_col, fix_col):
    """R2 gain over the numeric controls of each score, and the paired t of the squared-residual reduction
    (positive = the fixed score explains the reaction better), on the releases where both scores exist."""
    A = A[A[base_col].notna() & A[fix_col].notna()]
    db, df = _prepared(A, base_col), _prepared(A, fix_col)
    ctrl = [c for c in NUM_CONTROLS if db[c].std() > 0]
    e0 = _resid(db, ctrl)
    eb, ef = _resid(db, ctrl + ['_score']), _resid(df, ctrl + ['_score'])
    tss = ((db.y - db.y.mean()) ** 2).sum()
    diff = eb ** 2 - ef ** 2
    return {'n': int(len(A)), 'r2_gain_base': float(((e0 ** 2).sum() - (eb ** 2).sum()) / tss),
            'r2_gain_fix': float(((e0 ** 2).sum() - (ef ** 2).sum()) / tss),
            'delta': float(diff.sum() / tss), 't': cluster_t_of_mean(diff, db.event_date)}


def r2_gain(A, cols):
    """R2 gain over the numeric controls of a set of scores (each standardized within A)."""
    A = A.dropna(subset=cols)
    d = _prepared(A, cols[0])
    for c in cols[1:]:
        d[c] = e2.standardize(A[c])
    d['_s0'] = d['_score']
    ctrl = [c for c in NUM_CONTROLS if d[c].std() > 0]
    e0 = _resid(d, ctrl)
    e1 = _resid(d, ctrl + ['_s0'] + list(cols[1:]))
    return float(((e0 ** 2).sum() - (e1 ** 2).sum()) / ((d.y - d.y.mean()) ** 2).sum())


# ---------- rolling models (trained on releases whose window closed before the quarter began) ----------

def quarters(p):
    return pd.PeriodIndex(p.accepted_et.str[:10], freq='Q')


def rolling_ridge(p, X):
    """E5's rolling ridge. p and X aligned row by row."""
    pred, _ = e5.rolling_ridge(p[['accession', 'accepted_et', 'window_end', 'abn_r_0_1']].reset_index(drop=True), X)
    return pred


def rolling_tfidf(p, text_of):
    """E2's TF-IDF + ridge (same settings as events/scorers/tfidf.py) on text_of(accession)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from events.scorers import tfidf as T
    q = quarters(p)
    pred = pd.Series(np.nan, index=p.accession.to_numpy())
    for quarter in sorted(q.unique()):
        start = quarter.start_time.strftime('%Y-%m-%d')
        train = p[(p.window_end < start) & p.abn_r_0_1.notna()]
        test = p[q == quarter]
        if len(train) < T.MIN_TRAIN:
            continue
        vec = TfidfVectorizer(max_features=T.MAX_FEATURES, min_df=T.MIN_DF, sublinear_tf=True, stop_words='english')
        X = vec.fit_transform([text_of(a) for a in train.accession])
        model = Ridge(alpha=T.ALPHA).fit(X, train.abn_r_0_1.to_numpy())
        pred[test.accession.to_numpy()] = model.predict(vec.transform([text_of(a) for a in test.accession]))
    return pred


LOGIT_FEATURES = ['sue', 'revenue_surprise', 'abn_pre', 'log_mcap', 'surprise_pct']


def logit_matrix(train, test, score_cols):
    """Features winsorized at the training 1%/99% points, missing values filled with the training mean plus a
    missing flag, standardized with the training mean and sd."""
    cols = LOGIT_FEATURES + list(score_cols)
    Xtr, Xte = [], []
    for c in cols:
        a, b = train[c].astype(float), test[c].astype(float)
        if a.notna().sum() > 1:
            lo, hi = a.quantile([0.01, 0.99])
            a, b = a.clip(lo, hi), b.clip(lo, hi)
        mu = a.mean() if a.notna().any() else 0.0
        fa, fb = a.isna().astype(float), b.isna().astype(float)
        a, b = a.fillna(mu), b.fillna(mu)
        sd = a.std()
        sd = sd if np.isfinite(sd) and sd > 0 else 1.0
        Xtr += [(a - mu) / sd]
        Xte += [(b - mu) / sd]
        if c in LOGIT_FEATURES and fa.std() > 0:
            Xtr.append(fa)
            Xte.append(fb)
    return np.column_stack(Xtr), np.column_stack(Xte)


def rolling_logit(p, score_cols=()):
    """Probability that the main-window reaction is positive, from a logistic regression on the numeric features
    (plus score_cols) fitted every quarter on releases whose window closed before the quarter began."""
    from sklearn.linear_model import LogisticRegression
    q = quarters(p)
    pred = pd.Series(np.nan, index=p.accession.to_numpy())
    for quarter in sorted(q.unique()):
        start = quarter.start_time.strftime('%Y-%m-%d')
        train = p[(p.window_end < start) & p.abn_r_0_1.notna()].dropna(subset=list(score_cols))
        test = p[(q == quarter)].dropna(subset=list(score_cols))
        if len(train) < 500 or not len(test):
            continue
        Xtr, Xte = logit_matrix(train, test, score_cols)
        model = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, (train.abn_r_0_1 > 0).to_numpy())
        pred[test.accession.to_numpy()] = model.predict_proba(Xte)[:, 1]
    return pred


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def auc_diff_ci(A, a_col, b_col, reps=300, seed=0):
    """AUC(b) - AUC(a) on A, with a 95% interval from resampling event dates."""
    A = A.dropna(subset=[a_col, b_col])
    y = (A.abn_r_0_1 > 0).to_numpy()
    point = auc(y, A[b_col]) - auc(y, A[a_col])
    dates = A.event_date.unique()
    by = {d: g.index.to_numpy() for d, g in A.groupby('event_date')}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        idx = np.concatenate([by[d] for d in rng.choice(dates, len(dates))])
        B = A.loc[idx]
        yy = (B.abn_r_0_1 > 0).to_numpy()
        draws.append(auc(yy, B[b_col]) - auc(yy, B[a_col]))
    return {'auc_a': auc(y, A[a_col]), 'auc_b': auc(y, A[b_col]), 'diff': float(point),
            'ci95': [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))], 'n': int(len(A))}


# ---------- text access ----------

_clean_cache = {}


def clean_text(text_path, n=None):
    if text_path not in _clean_cache:
        _clean_cache[text_path] = clean(texts.read(text_path))
    t = _clean_cache[text_path]
    return t[:n] if n else t


# ---------- sector data (month ends, cap weights, next-month returns) ----------

def month_ends(conn):
    from marketdb.db.market import load_trading_days
    days = pd.Series(pd.to_datetime(load_trading_days(conn, '2015-12-01', '2026-09-30')))
    ends = days.groupby(days.dt.to_period('M')).max()
    return [d.strftime('%Y-%m-%d') for d in ends]


def _sector_month(args):
    """Members of one month end with sector, cap weight and next-month return, the daily sector returns of the
    previous 252 trading days (weights of the month end) and the 3-month bill rate."""
    date, next_end = args
    from experiments.common import data as D
    from events.returns import OLD_SECTORS
    from marketdb.db.index import load_index_weights
    from marketdb.db.market import last_trading_days, load_prices
    conn = D.connect()
    w = load_index_weights(conn, date)
    sector = pd.read_sql('SELECT symbol, gics_sector FROM tickers', conn).set_index('symbol').gics_sector.replace(OLD_SECTORS)
    days = last_trading_days(conn, date, 253)
    prices = load_prices(conn, list(w.index), days[0], next_end)
    past = prices[prices.index <= pd.Timestamp(date)]
    fut = prices[prices.index >= pd.Timestamp(date)].ffill()
    ret_next = (fut.iloc[-1] / fut.iloc[0] - 1).reindex(w.index).fillna(0.0)
    daily = past.pct_change().iloc[1:].reindex(columns=w.index)
    daily = daily.apply(lambda c: c.fillna(daily.mean(axis=1)))
    m = pd.DataFrame({'symbol': w.index, 'sector': sector.reindex(w.index).to_numpy(), 'weight': w.to_numpy(),
                      'ret_next': ret_next.to_numpy()})
    sw = m.groupby('sector').weight.sum()
    sec_daily = pd.DataFrame({s: (daily[g.symbol].to_numpy() * (g.weight / g.weight.sum()).to_numpy()).sum(1)
                              for s, g in m.groupby('sector')}, index=daily.index)
    irx = conn.execute("SELECT adj_close FROM prices WHERE symbol = '^IRX' AND date <= ? ORDER BY date DESC LIMIT 1",
                       (date,)).fetchone()[0]
    return date, {'members': m.assign(month=date), 'sector_weight': sw / sw.sum(), 'sector_daily': sec_daily,
                  'rf_month': float(irx) / 100 / 12}


def sector_data(conn):
    """{month end: data of _sector_month} for 2015-12 .. 2026-07 (cached; contains next-month returns)."""
    import pickle
    path = CACHE / 'sector_months.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    ends = month_ends(conn)
    jobs = [(d, n) for d, n in zip(ends[:-1], ends[1:]) if '2015-12' <= d[:7] <= '2026-07']
    with ProcessPoolExecutor(5) as pool:
        data = dict(pool.map(_sector_month, jobs))
    path.write_bytes(pickle.dumps(data))
    return data


# ---------- sector aggregation (fix 3) and the sector metric (3.3) ----------

AGGREGATIONS = ['latest92_cap', 'latest92_eq', 'hl60_cap', 'hl60_eq', 'hl30_cap', 'hl30_eq']   # simplest first


def sector_scores(p, col, month, members, how):
    """Sector view of one month end: each member's latest release whose reaction had started by the month end,
    its score standardized across the releases in use, multiplied by the age weight (latest92: 1 up to 92 days;
    hlXX: 0.5 ** (age / XX) up to 182 days) and averaged over all members with cap or equal weights
    (members without a score count as 0, which shrinks towards the market)."""
    kind, weighting = how.split('_')
    max_age = 92 if kind == 'latest92' else 182
    lo = (pd.Timestamp(month) - pd.Timedelta(days=max_age)).strftime('%Y-%m-%d')
    r = p[(p.event_date <= month) & (p.event_date > lo) & p.symbol.isin(members.symbol) & p[col].notna()]
    r = r.sort_values(['event_date', 'accepted_at']).drop_duplicates('symbol', keep='last')
    if len(r) < 20:
        return None
    z = (r[col] - r[col].mean()) / r[col].std()
    age = (pd.Timestamp(month) - pd.to_datetime(r.event_date)).dt.days.to_numpy()
    mult = np.ones(len(r)) if kind == 'latest92' else 0.5 ** (age / int(kind[2:]))
    val = pd.Series(z.to_numpy() * mult, index=r.symbol.to_numpy())
    m = members.assign(v=members.symbol.map(val).fillna(0.0))
    wt = m.weight if weighting == 'cap' else pd.Series(1.0, index=m.index)
    return (m.v * wt).groupby(m.sector).sum() / wt.groupby(m.sector).sum()


def sector_ics(p, col, sd, months, how):
    """Monthly Spearman correlation across the 11 sectors between the view and the next-month sector return
    in excess of the market (cap-weighted). Series indexed by month end."""
    out = {}
    for month in months:
        x = sd[month]
        view = sector_scores(p, col, month, x['members'], how)
        if view is None:
            continue
        m = x['members']
        sec = (m.weight * m.ret_next).groupby(m.sector).sum() / m.weight.groupby(m.sector).sum()
        mkt = float((m.weight * m.ret_next).sum() / m.weight.sum())
        out[month] = stats.spearmanr(view.reindex(sec.index), sec - mkt).statistic
    return pd.Series(out, dtype=float)


def newey_west_t(x, lag=3):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n, mu = len(x), x.mean()
    e = x - mu
    s = e @ e / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(mu / np.sqrt(s / n)) if s > 0 else 0.0


def sector_compare(p, base, fix, sd, months):
    """base, fix: (score column, aggregation). Mean IC of each and the Newey-West t of the monthly difference."""
    a = sector_ics(p, base[0], sd, months, base[1])
    b = sector_ics(p, fix[0], sd, months, fix[1])
    both = pd.concat([a, b], axis=1, keys=['a', 'b']).dropna()
    diff = both.b - both.a
    return {'months': int(len(both)), 'ic_base': float(both.a.mean()), 'ic_fix': float(both.b.mean()),
            'delta': float(diff.mean()), 't': newey_west_t(diff)}


# ---------- sector Black-Litterman (11 sectors + 3-month bill; reported, not part of the keep rule) ----------

DELTA, TAU, IC, TE, COST = 2.5, 0.025, 0.05, 0.03, 0.001


def sector_blm(p, col, how, sd, months):
    """Monthly sector BLM: prior = equilibrium of the sector cap weights, one view per sector
    q = prior + IC * sigma * normal score of the sector view, He-Litterman Omega, ex-ante TE at most 3 % a year
    against the sector cap weights (bill weight 0), long only. Returns monthly rows (gross, net, benchmark)."""
    from sklearn.covariance import LedoitWolf
    from blm.black_litterman import posterior_returns_general
    from blm.optimizer import optimize_with_tracking_error
    rows, prev = [], None
    for month in months:
        x = sd[month]
        view = sector_scores(p, col, month, x['members'], how)
        if view is None:
            continue
        sec = list(x['sector_weight'].index)
        daily = x['sector_daily'][sec]
        cov_s = LedoitWolf().fit(daily.to_numpy()).covariance_
        names = sec + ['BILL']
        cov = np.zeros((len(names), len(names)))
        cov[:-1, :-1] = cov_s
        cov[-1, -1] = 1e-10
        cov = pd.DataFrame(cov, index=names, columns=names)
        wb = pd.Series(list(x['sector_weight'].to_numpy()) + [0.0], index=names)
        prior = DELTA * cov @ wb
        z = pd.Series(stats.norm.ppf((stats.rankdata(view.reindex(sec)) - 0.375) / (len(sec) + 0.25)), index=sec)
        sigma = pd.Series(np.sqrt(np.diag(cov_s)), index=sec)
        pick = pd.DataFrame(np.eye(len(names))[:len(sec)], columns=names)
        q = pd.Series(prior[sec].to_numpy() + IC * sigma.to_numpy() * z.to_numpy())
        omega = TAU * np.diag(np.diag(pick.to_numpy() @ cov.to_numpy() @ pick.to_numpy().T))
        mu = posterior_returns_general(prior, cov, pick, q, omega, TAU)
        w, status = optimize_with_tracking_error(mu, cov, DELTA, wb, TE / np.sqrt(252))
        m = x['members']
        r_sec = (m.weight * m.ret_next).groupby(m.sector).sum() / m.weight.groupby(m.sector).sum()
        r = pd.Series(list(r_sec.reindex(sec).to_numpy()) + [x['rf_month']], index=names)
        gross, bench = float(w @ r), float(wb @ r)
        traded = float((w - (wb if prev is None else prev)).abs().sum())
        rows.append({'month': month, 'gross': gross, 'net': gross - COST * traded, 'bench': bench,
                     'turnover': traded / 2, 'bill': float(w['BILL']), 'status': status})
        prev = w
    df = pd.DataFrame(rows).set_index('month')
    df['active_net'] = df.net - df.bench
    return df


def blm_summary(df):
    a = df.active_net
    return {'months': int(len(df)), 'active_net_monthly': float(a.mean()), 'nw_t': newey_west_t(a),
            'ir_net': float(a.mean() / a.std() * np.sqrt(12)) if a.std() > 0 else 0.0,
            'turnover': float(df.turnover.mean()), 'bill_mean': float(df.bill.mean())}
