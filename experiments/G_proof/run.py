"""
G1 and G5 (first half) of the proof plan (plan.md): S4's main portfolio (2015/quarter, chrono_ridge, covariance B,
calibrated TE limit) on returns cleaned of impossible days (R1's exploration, main threshold), with
  real     the real views
  shuffle  the views permuted among the stocks that have one, on every rebalance date (seeds a..b-1)
and the daily active returns (BLM - band) written to results/<sample>/.

    python experiments/G_proof/run.py orig real
    python experiments/G_proof/run.py rep shuffle 0 15
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SAMPLE, MODE = sys.argv[1], sys.argv[2]
if SAMPLE in ('rep', 'g3'):              # before the imports: smallcap.build and S1-S4 read these when loaded
    os.environ['SMALLCAP_DB'] = str(REPO / 'data' / f'smallcap_{SAMPLE}.sqlite')
    os.environ['SMALLCAP_CACHE'] = str(REPO / 'data' / 'cache' / f'smallcap_{SAMPLE}')
    os.environ['SMALLCAP_RESULTS'] = f'results_{SAMPLE}'
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from events.returns import load_closes  # noqa: E402
from smallcap.build import open_db  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s4 = _load('s4_run', REPO / 'experiments' / 'S4_te_calibration' / 'run.py')
s3, bp, s1, x02 = s4.s3, s4.bp, s4.s1, s4.x02
START, FREQ, MODEL = '2015', 'quarter', 'chrono_ridge'
CLEAN = (3.0, -0.8)                      # R1 exploration's main threshold: +300% / -80%
OUT = HERE / 'results' / SAMPLE


class Inputs:
    """Covariance, band weights, sigma and the factors of correction B of each rebalance date (shared by all runs)."""

    def __init__(self, p, f, comp, closes, rets):
        self.p, self.f, self.comp, self.closes, self.rets = p, f, comp, closes, rets
        self.days = list(rets.index)
        self.index = {d: n for n, d in enumerate(self.days)}
        self.cache, self.s_cache = {}, {}

    def at(self, date):
        if date not in self.cache:
            u = self.comp.assign(m=date)
            u = u.join(s1.attach_float(u, self.f, 'm'))
            u = u[u.in_band == 1].set_index('symbol')
            i = self.index[date]
            r252 = self.rets.iloc[max(0, i - 251):i + 1][u.index.intersection(self.rets.columns)]
            r252 = r252.loc[:, (r252.notna().sum() >= bp.MIN_HISTORY) & self.closes.loc[date].reindex(r252.columns).notna()]
            cov = x02.lw(r252.fillna(0.0))
            w = np.exp(u.log_float.reindex(cov.index))
            self.cache[date] = (cov, w / w.sum(), r252.std())
        return self.cache[date]

    def factors(self, date, cov):
        if date not in self.s_cache:
            self.s_cache[date] = s3.scale_factors(self.p, date, self.index, self.days, cov)[0]
        return self.s_cache[date]


def run_b_cal(inp, rng=None):
    """S4's loop for B_cal (as in R1's explore.run_portfolios). rng: permute the views on every rebalance date.
    Returns the daily frame: date, port, bench, active (before costs, as S4's u), and on the first day of each holding
    period the amounts traded at the rebalance by the BLM (traded) and by the band (traded_band), so that the active
    return after a one-way cost c is active - c * (traded - traded_band)."""
    rets, index = inp.rets, inp.index
    dates, last = bp.schedule(inp.days, bp.STARTS[START], bp.FREQS[FREQ])
    held, held_band, z_hist, rows = None, None, [], []
    for k, d in enumerate(dates):
        end = dates[k + 1] if k + 1 < len(dates) else last
        cov0, w_mkt, sigma0 = inp.at(d)
        h = index[end] - index[d]
        s = inp.factors(d, cov0)
        cov = cov0.mul(s, axis=0).mul(s, axis=1)
        sigma = sigma0.reindex(cov.index) * s
        limit = s4.TE
        if len(z_hist) >= s4.MIN_DAYS:
            b_past = float(np.sqrt(np.mean(np.square(z_hist[-s4.HIST_DAYS:]))))
            limit = float(np.clip(s4.TE / b_past, *s4.L_RANGE))
        z = bp.views(inp.p, MODEL, d, set(cov.index))
        if rng is not None and len(z) > 1:
            z = pd.Series(rng.permutation(z.values), index=z.index)
        try:
            w, _ = s4.blm_weights(cov, w_mkt, sigma, z, max(h, 1), limit)
        except Exception:                # cvxpy's SolverError: hold the band (as R1's exploration)
            w = w_mkt
        act_w = w.reindex(cov.index).fillna(0.0) - w_mkt
        te_ex = float(np.sqrt(act_w.to_numpy() @ cov.to_numpy() @ act_w.to_numpy()))

        def turnover(new, old):
            old = pd.Series(dtype=float) if old is None else old
            idx = new.index.union(old.index)
            return float((new.reindex(idx).fillna(0) - old.reindex(idx).fillna(0)).abs().sum())
        traded, traded_band = turnover(w, held), turnover(w_mkt, held_band)
        period = rets.iloc[index[d] + 1:index[end] + 1][w.index].fillna(0.0)
        cur, bcur = w.copy(), w_mkt.reindex(w.index).fillna(0.0)
        first = True
        for day, r in period.iterrows():
            port, bench = float((cur * r).sum()), float((bcur * r).sum())
            if te_ex > 0:
                z_hist.append((port - bench) / te_ex)
            rows.append((day, port, bench, traded if first else 0.0, traded_band if first else 0.0))
            first = False
            cur, bcur = cur * (1 + r) / (1 + port), bcur * (1 + r) / (1 + bench)
        held, held_band = cur, bcur
    df = pd.DataFrame(rows, columns=['date', 'port', 'bench', 'traded', 'traded_band'])
    df['active'] = df.port - df.bench
    return df


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    conn = open_db()
    p = s3.panel(False)
    f = s1.float_known(conn)
    comp = pd.read_sql('SELECT cik, symbol FROM sc_sample', conn)
    closes = load_closes(conn, sorted(comp.symbol), '2014-12-01', s1.LATEST_PRICE)
    raw = closes.pct_change()
    rets = raw.where((raw <= CLEAN[0]) & (raw >= CLEAN[1]))
    inp = Inputs(p, f, comp, closes, rets)
    if MODE == 'real':
        df = run_b_cal(inp)
        df.to_csv(OUT / 'real.csv', index=False)
        ev = df[df.date >= s4.EVAL[START]]
        for c in (0.0, 0.003, 0.005, 0.010):
            net = ev.active - c * (ev.traded - ev.traded_band)
            print(SAMPLE, 'real, one-way cost', c, 'annual', round(net.mean() * 252, 4), flush=True)
    else:
        a, b = int(sys.argv[3]), int(sys.argv[4])
        frames = []
        for seed in range(a, b):
            df = run_b_cal(inp, np.random.default_rng(seed))
            frames.append(df[['date', 'active', 'traded', 'traded_band']].assign(seed=seed))
            print(SAMPLE, 'shuffle', seed, 'annual', round(df[df.date >= s4.EVAL[START]].active.mean() * 252, 4),
                  f'({time.time() - t0:.0f}s)', flush=True)
            pd.concat(frames).to_csv(OUT / f'shuffle_{a}_{b}.csv', index=False)
    print('done', f'({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
