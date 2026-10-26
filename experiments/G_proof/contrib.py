"""
Exploration after G1, G2 and G5 (plan.md, "寄与と G6"):
  contrib  the real BLM's active return (BLM - band, before costs) split by stock: sum over days of
           (BLM weight - band weight) * the stock's return, in total and by year (2017-)
  G6       on the BLM's schedule, universe and cleaned returns, the stocks with a view whose score is in the top 20%,
           weighted by float (as S1's monthly contest) or equally, against the same band
The views, covariance, calibration and cleaning are run.py's (the BLM is run.run_b_cal's loop, repeated here so that
the weights can be kept).

    python experiments/G_proof/contrib.py orig      -> results/orig/contrib.csv, results/orig/simple.csv
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = sys.argv[:2] + ['contrib']          # run.py reads the sample and a mode from sys.argv
import run as G  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TOP = 0.2


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    t0 = time.time()
    conn = G.open_db()
    p = G.s3.panel(False)
    f = G.s1.float_known(conn)
    comp = pd.read_sql('SELECT cik, symbol FROM sc_sample', conn)
    closes = G.load_closes(conn, sorted(comp.symbol), '2014-12-01', G.s1.LATEST_PRICE)
    raw = closes.pct_change()
    rets = raw.where((raw <= G.CLEAN[0]) & (raw >= G.CLEAN[1]))
    inp = G.Inputs(p, f, comp, closes, rets)
    index = inp.index
    dates, last = G.bp.schedule(inp.days, G.bp.STARTS[G.START], G.bp.FREQS[G.FREQ])
    z_hist, contrib, simple_rows = [], [], []
    held = {'blm': None, 'band': None, 'top_float': None, 'top_equal': None}
    for k, d in enumerate(dates):
        end = dates[k + 1] if k + 1 < len(dates) else last
        cov0, w_mkt, sigma0 = inp.at(d)
        h = index[end] - index[d]
        s = inp.factors(d, cov0)
        cov = cov0.mul(s, axis=0).mul(s, axis=1)
        sigma = sigma0.reindex(cov.index) * s
        limit = G.s4.TE
        if len(z_hist) >= G.s4.MIN_DAYS:
            limit = float(np.clip(G.s4.TE / float(np.sqrt(np.mean(np.square(z_hist[-G.s4.HIST_DAYS:])))), *G.s4.L_RANGE))
        z = G.bp.views(p, G.MODEL, d, set(cov.index))
        try:
            w, _ = G.s4.blm_weights(cov, w_mkt, sigma, z, max(h, 1), limit)
        except Exception:
            w = w_mkt
        act_w = w.reindex(cov.index).fillna(0.0) - w_mkt
        te_ex = float(np.sqrt(act_w.to_numpy() @ cov.to_numpy() @ act_w.to_numpy()))
        top = z[z >= z.quantile(1 - TOP)].index if len(z) else pd.Index([])
        wf = w_mkt.reindex(top)
        new = {'blm': w, 'band': w_mkt, 'top_float': wf / wf.sum() if len(top) else w_mkt,
               'top_equal': pd.Series(1.0 / len(top), index=top) if len(top) else w_mkt}
        traded = {}
        for key, nw in new.items():
            old = held[key] if held[key] is not None else pd.Series(dtype=float)
            idx = nw.index.union(old.index)
            traded[key] = float((nw.reindex(idx).fillna(0) - old.reindex(idx).fillna(0)).abs().sum())
        names = w.index.union(top)
        period = rets.iloc[index[d] + 1:index[end] + 1][names].fillna(0.0)
        cur = {key: nw.reindex(names).fillna(0.0) for key, nw in new.items()}
        first = True
        for day, r in period.iterrows():
            ret = {key: float((c * r).sum()) for key, c in cur.items()}
            if te_ex > 0:
                z_hist.append((ret['blm'] - ret['band']) / te_ex)
            c_i = (cur['blm'] - cur['band']) * r
            contrib.append(c_i[c_i != 0].rename(day))
            simple_rows.append({'date': day, **{key: v for key, v in ret.items()},
                                **{f'traded_{key}': (traded[key] if first else 0.0) for key in traded}})
            first = False
            cur = {key: c * (1 + r) / (1 + ret[key]) for key, c in cur.items()}
        held = cur
        print(G.SAMPLE, d, f'({time.time() - t0:.0f}s)', flush=True)
    sim = pd.DataFrame(simple_rows)
    sim.to_csv(G.OUT / 'simple.csv', index=False)
    c = pd.DataFrame(contrib).fillna(0.0)
    c.index.name = 'date'
    ev = c[c.index >= G.s4.EVAL[G.START]]
    by_year = ev.groupby(ev.index.str[:4]).sum().T
    out = pd.DataFrame({'total': ev.sum()}).join(by_year).sort_values('total', ascending=False)
    out.index.name = 'symbol'
    out.to_csv(G.OUT / 'contrib.csv')
    print('done', f'({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
