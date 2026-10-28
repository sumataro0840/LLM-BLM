"""
A worked example of the small-cap pipeline (S1 -> S4 -> experiment G, original 1,000 companies) on one rebalance
date, with every intermediate number, for three stocks. It reuses the project code (experiments/G_proof/run.py and
the modules it loads), replays the real run up to the date (for the calibrated TE limit), refits the ridge of the
quarter and checks that it reproduces the stored scores, then works the BLM through by hand for the three stocks.

    .venv/Scripts/python.exe 資料集/02_実験/計算例_LLMからポートフォリオまで/作成スクリプト/compute.py

Output: ../values.json (read by the TeX through make_tex.py).
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.argv = [sys.argv[0], 'orig', 'real']          # G_proof/run.py reads the sample and mode when loaded

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402

spec = importlib.util.spec_from_file_location('g_run', REPO / 'experiments' / 'G_proof' / 'run.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
s4, s3, bp, s1 = G.s4, G.s3, G.bp, G.s1
from blm.black_litterman import posterior_returns_general  # noqa: E402
from blm.optimizer import optimize_with_tracking_error  # noqa: E402
from events.returns import load_closes  # noqa: E402
from smallcap.build import open_db  # noqa: E402

D = '2024-03-28'            # the rebalance date of the example (last trading day of 2024-Q1)
TARGETS = (1.5, 0.0, -1.5)  # the z of the three stocks, roughly: high, middle, low


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    conn = open_db()
    p = s3.panel(False)
    f = s1.float_known(conn)
    comp = pd.read_sql('SELECT cik, symbol FROM sc_sample', conn)
    closes = load_closes(conn, sorted(comp.symbol), '2014-12-01', s1.LATEST_PRICE)
    raw = closes.pct_change()
    rets = raw.where((raw <= G.CLEAN[0]) & (raw >= G.CLEAN[1]))
    inp = G.Inputs(p, f, comp, closes, rets)
    index = inp.index

    # ---- replay G's real run up to D (run_b_cal), for the calibrated limit and a check against real.csv
    dates, last = bp.schedule(inp.days, bp.STARTS[G.START], bp.FREQS[G.FREQ])
    z_hist = []
    for k, d in enumerate(dates):
        end = dates[k + 1] if k + 1 < len(dates) else last
        cov0, w_mkt, sigma0 = inp.at(d)
        h = index[end] - index[d]
        s = inp.factors(d, cov0)
        cov = cov0.mul(s, axis=0).mul(s, axis=1)
        sigma = sigma0.reindex(cov.index) * s
        limit, b_past = s4.TE, None
        if len(z_hist) >= s4.MIN_DAYS:
            b_past = float(np.sqrt(np.mean(np.square(z_hist[-s4.HIST_DAYS:]))))
            limit = float(np.clip(s4.TE / b_past, *s4.L_RANGE))
        z = bp.views(p, G.MODEL, d, set(cov.index))
        try:
            w, _ = s4.blm_weights(cov, w_mkt, sigma, z, max(h, 1), limit)
        except Exception:
            w = w_mkt
        act_w = w.reindex(cov.index).fillna(0.0) - w_mkt
        te_ex = float(np.sqrt(act_w.to_numpy() @ cov.to_numpy() @ act_w.to_numpy()))
        period = rets.iloc[index[d] + 1:index[end] + 1][w.index].fillna(0.0)
        if d == D:
            break
        cur, bcur = w.copy(), w_mkt.reindex(w.index).fillna(0.0)
        for _, r in period.iterrows():
            port, bench = float((cur * r).sum()), float((bcur * r).sum())
            if te_ex > 0:
                z_hist.append((port - bench) / te_ex)
            cur, bcur = cur * (1 + r) / (1 + port), bcur * (1 + r) / (1 + bench)
        print('replayed', d, flush=True)
    assert d == D
    real = pd.read_csv(G.OUT / 'real.csv')
    first_day = period.index[0]
    stored = real[real.date == first_day].iloc[0]
    port0 = float((w * period.iloc[0]).sum())
    bench0 = float((w_mkt.reindex(w.index).fillna(0.0) * period.iloc[0]).sum())
    print('check first day', first_day, port0, stored.port, bench0, stored.bench, flush=True)

    # ---- the universe on D and the conditions to get a view
    u = comp.assign(m=D)
    u = u.join(s1.attach_float(u, f, 'm'))
    lo = (pd.Timestamp(D) - pd.Timedelta(days=bp.VIEW_DAYS)).strftime('%Y-%m-%d')
    cand = p[(p.event_date <= D) & (p.event_date > lo) & p.symbol.isin(cov.index)]
    latest = (cand[cand[G.MODEL].notna()].sort_values(['event_date', 'accepted_at'])
              .drop_duplicates('symbol', keep='last').set_index('symbol'))
    universe = {'n_sample': int(len(u)), 'n_band': int((u.in_band == 1).sum()), 'n_cov': int(len(cov.index)),
                'n_recent': int(cand.symbol.nunique()), 'n_view': int(len(z)), 'view_from': lo,
                'min_history': int(bp.MIN_HISTORY)}

    # ---- three stocks whose latest release was accepted in 2024-Q1 (same ridge model), z near the targets
    q1 = latest[latest.accepted_et.str[:7].between('2024-01', '2024-03')]
    picks = []
    for t in TARGETS:
        zz = z[q1.index.difference(picks)]
        picks.append((zz - t).abs().idxmin())
    rel = latest.loc[picks]

    # ---- the ridge of 2024-Q1, refitted as E5.rolling_ridge (through S1.rolling)
    X = s1.chrono('chrono_raw_v2015', 'mean_last', p, False)
    ok = np.isfinite(X).all(axis=1)
    fr = p.loc[ok, ['accession', 'accepted_et', 'window_end', 'abn_r_0_1']].reset_index(drop=True)
    fr['window_end'] = fr.window_end.fillna('9999-12-31')
    Xo = X[ok]
    tr = ((fr.window_end < '2024-01-01') & fr.abn_r_0_1.notna()).to_numpy()
    y_raw = fr.abn_r_0_1.to_numpy()[tr]
    q01, q99 = np.quantile(y_raw, [0.01, 0.99])
    y_c = np.clip(y_raw, q01, q99)
    y_mean, y_sd = float(y_c.mean()), float(y_c.std())
    y = (y_c - y_mean) / y_sd
    mu, sd = Xo[tr].mean(0), Xo[tr].std(0) + 1e-8
    model = RidgeCV(alphas=s1.e5.ALPHAS).fit((Xo[tr] - mu) / sd, y)
    beta, b0 = model.coef_, float(model.intercept_)
    train_dates = fr.accepted_et[tr].str[:10]
    acc_row = {a: i for i, a in enumerate(fr.accession)}
    stocks = []
    for sym in picks:
        r = rel.loc[sym]
        x = Xo[acc_row[r.accession]]
        xs = (x - mu) / sd
        contrib = (xs * beta).astype(np.float64)
        pred = float(b0 + contrib.sum())
        pred_lib = float(model.predict(xs[None, :])[0])
        top = np.argsort(-np.abs(contrib))[:3]
        stocks.append({
            'symbol': sym, 'accession': r.accession, 'event_date': r.event_date, 'accepted_et': r.accepted_et,
            'x_first': [float(v) for v in x[:4]], 'x_norm': float(np.linalg.norm(x)),
            'xs_first': [float(v) for v in xs[:4]], 'contrib_first': [float(v) for v in contrib[:4]],
            'contrib_rest': float(contrib[4:].sum()), 'contrib_sum': float(contrib.sum()),
            'top_dims': [[int(j), float(xs[j]), float(beta[j]), float(contrib[j])] for j in top],
            'pred': pred, 'pred_lib': pred_lib, 'pred_stored': float(r[G.MODEL]),
            'pred_return': pred * y_sd + y_mean,
        })
    ridge = {'quarter': '2024Q1', 'n_train': int(tr.sum()), 'train_from': train_dates.min(),
             'train_to': train_dates.max(), 'window_end_max': fr.window_end[tr].max(), 'dim': int(Xo.shape[1]),
             'alpha': float(model.alpha_), 'alphas': [float(s1.e5.ALPHAS[0]), float(s1.e5.ALPHAS[-1])],
             'n_alphas': int(len(s1.e5.ALPHAS)), 'min_train': int(s1.e5.MIN_TRAIN),
             'q01': float(q01), 'q99': float(q99), 'y_mean': y_mean, 'y_sd': y_sd, 'b0': b0,
             'mu_first': [float(v) for v in mu[:4]], 'sd_first': [float(v) for v in sd[:4]],
             'beta_first': [float(v) for v in beta[:4]], 'beta_norm': float(np.linalg.norm(beta)),
             'n_quarter': int((pd.PeriodIndex(fr.accepted_et.str[:10], freq='Q') == pd.Period('2024Q1')).sum())}

    # ---- Blom normal scores among all the stocks with a view
    sc = latest[G.MODEL].reindex(z.index)
    ranks = pd.Series(stats.rankdata(sc.values, method='average'), index=sc.index)
    N = len(sc)
    for st in stocks:
        rk = float(ranks[st['symbol']])
        qt = (rk - 0.375) / (N + 0.25)
        st.update({'rank': rk, 'N': N, 'quantile': qt, 'z': float(z[st['symbol']]),
                   'z_check': float(stats.norm.ppf(qt))})

    # ---- covariance and the band on D
    i = index[D]
    r252 = rets.iloc[max(0, i - 251):i + 1][cov0.index].fillna(0.0)
    shrink = float(LedoitWolf().fit(r252.values).shrinkage_)
    S3 = cov.loc[picks, picks]
    for st in stocks:
        sym = st['symbol']
        st.update({'sigma0': float(sigma0[sym]), 's': float(s[sym]), 'sigma': float(sigma[sym]),
                   'w_band_full': float(w_mkt[sym]), 'w_blm_full': float(w.get(sym, 0.0)),
                   'n_days': int(rets.iloc[max(0, i - 251):i + 1][sym].notna().sum())})

    # ---- the BLM by hand for a market of the three stocks (same formulas and code as S4.blm_weights)
    w3 = w_mkt[picks] / w_mkt[picks].sum()
    prior = bp.DELTA * S3 @ w3
    zz, sig = z[picks], sigma[picks]
    q = prior + bp.IC * sig * zz / np.sqrt(h)
    P = pd.DataFrame(np.eye(3), columns=picks)
    omega = bp.TAU * np.diag(np.diag(P.values @ S3.values @ P.values.T))
    post = posterior_returns_general(prior, S3, P, pd.Series(q.values), omega, bp.TAU)
    tS = bp.TAU * S3.values
    M = tS @ np.linalg.inv(tS + omega)          # with P = I: posterior = prior + M (q - prior)
    w_opt, status = optimize_with_tracking_error(post, S3, bp.DELTA, w3, limit / np.sqrt(252))
    a = (w_opt - w3).values
    te3 = float(np.sqrt(a @ S3.values @ a) * np.sqrt(252))
    w_free, _ = optimize_with_tracking_error(post, S3, bp.DELTA, w3, 10.0)
    a_free = (w_free - w3).values
    te_free = float(np.sqrt(a_free @ S3.values @ a_free) * np.sqrt(252))
    # the objective: expected return minus the risk penalty, at the band, the result and the unconstrained result
    def obj(v):
        v = np.asarray(v)
        return float(post.values @ v - bp.DELTA / 2 * v @ S3.values @ v)

    # ---- the holding period D -> end
    per = rets.iloc[index[D] + 1:index[end] + 1][picks].fillna(0.0)
    growth = (1 + per).prod() - 1
    port_ret = float((w_opt * growth).sum())
    band_ret = float((w3 * growth).sum())
    traded = float((w_opt - w3).abs().sum())

    out = {
        'date': D, 'end': end, 'h': int(h), 'stocks': stocks, 'ridge': ridge, 'universe': universe,
        'cov': {'shrink': shrink, 'S3_daily': S3.values.tolist(), 'S3_annual': (S3.values * 252).tolist(),
                'corr': (S3.values / np.outer(np.sqrt(np.diag(S3.values)), np.sqrt(np.diag(S3.values)))).tolist()},
        'calib': {'b_past': b_past, 'n_hist': len(z_hist), 'limit': limit, 'TE': s4.TE, 'range': list(s4.L_RANGE),
                  'hist_days': s4.HIST_DAYS, 'min_days': s4.MIN_DAYS},
        'params': {'IC': bp.IC, 'TAU': bp.TAU, 'DELTA': bp.DELTA, 'VIEW_DAYS': bp.VIEW_DAYS},
        'blm3': {'w3': w3.tolist(), 'prior_daily': prior.tolist(), 'q_daily': q.tolist(),
                 'tilt_daily': (q - prior).tolist(), 'omega_diag': np.diag(omega).tolist(), 'M': M.tolist(),
                 'post_daily': post.tolist(), 'w_opt': w_opt.tolist(), 'active': a.tolist(), 'te3': te3,
                 'status': str(status), 'w_free': w_free.tolist(), 'te_free': te_free,
                 'obj_band': obj(w3), 'obj_opt': obj(w_opt), 'obj_free': obj(w_free)},
        'full': {'te_ex_annual': te_ex * np.sqrt(252), 'n_held': int((w > 1e-6).sum()), 'n_universe': int(len(w))},
        'period': {'growth': growth.tolist(), 'port': port_ret, 'band': band_ret, 'active': port_ret - band_ret,
                   'traded': traded, 'cost': 0.003 * traded},
        'check': {'first_day': first_day, 'port_replay': port0, 'port_stored': float(stored.port),
                  'bench_replay': bench0, 'bench_stored': float(stored.bench)},
    }
    (HERE.parent / 'values.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
