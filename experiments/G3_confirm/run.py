"""
G3 (preregistration.md): does the calibrated small-cap BLM beat its band after costs, on the 400 companies used by
no earlier experiment (data/smallcap_g3.sqlite)? Run once, after S1's and S2's run.py have built their panels of the
G3 sample (start.ps1 runs the three in order).

    python experiments/G3_confirm/run.py --dry    # noise in place of every return (results/dry/)
    python experiments/G3_confirm/run.py          # once, after preregistration.md and this file are committed

The portfolio is experiments/G_proof/run.py's run_b_cal (S4's main BLM on returns cleaned of impossible days).
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.argv = [sys.argv[0], 'g3', 'g3'] + sys.argv[1:]      # G_proof/run.py reads the sample from sys.argv[1]
sys.path.insert(0, str(REPO / 'experiments' / 'G_proof'))
import run as G  # noqa: E402
import analyze as A  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

spec = importlib.util.spec_from_file_location('r1_run', REPO / 'experiments' / 'R1_replication' / 'run.py')
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)

MAIN_COST = 0.003
COSTS = (0.0, 0.003, 0.005, 0.010)
N_SHUFFLE = 100
EVAL = '2017-01-01'


def net(df, c):
    return df.active - c * (df.traded - df.traded_band)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = argparse.ArgumentParser()
    a.add_argument('--dry', action='store_true')
    a.add_argument('--shuffles', type=int, default=N_SHUFFLE)   # fewer only to check the dry run
    args = a.parse_args(sys.argv[3:])
    out_dir = HERE / 'results' / ('dry' if args.dry else '')
    if not args.dry and args.shuffles != N_SHUFFLE:
        raise SystemExit('本番は偽物 100 回（事前登録）')
    if not args.dry and (HERE / 'results' / 'summary.json').exists():
        raise SystemExit('results/summary.json があります。G3 は実行済みです（事前登録どおり 1 回だけ）')
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    conn = G.open_db()
    p = G.s3.panel(args.dry)
    f = G.s1.float_known(conn)
    comp = pd.read_sql('SELECT cik, symbol FROM sc_sample', conn)
    closes = G.load_closes(conn, sorted(comp.symbol), '2014-12-01', G.s1.LATEST_PRICE)
    raw = closes.pct_change()
    if args.dry:
        raw = pd.DataFrame(np.random.default_rng(0).normal(0, 0.03, raw.shape), index=raw.index,
                           columns=raw.columns).where(raw.notna())
    rets = raw.where((raw <= G.CLEAN[0]) & (raw >= G.CLEAN[1]))
    inp = G.Inputs(p, f, comp, closes, rets)
    real = G.run_b_cal(inp)
    real.to_csv(out_dir / 'real.csv', index=False)
    ev = real[real.date >= EVAL].reset_index(drop=True)
    res = {'commit': commit, 'dry': args.dry, 'companies': int(len(comp)), 'days': int(len(ev)),
           'n_returns_removed': int((raw.notna() & rets.isna()).sum().sum())}
    res['primary'] = r1.mean_test(net(ev, MAIN_COST).to_numpy())
    res['primary']['supported'] = bool(res['primary']['p_one_sided'] < 0.05)
    print('primary (one-way 0.3%)', res['primary'], f'({time.time() - t0:.0f}s)', flush=True)
    res['costs'] = {f'cost_{c}': r1.mean_test(net(ev, c).to_numpy()) for c in COSTS}
    reb = ev[ev.traded > 0]
    res['turnover'] = {'blm': float(reb.traded.mean()), 'band': float(reb.traded_band.mean())}
    # secondary: shuffled views
    fakes, fake_days = [], []
    for seed in range(args.shuffles):
        df = G.run_b_cal(inp, np.random.default_rng(seed))
        df = df[df.date >= EVAL]
        fakes.append(float(net(df, MAIN_COST).mean() * 252))
        fake_days.append(net(df, MAIN_COST).to_numpy())
        if seed % 10 == 9:
            print('shuffle', seed, f'({time.time() - t0:.0f}s)', flush=True)
    real_ann = float(net(ev, MAIN_COST).mean() * 252)
    fk = np.array(fakes)
    res['shuffled_views'] = {'real_annual': real_ann, 'fake_mean': float(fk.mean()), 'fake_q05': float(np.quantile(fk, 0.05)),
                             'fake_q95': float(np.quantile(fk, 0.95)), 'fake_ge_real': int((fk >= real_ann).sum()),
                             'p': float((1 + (fk >= real_ann).sum()) / (1 + len(fk)))}
    # secondary: factors
    y = net(ev, MAIN_COST).set_axis(ev.date)
    res['factors'] = A.g2(y, A.factors())
    fake_mean = pd.Series(np.mean(fake_days, axis=0), index=ev.date)
    res['factors_real_minus_fake_mean'] = A.g2(y - fake_mean, A.factors())
    res['seconds'] = round(time.time() - t0)
    (out_dir / 'summary.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
