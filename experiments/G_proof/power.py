"""
G3's power, before its preregistration: is a sample of 400 companies (all that is left, data/smallcap_g3.sqlite) enough?
Random subsets of 400 companies of the samples already used (original and R1's), each run as plan.md's real portfolio
(S4's main BLM on cleaned returns, run.py), with the band of the same 400 companies. The share of subsets whose R1 test
(after a one-way cost of 0.3%) gives p < 0.05 estimates the power. Uses no data of the G3 sample.

    python experiments/G_proof/power.py orig power 0 5      # subsets 0..4 (seed 1000 + k)
Writes results/<sample>/power_<a>_<b>.csv (daily series of each subset).
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as G  # noqa: E402  (reads the sample from sys.argv[1] and sets its database)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

N_SUB = 400


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    t0 = time.time()
    a, b = int(sys.argv[3]), int(sys.argv[4])
    G.OUT.mkdir(parents=True, exist_ok=True)
    conn = G.open_db()
    p = G.s3.panel(False)
    f = G.s1.float_known(conn)
    comp = pd.read_sql('SELECT cik, symbol FROM sc_sample', conn)
    closes = G.load_closes(conn, sorted(comp.symbol), '2014-12-01', G.s1.LATEST_PRICE)
    raw = closes.pct_change()
    rets_all = raw.where((raw <= G.CLEAN[0]) & (raw >= G.CLEAN[1]))
    frames = []
    for k in range(a, b):
        sub = comp.iloc[np.sort(np.random.default_rng(1000 + k).choice(len(comp), N_SUB, replace=False))]
        cols = sorted(set(sub.symbol) & set(closes.columns))
        inp = G.Inputs(p, f, sub.reset_index(drop=True), closes[cols], rets_all[cols])
        df = G.run_b_cal(inp)
        frames.append(df.assign(subset=k))
        ev = df[df.date >= G.s4.EVAL[G.START]]
        net = ev.active - 0.003 * (ev.traded - ev.traded_band)
        print(G.SAMPLE, 'subset', k, 'annual after 0.3%', round(net.mean() * 252, 4), f'({time.time() - t0:.0f}s)', flush=True)
        pd.concat(frames).to_csv(G.OUT / f'power_{a}_{b}.csv', index=False)
    print('done', f'({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
