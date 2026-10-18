"""
H05: experiment G asked again with the text only (T) and with the numbers only (N).

    python experiments/H05_text_vs_numbers/run.py ask [test]   # API calls (logged in results/answers_*.jsonl)
    python experiments/H05_text_vs_numbers/run.py evaluate

The prompts of G (過去のコードと記録/総合報告1の実験/premium_results.pkl) are split into their blocks; G itself is the
condition with both. The request settings are those of G (gpt-4o-mini, temperature 1.0, n = 30, one token,
top 20 log-probabilities; the score is read from the probabilities of the digits 1-5).
Decision rules: 資料集/02_実験/実験1_仮説の検証/LLM-BLM_実験仮説.pdf, chapter H05.
"""
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import data as D
from experiments.common import llm
from experiments.common.metrics import t_summary

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
G_RESULTS = D.REPO / '過去のコードと記録' / '総合報告1の実験' / 'premium_results.pkl'
SYSTEM = ("You are a financial analyst. Use only the information given below. "
          "Answer with a single digit from 1 to 5 and nothing else.")      # the system prompt of G
PARAMS = {'max_tokens': 1, 'temperature': 1.0, 'n': 30, 'logprobs': True, 'top_logprobs': 20}


def split_prompt(prompt):
    """Blocks of a G prompt: date, market numbers, outlook text, optional sector numbers, question."""
    blocks = prompt.split('\n\n')
    out = {'date': blocks[0], 'question': '\n\n'.join(b for b in blocks if b.startswith('Question:'))}
    for b in blocks[1:]:
        if b.startswith('[Market]'):
            out['market'] = b
        elif b.startswith('[Monthly outlook'):
            out['outlook'] = b
        elif b.startswith('[Sector:'):
            out['sector'] = b
    return out


def build(row, condition):
    b = split_prompt(row.prompt)
    if condition == 'T':
        parts = [b['date'], b['outlook'], b['question']]
    elif condition == 'N':
        parts = [b['date'], b['market'], *([b['sector']] if 'sector' in b else []), b['question']]
    else:
        raise ValueError(condition)
    return '\n\n'.join(parts)


def ask(test=False):
    OUT.mkdir(exist_ok=True)
    g = pd.read_pickle(G_RESULTS)
    if test:
        g = g[g.date == g.date.min()].head(3)
    api = llm.client()
    jobs = []
    for condition in ('T', 'N'):
        log = OUT / f'answers_{condition}.jsonl'
        done = llm.done_keys(log, ['date', 'kind', 'name'])
        jobs += [(condition, log, r) for r in g.itertuples() if (r.date, r.kind, r.name) not in done]
    print(len(jobs), 'requests to send', flush=True)
    if test:
        for c, _, r in jobs[:2]:
            print(f'--- {c}\n{build(r, c)}')

    def one(job):
        condition, log, r = job
        try:
            llm.chat(api, [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': build(r, condition)}],
                     log, {'condition': condition, 'date': r.date, 'kind': r.kind, 'name': r.name}, **PARAMS)
            return None
        except Exception as e:
            return f'{r.date} {r.name}: {str(e)[:120]}'
    with ThreadPoolExecutor(8) as pool:
        errors = [e for e in pool.map(one, jobs) if e]
    print('failed:', len(errors), errors[:3])
    for c in ('T', 'N'):
        print(c, llm.spending(OUT / f'answers_{c}.jsonl'))


def score_of(top):
    """Probability-weighted digit 1-5 and the probability mass on the digits."""
    p = np.array([math.exp(top[d]) if d in top else 0.0 for d in '12345'])
    mass = p.sum()
    p = p / mass
    return float(p @ np.arange(1, 6)), float(mass), float(np.sqrt(p @ np.arange(1, 6) ** 2 - (p @ np.arange(1, 6)) ** 2))


def realized(conn, dates):
    """Next month's premia with the definitions of G (premium_analyse.py)."""
    ends = [r[0] for r in conn.execute("SELECT MAX(date) FROM prices WHERE symbol='^GSPC' GROUP BY substr(date,1,7) ORDER BY 1")]
    nxt = {d: ends[ends.index(d) + 1] for d in dates}
    days = sorted(set(dates) | set(nxt.values()))
    px = pd.read_sql("SELECT date, symbol, adj_close FROM prices WHERE date IN (%s)" % ",".join("'%s'" % x for x in days),
                     conn).pivot(index='date', columns='symbol', values='adj_close')
    etf = dict(conn.execute("SELECT gics_sector, symbol FROM tickers WHERE gics_sector IS NOT NULL "
                            "AND symbol NOT IN (SELECT symbol FROM index_membership)").fetchall())
    real = []
    for d in dates:
        r = px.loc[nxt[d]] / px.loc[d] - 1
        m = pd.read_sql("SELECT symbol, metric, value FROM stock_metrics WHERE as_of_date = ?", conn, params=(d,)).pivot(
            index='symbol', columns='metric', values='value')
        m['cap'] = pd.read_sql("SELECT symbol, market_cap FROM index_weights WHERE date = ?", conn, params=(d,), index_col='symbol').market_cap

        def spread(col, high_is_top):
            x = m[col].dropna()
            x = x[x.index.intersection(r.dropna().index)]
            q = x.rank(pct=True)
            top, bot = r[q[q > 2 / 3].index].mean(), r[q[q <= 1 / 3].index].mean()
            return top - bot if high_is_top else bot - top
        tb = conn.execute("SELECT close FROM prices WHERE symbol='^IRX' AND date <= ? ORDER BY date DESC LIMIT 1", (d,)).fetchone()[0] / 100 / 12
        real += [(d, 'market', r['^GSPC'] - tb), (d, 'beta', spread('beta_1y', True)), (d, 'volatility', spread('volatility_30d', True)),
                 (d, 'value', spread('pe_trailing', False)), (d, 'momentum', spread('momentum_12m', True)),
                 (d, 'safety', spread('debt_to_equity', False)), (d, 'size', spread('cap', False))]
        real += [(d, s, r[e] - r['^GSPC']) for s, e in etf.items()]
    return pd.DataFrame(real, columns=['date', 'name', 'realized'])


def fisher_z_diff(r1, r2, n):
    """Two-sided p-value of r1 != r2 for two correlations over n points (treated as independent)."""
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(2 / (n - 3))
    return float(2 * stats.norm.sf(abs(z)))


def evaluate():
    conn = D.connect()
    g = pd.read_pickle(G_RESULTS)
    frames = []
    g_rows = g.assign(condition='G')
    g_rows['top'] = [{d: math.log(p) for d, p in zip('12345', json.loads(a)['probs']) if p > 0} for a in g.answer]
    frames.append(g_rows[['condition', 'date', 'kind', 'name', 'top', 'sp_1m', 'vix']])
    for c in ('T', 'N'):
        rows = llm.load_log(OUT / f'answers_{c}.jsonl')
        df = pd.DataFrame([{'condition': c, 'date': r['date'], 'kind': r['kind'], 'name': r['name'], 'top': r['top_logprobs'][0]}
                           for r in rows])
        frames.append(df.merge(g[['date', 'kind', 'name', 'sp_1m', 'vix']], on=['date', 'kind', 'name']))
    t = pd.concat(frames, ignore_index=True)
    t[['score', 'mass', 'answer_sd']] = pd.DataFrame([score_of(x) for x in t.top], index=t.index)
    t = t.merge(realized(conn, sorted(g.date.unique())), on=['date', 'name'])
    t.drop(columns=['top']).to_csv(OUT / 'scores.csv', index=False)

    summary = {}
    for c in ('G', 'T', 'N'):
        s = t[t.condition == c]
        f, sec = s[s.kind == 'factor'], s[s.kind == 'sector']
        mk = f[f.name == 'market'].sort_values('date')
        trace = stats.spearmanr(mk.score, mk.sp_1m)
        trace_vix = stats.spearmanr(mk.score, mk.vix)
        ic = sec.groupby('date').apply(lambda x: stats.spearmanr(x.score, x.realized)[0])
        factors = {}
        for name, x in f.groupby('name'):
            call = np.sign(x.score - 3)
            factors[name] = {'mean_score': float(x.score.mean()), 'sd_over_months': float(x.score.std()),
                             'months_above_3': int((x.score > 3).sum()), 'months_realized_positive': int((x.realized > 0).sum()),
                             'hit_rate': float((np.sign(x.realized) == call).mean()),
                             'rank_corr_realized': float(stats.spearmanr(x.score, x.realized)[0]) if x.score.nunique() > 1 else None,
                             'rank_corr_sp_1m': float(stats.spearmanr(x.score, x.sp_1m)[0]) if x.score.nunique() > 1 else None}
        summary[c] = {
            'market_score_vs_sp_1m': {'rho': float(trace[0]), 'p': float(trace[1])},
            'market_score_vs_vix': {'rho': float(trace_vix[0]), 'p': float(trace_vix[1])},
            'sector_ic': t_summary(ic),
            'sector_score_sd_within_month': float(sec.groupby('date').score.std().mean()),
            'sector_score_sd_over_months': float(sec.groupby('name').score.std().mean()),
            'sector_score_mean': float(sec.score.mean()),
            'probability_mass_min': float(s.mass.min()),
            'factors': factors,
        }
    n = 30
    tr_t, tr_g = summary['T']['market_score_vs_sp_1m'], summary['G']['market_score_vs_sp_1m']
    trace_gone = tr_t['p'] >= 0.05 and fisher_z_diff(tr_t['rho'], tr_g['rho'], n) < 0.05 and tr_t['rho'] < tr_g['rho']
    ic_t = summary['T']['sector_ic']
    if trace_gone and ic_t['mean'] > 0 and ic_t['p'] < 0.05:
        verdict = 'supported (no tracing, and the text gives sector calls that hit)'
    elif trace_gone:
        verdict = 'tracing gone, but hitting is not shown'
    else:
        verdict = 'tracing not removed'
    summary['primary'] = {'sector_ic_T': ic_t, 'trace_T_vs_G_fisher_p': fisher_z_diff(tr_t['rho'], tr_g['rho'], n),
                          'trace_gone': bool(trace_gone), 'verdict': verdict}
    summary['spending'] = {c: llm.spending(OUT / f'answers_{c}.jsonl') for c in ('T', 'N')}
    summary['deviation_from_design'] = ('The design says "temperature 0 (same as G)"; G used temperature 1.0 with n = 30 '
                                        'and the probabilities of the first token. G was followed, as the design intended.')
    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))


if __name__ == '__main__':
    if sys.argv[1] == 'ask':
        ask(test=len(sys.argv) > 2 and sys.argv[2] == 'test')
    else:
        evaluate()
