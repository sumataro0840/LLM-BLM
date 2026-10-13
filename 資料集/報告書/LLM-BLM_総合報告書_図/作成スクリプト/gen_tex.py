"""Auto-generated TeX snippets (tables, listings) -> build/inc_*.tex"""
import sys, re, json, html, ast, glob, subprocess, sqlite3, numpy as np, pandas as pd
R = '/Users/kazuki/develop/工房/LLM-BLM/'; K = '/Users/kazuki/develop/工房/'; E = K + 'LLM-BLM_実験/'
B = '/private/tmp/claude-501/-Users-kazuki-develop---/9d2180f2-ca7f-42f9-a43d-a8ea8ccec7c4/scratchpad/report/build/'
T = '/private/tmp/claude-501/-Users-kazuki-develop---/9d2180f2-ca7f-42f9-a43d-a8ea8ccec7c4/scratchpad/report/txt/'
sys.path.insert(0, R)
N = json.load(open(B + 'numbers.json'))
conn = sqlite3.connect(f"file:{R}data/blm.sqlite?mode=ro", uri=True)

def esc(s):
    s = str(s)
    for a, b in [('\\', '\\textbackslash{}'), ('&', '\\&'), ('%', '\\%'), ('$', '\\$'), ('#', '\\#'), ('_', '\\_'), ('{', '\\{'), ('}', '\\}'), ('~', '\\textasciitilde{}'), ('^', '\\textasciicircum{}'), ('<', '\\textless{}'), ('>', '\\textgreater{}'), ('|', '\\textbar{}')]:
        s = s.replace(a, b)
    return s.replace('\\textbackslash\\{\\}', '\\textbackslash{}')
def code(s):
    t = esc(s).replace('\\_', '\\_\\allowbreak{}')
    t = re.sub(r'([/?=.])(?=[A-Za-z0-9\\{])', r'\1\\allowbreak{}', t) if len(str(s)) > 28 else t
    return '\\texttt{' + t.replace('≤', '$\\leq$').replace('≥', '$\\geq$') + '}'
def out(name, s): open(B + f'inc_{name}.tex', 'w').write(s); print(name, len(s))
def longtable(spec, head, rows, size='\\small', caption=None, label=None):
    s = '{' + size + '\n\\begin{longtable}{' + spec + '}\n'
    if caption: s += f'\\caption{{{caption}}}' + (f'\\label{{{label}}}' if label else '') + '\\\\\n'
    h = '\\toprule\n' + ' & '.join(f'\\textbf{{{x}}}' for x in head) + ' \\\\\n\\midrule\n'
    s += h + '\\endfirsthead\n' + h + '\\endhead\n\\bottomrule\n\\endfoot\n'
    for r in rows: s += ' & '.join(str(x) for x in r) + ' \\\\\n'
    return s + '\\end{longtable}\n}\n'
def listing(text, caption=None, lang=None, label=None):
    opt = ([f'language={lang}'] if lang else []) + ([f'caption={{{caption}}}'] if caption else []) + ([f'label={{{label}}}'] if label else [])
    return '\\begin{lstlisting}' + (f'[{",".join(opt)}]' if opt else '') + '\n' + text.rstrip('\n') + '\n\\end{lstlisting}\n'

def tabular(spec, head, rows, size='\\small', fit=False):
    s = '{' + size + '\n' + ('\\resizebox{\\linewidth}{!}{%\n' if fit else '') + '\\begin{tabular}{' + spec + '}\n\\toprule\n' + ' & '.join(head) + ' \\\\\n\\midrule\n'
    for r in rows: s += ' & '.join(str(x) for x in r) + ' \\\\\n'
    return s + '\\bottomrule\n\\end{tabular}' + ('}' if fit else '') + '\n}\n'
pct = lambda x, d=1: '---' if pd.isna(x) else f'{x * 100:+.{d}f}\\%'
pctu = lambda x, d=1: '---' if pd.isna(x) else f'{x * 100:.{d}f}\\%'

# ---------- table / view definitions from the DB design HTML
src = open(K + 'LLM-BLM_DB設計.html').read()
def inline(h):
    h = re.sub(r'<code>(.*?)</code>', lambda m: '\x00C' + m.group(1) + '\x00E', h, flags=re.S)
    h = re.sub(r'<b>(.*?)</b>', lambda m: '\x00B' + m.group(1) + '\x00E', h, flags=re.S)
    h = re.sub(r'<span class="pill[^"]*">(.*?)</span>', lambda m: m.group(1) + '\x00S', h)
    h = html.unescape(re.sub(r'<[^>]+>', '', h))
    parts = re.split(r'(\x00C.*?\x00E|\x00B.*?\x00E|\x00S)', h, flags=re.S); o = ''
    for p in parts:
        if p.startswith('\x00C'): o += code(html.unescape(p[2:-2]))
        elif p.startswith('\x00B'): o += '\\textbf{' + esc(p[2:-2]) + '}'
        elif p == '\x00S': o += '、'
        else: o += esc(p).replace('≤', '$\\leq$').replace('≥', '$\\geq$')
    return re.sub(r'、(\s*(&|$))', r'\1', o).strip().rstrip('、')
cards = re.findall(r'<div class="card tcard"><h3>(\S+) <span class="role">(.*?)</span></h3>(.*?)(?=<div class="card tcard">|</section>)', src, flags=re.S)
tabs, views = '', ''
for name, role, body in cards:
    body = re.sub(r'<details>.*?</details>', '', body, flags=re.S)
    s = f'\\subsection*{{{code(name)}　{inline(role)}}}\n'
    for tb in re.findall(r'<table>(.*?)</table>', body, flags=re.S):
        trs = re.findall(r'<tr>(.*?)</tr>', tb, flags=re.S); head = [inline(x) for x in re.findall(r'<th[^>]*>(.*?)</th>', trs[0], flags=re.S)]
        rows = [[inline(x) for x in re.findall(r'<td[^>]*>(.*?)</td>', tr, flags=re.S)] for tr in trs[1:]]
        n = len(head)
        if head[0] == 'カラム' and n == 4: spec = 'p{3.3cm}p{1.4cm}p{3.0cm}p{6.4cm}'
        elif head[0] == 'カラム' and n == 3: spec = 'p{3.6cm}p{1.8cm}p{8.9cm}'
        elif n == 3: spec = 'p{4.6cm}p{6.6cm}p{3.1cm}'
        else: spec = 'p{' + f'{14.6 / n:.1f}' + 'cm}' * n
        rows = [r for r in rows if len(r) == n]
        s += longtable(spec, head, rows, '\\footnotesize')
    for p in re.findall(r'<p class="muted">(.*?)</p>', body, flags=re.S): s += '{\\small ' + inline(p) + '\\par}\n\\medskip\n'
    if 'CREATE VIEW' in src[src.find(f'<h3>{name} '):src.find(f'<h3>{name} ') + len(body) + 4000] and name in N.get('views_list', []): pass
    (views if name in ('market_caps', 'index_weights', 'fundamentals_quarterly', 'llm_view_stats', 'portfolio_returns', 'sector_metric_scores', 'stock_metric_scores') else tabs).__class__  # no-op
    if name in ('market_caps', 'index_weights', 'fundamentals_quarterly', 'llm_view_stats', 'portfolio_returns', 'sector_metric_scores', 'stock_metric_scores'): views += s
    else: tabs += s
out('tables_def', tabs); out('views_def', views); print('cards', [c[0] for c in cards])

# ---------- listings
lst = '\\begin{lstlisting}[language=SQL]\n' + open(R + 'marketdb/db/schema.sql').read().rstrip() + '\n\\end{lstlisting}\n'
out('schema', lst)
for f, nm in [('blm/equilibrium.py', 'code_equilibrium'), ('blm/black_litterman.py', 'code_bl'), ('blm/optimizer.py', 'code_optimizer'), ('blm/strategies.py', 'code_strategies'), ('marketdb/transform/stock.py', 'code_stock'), ('reports/items.py', 'code_items'), ('blm/inputs.py', 'code_inputs')]:
    out(nm, listing(open(R + f).read(), lang='Python'))

# ---------- settings (.env.example)
rows, sect, com = [], '', []
for line in open(R + '.env.example').read().split('\n'):
    if line.startswith('# ----'): sect = line.strip('# -').strip(); com = []
    elif line.startswith('#'): com.append(line[1:].strip())
    elif '=' in line:
        k, v = line.split('=', 1)
        if k == 'OPENAI_API_KEY': v = '（各自の API キー。Git には入れない）'
        rows.append((code(k), code(v) if v and '（' not in v else esc(v) or '（空）', esc(' '.join(com)))); com = []
out('env', longtable('p{4.6cm}p{3.6cm}p{6.2cm}', ['設定名', '例（.env.example の値）', '意味'], rows, '\\footnotesize'))

# ---------- row counts
jp = {'tickers': '銘柄・指数・セクターETFの一覧', 'prices': '日次の株価', 'shares_outstanding': '発行済み株式数の履歴', 'index_changes': 'S\\&P500 の銘柄入れ替え', 'index_membership': '銘柄ごとの在籍期間', 'financial_items': '財務諸表の項目', 'financial_facts': '財務諸表の値',
      'llm_runs': 'LLM の実行', 'llm_views': 'LLM の回答', 'portfolio_runs': '比率の計算', 'portfolio_weights': '比率', 'sector_metrics': 'セクターごとの指標', 'stock_metrics': '銘柄ごとの指標', 'reports': '月次レポートのファイル', 'report_runs': 'レポートの分析', 'report_values': 'レポートから読み取った値'}
order = list(jp); out('rows', longtable('p{4.2cm}p{6.2cm}r', ['テーブル', '内容', '行数'], [(code(k), jp[k], f"{N['rows'][k]:,}") for k in order], '\\small'))

# ---------- performance
lab = {'market_cap': '時価総額加重', 'equal_weight': '等金額', 'mean_variance_original': '平均分散（元の計算）', 'black_litterman_original': 'BL（元の計算）', 'mean_variance': '平均分散（直した計算）', 'black_litterman': 'BL（直した計算）', 'sp500_index': '参考：S\\&P500 指数（配当なし）'}
P = N['perf']; rows = []
for k in lab:
    p = P[k]; rows.append((lab[k], pct(p['annual_return']), pctu(p['annual_volatility']), f"{p['sharpe_ratio']:.2f}", pct(p['max_drawdown']), pct(p['worst_month']), '---' if k == 'market_cap' else pct(p['annual_active_return']), '---' if k == 'market_cap' else f"{int(p['months_above_benchmark'])}/30"))
out('perf_tabular', tabular('lrrrrrrr', ['戦略', '年率リターン', '年率リスク', 'シャープ', '最大下落率', '最悪の月', '対 時価総額', '上回った月'], rows, '\\footnotesize', fit=True))
rows = [(lab[k], f"{P[k]['effective_holdings']:.1f}", pctu(P[k]['max_weight'], 0), pctu(P[k]['turnover'], 0)) for k in lab if k != 'sp500_index']
out('concentration_tabular', tabular('lrrr', ['戦略', '実質的な銘柄数', '最大の比率（平均）', '毎月入れ替わる割合'], rows, '\\small'))

# ---------- monthly tables
m = pd.read_csv(B + 'views_monthly.csv')
rows = [(r.as_of_date, r.n, pct(r.view_mean, 2), pctu(r.share_pos, 0), pct(r.input_mean, 2), f'{r.corr_input:.2f}', f'{r.slope:.2f}', f'{r.rank_sd_vol:.2f}', pct(r.realized_daily, 2), f'{r.rank_ic:+.2f}') for r in m.itertuples()]
out('views_monthly', longtable('lrrrrrrrrr', ['判断日', '銘柄数', '予想の平均', 'プラスの割合', '入力の平均', '入力との相関', '傾き', 'ばらつきと値動き', '翌月の実際', '順位相関'], rows, '\\scriptsize'))
r = pd.read_csv(B + 'monthly_returns.csv', index_col=0)
cols = ['market_cap', 'equal_weight', 'mean_variance_original', 'black_litterman_original', 'mean_variance', 'black_litterman', 'sp500_index']
rows = [(d, *[pct(x[c]) for c in cols]) for d, x in r.iterrows()]
out('monthly_returns', longtable('lrrrrrrr', ['判断日', '時価総額', '等金額', '平均分散（元）', 'BL（元）', '平均分散（直）', 'BL（直）', '指数'], rows, '\\scriptsize'))
rp = pd.read_csv(B + 'reports.csv'); f0 = lambda x: '---' if pd.isna(x) else f'{x:,.0f}'; f1 = lambda x: '---' if pd.isna(x) else f'{x:.1f}'; fs = lambda x: '---' if pd.isna(x) else f'{x:+.0f}'
rows = [(q.report_month, q.published_date, int(q.n_pages), fs(q.us_economy_stance), fs(q.us_equity_stance), fs(q.us_bond_stance), f0(q.sp500_latest), f0(q.sp500_target_this_year), f0(q.sp500_target_next_year), f1(q.us_gdp_growth_this_year), f1(q.us_gdp_growth_next_year)) for q in rp.itertuples()]
out('reports', longtable('llrrrrrrrrr', ['号', '公表日', 'ページ', '経済', '株式', '債券', 'S\\&P 直近', '当年末', '翌年末', 'GDP 当年', 'GDP 翌年'], rows, '\\scriptsize'))
sm = pd.read_csv(B + 'sector_metrics.csv'); g = sm[sm.as_of_date == '2026-05-29']; v = g.pivot(index='gics_sector', columns='metric', values='value'); z = g.pivot(index='gics_sector', columns='metric', values='z_score')
cell = lambda s, k, f: f.format(v.loc[s, k]) + f' ({z.loc[s, k]:+.2f})'
rows = [(esc(s), cell(s, 'momentum_3m', '{:+.3f}'), cell(s, 'momentum_6m', '{:+.3f}'), cell(s, 'momentum_12m', '{:+.3f}'), cell(s, 'volatility_30d', '{:.3f}'), cell(s, 'pe_trailing', '{:.1f}'), cell(s, 'pb', '{:.1f}'), cell(s, 'ps', '{:.1f}')) for s in v.index]
out('sector_example_tabular', tabular('lrrrrrrr', ['セクター', 'モメンタム 3 か月', '6 か月', '12 か月', 'ボラティリティ', 'PER', 'PBR', 'PSR'], rows, '\\footnotesize', fit=True))
ex = pd.read_csv(B + 'stock_example.csv'); vv = ex.pivot(index='metric', columns='symbol', values='value'); pp = ex.pivot(index='metric', columns='symbol', values='sector_percentile')
def sc(mt, s):
    if mt not in vv.index or pd.isna(vv.loc[mt, s]): return 'n/a'
    return f'{vv.loc[mt, s]:.3g}' + ('' if pd.isna(pp.loc[mt, s]) else f' ({pp.loc[mt, s]:.0f})')
mo = ['momentum_12m', 'return_1m', 'relative_momentum_12m', 'volatility_30d', 'beta_1y', 'pe_trailing', 'pb', 'revenue_growth_yoy', 'net_income_growth_yoy', 'net_margin', 'roe', 'debt_to_equity', 'fundamentals_age_days', 'filing_age_days']
out('stock_example_tabular', tabular('lrrrr', ['指標', 'NVDA（半導体）', 'JPM（銀行）', 'PLD（REIT）', 'XOM（石油）'], [(code(k), sc(k, 'NVDA'), sc(k, 'JPM'), sc(k, 'PLD'), sc(k, 'XOM')) for k in mo], '\\small'))
smc = N['stock_metrics']; out('stock_counts', longtable('p{5cm}rr', ['指標', '行数', '判断日の数'], [(code(k), f"{smc[k]['n']:,}", smc[k]['dates']) for k in mo], '\\small'))

# ---------- excluded tickers
txt = open(T + 'LLM-BLM_データの扱い方.txt').read(); seg = txt[txt.find('銘柄 | 会社 | 除外の理由'):txt.find('## 7DB')]
rows = [[esc(c.strip()) for c in l.split('|')[:5]] for l in seg.split('\n')[1:] if l.count('|') >= 5]
out('excluded', longtable('lp{3.6cm}p{4.0cm}lp{3.0cm}', ['銘柄', '会社', '除外の理由', '除外日', '背景'], rows, '\\scriptsize')); print('excluded', len(rows))

# ---------- commits and files
log = subprocess.run(['git', '-C', R, 'log', '--since=2026-10-01', '--format=%h|%ad|%s', '--date=short'], capture_output=True, text=True).stdout.strip().split('\n')[::-1]
out('commits', longtable('llp{10.5cm}', ['コミット', '日付', '内容'], [(code(a), b, esc(c)) for a, b, c in (l.split('|', 2) for l in log)], '\\footnotesize'))
rows = []
for f in sorted(glob.glob(R + '*.py') + glob.glob(R + 'marketdb/**/*.py', recursive=True) + glob.glob(R + 'llm/*.py') + glob.glob(R + 'blm/*.py') + glob.glob(R + 'reports/*.py')):
    s = open(f).read()
    if len(s.strip().split('\n')) < 2: continue
    try: doc = (ast.get_docstring(ast.parse(s)) or '').split('\n')[0]
    except Exception: doc = ''
    rows.append((code(f.replace(R, '')), len(s.split('\n')) - 1, esc(doc)))
out('files', longtable('p{5.2cm}rp{8.2cm}', ['ファイル', '行数', '役割（ファイル冒頭の説明）'], rows, '\\footnotesize')); print('total lines', sum(r[1] for r in rows))

# ---------- prompts
from llm.prompts import SYSTEM_PROMPT, make_user_prompt
dd = lambda s: '\n'.join(l.strip() for l in s.strip().split('\n'))
up = dd(make_user_prompt({'ticker': 'AAPL', 'Security': 'Apple Inc.', 'GICS Sector': 'Information Technology', 'GICS Sub-Industry': 'Technology Hardware, Storage & Peripherals', 'pct_change': '[0.0016, 0.0078, -0.0071, 0.0124, ...]'}))
out('prompt_run1', listing(dd(SYSTEM_PROMPT), 'システムプロンプト（実行番号 1）', label='lst:run1') + listing(up, 'ユーザープロンプトの例（日次リターンは約20個）'))
from reports.items import ITEMS
out('prompt_report', listing(N['report_prompt'], 'レポートの読み取り（実行番号 2）のシステムプロンプトとユーザープロンプトの型'))
out('report_items', longtable('p{4.4cm}p{1.2cm}p{8.6cm}', ['項目名', '型', '説明（LLM に渡す英文）'], [(code(i.name), i.type, esc(i.description)) for i in ITEMS], '\\scriptsize'))
res = pd.read_pickle(E + 'industry_results.pkl'); a = res[(res.group == 'reit') & (res.symbol == 'PLD') & (res.date == '2025-06-30')].set_index('cond').prompt
sys.path.insert(0, E); import industry_exp as ie
out('prompt_expD', listing(ie.SYSTEM, '実験 D のシステムプロンプト') + listing(a['with'], '実験 D のユーザープロンプト（指標を渡す条件。PLD、判断日 2025-06-30）', label='lst:expD') + listing(a['na'].split('[Fundamentals]')[1].strip(), '同じ銘柄で、問題の指標を n/a にした条件（Fundamentals の部分）'))
tt = json.load(open(E + 'turns_texts.json')); q1 = next(x for x in tt['turn1'] if x['sub'] == 'Retail REITs'); q2 = next(x for x in tt['turn2'] if x['sub'] == 'Aerospace & Defense' and x['date'] == '2025-06-30')
import turns_exp as te
out('prompt_expF', listing(q1['question'], 'ターン 1 の質問（Retail REITs）', label='lst:expF') + listing(q1['answer'], 'ターン1の回答') + listing(q2['question'], 'ターン2の質問（Aerospace \\& Defense、2025-06-30）') + listing(q2['answer'], 'ターン2の回答') + listing(te.ASK_NUMBER + '\n\n' + te.ASK_RATING, 'ターン3の最後に付ける指示（上：数字、下：5段階）'))
pr = pd.read_pickle(E + 'premium_scored.pkl'); ex1 = pr[(pr.date == '2025-06-30') & (pr.name == 'value')].prompt.iloc[0]; ex2 = pr[(pr.date == '2025-06-30') & (pr.name == 'Energy')].prompt.iloc[0]
import premium_exp as pe
out('prompt_expG', listing(pe.SYSTEM, '実験 G のシステムプロンプト') + listing(ex1, '実験 G のユーザープロンプト（割安さ、判断日 2025-06-30）') + listing(ex2[ex2.find('[Sector'):], 'セクターの問いで足される部分（Energy）'))
pick = ['Retail REITs', 'Telecom Tower REITs', 'Industrial REITs', 'Office REITs', 'Data Center REITs', 'Health Care REITs', 'Aerospace & Defense', 'Building Products', 'Electrical Components & Equipment', 'Human Resource & Employment Services', 'Passenger Airlines', 'Rail Transportation']
rows = [(esc(x['sector']), esc(x['sub']), esc(x['answer'])) for x in sorted(tt['turn1'], key=lambda x: (x['sector'], x['sub'])) if x['sub'] in pick]
print('turn1 rows', len(rows), sorted(set(x['sub'] for x in tt['turn1'])))
out('turn1', longtable('p{1.9cm}p{2.9cm}p{9.6cm}', ['セクター', 'サブ業種', 'ターン1の回答（全文）'], rows, '\\scriptsize'))

# ---------- experiment G tables
f = pr[pr.kind == 'factor']; jpn = {'market': '市場', 'beta': 'ベータ', 'volatility': '変動の大きさ', 'value': '割安さ', 'momentum': '勢い', 'safety': '安全性', 'size': '規模'}; rows = []
for nme, gq in f.groupby('name', sort=False):
    call = np.sign(gq.score - 3)
    rows.append((jpn[nme], f'{gq.score.mean():.2f}', int((gq.score > 3).sum()), int((gq.realized > 0).sum()), f'{(np.sign(gq.realized) == call).mean():.2f}', f'{gq.score.rank().corr(gq.realized.rank()):+.2f}', f'{gq.score.rank().corr(gq.sp_1m.rank()):+.2f}', f'{gq.score.rank().corr(gq.vix.rank()):+.2f}'))
out('expG_factors_tabular', tabular('lrrrrrrr', ['リスク', '点数の平均', '点数 $>3$ の月', '実際 $>0$ の月', '的中率', '点数と実際', '点数と直近リターン', '点数と VIX'], rows, '\\small', fit=True))
print('done')
